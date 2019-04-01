# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 00:04:24 2018

@author: 47532
"""

import numpy as np
import pandas as pd
import ast
import cv2
from PIL import Image
#%% ========= Define a function to convert raw strokes into 256x256 images   =============================
def draw_cv2(raw_strokes, size=256, lw=3):   
    step = len(raw_strokes)
    max_x = max( [max(x)  for x, y in raw_strokes ] )
    max_y = max( [max(y)  for x, y in raw_strokes ] )   
    max_x, max_y =  max_x + 1, max_y + 1
    img = np.zeros((max_y, max_x), np.uint8)   
    for i in range(step):
        stroke = raw_strokes[i]        
        x, y=stroke[0], stroke[1]
        for i in range(len(x) - 1):
            _ = cv2.line(img, (x[i], y[i]),(x[i + 1], y[i + 1]), 255, lw)   
    top = max ((size- max_y) // 2 , 0 )
    bottom = max ( size- top - max_y , 0)
    left = max ((size - max_x) // 2, 0 )
    right = max (size - max_x - left, 0 )
    img =  cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT)
    return img

#%% =============== Draw a mean image for the Chinese airplane doodles  =============================
df_cn = pd.read_csv('df_cn.csv')
airplane = df_cn[df_cn['word'] == 'airplane']
image = draw_cv2(airplane['drawing'][0], size=256, lw=3)
original_shape=image.shape
flat = image.ravel()
facevector = np.matrix(flat)
facematrix=facevector
for i in range(1,len(airplane)):
    image = draw_cv2(airplane['drawing'][i], size=256, lw=3)
    flat = image.ravel()
    facevector = np.matrix(flat)
    facematrix=np.r_[facematrix,facevector]
facematrix_t = np.transpose(facematrix) 

# Visualize the first Chinese airplane doodle
face_example = np.asarray(facematrix_t[:,0]).reshape(original_shape)
# make a PIL image and save it to jpg
face_example_img = Image.fromarray(face_example, 'L')
face_example_img.show()
face_example_img.save("cn-airplane.jpg")

# Draw the mean image of the Chinese airplane doodles
mean_facematrix = facematrix_t.mean(1)
mean_face = np.asarray(mean_facematrix).reshape(original_shape).astype(np.uint8)
mean_face_img = Image.fromarray(mean_face, 'L')
mean_face_img.show()
mean_face_img.save("cn-mean-airplane.jpg")

#%% =============== Draw a mean image for the Chinese airplane doodles  =============================
df_us = pd.read_csv('us_all_100.csv')
airplane = df_us[df_us['word'] == 'airplane'].reset_index(drop=True)
airplane['drawing'] = airplane['drawing'].map(ast.literal_eval)
image = draw_cv2(airplane['drawing'][0], size=256, lw=3)
original_shape=image.shape
flat = image.ravel()
facevector = np.matrix(flat)
facematrix=facevector
for i in range(1,len(airplane)):
    image = draw_cv2(airplane['drawing'][i], size=256, lw=3)
    flat = image.ravel()
    facevector = np.matrix(flat)
    facematrix=np.r_[facematrix,facevector]
facematrix_t = np.transpose(facematrix)
 
# Visualize the first American airplane doodle
face_example = np.asarray(facematrix_t[:,0]).reshape(original_shape)
# make a PIL image and save it to jpg
face_example_img = Image.fromarray(face_example, 'L')
face_example_img.show()
face_example_img.save("us-airplane.jpg")

# Draw the mean image of the American airplane doodles
mean_facematrix = facematrix_t.mean(1)
mean_face = np.asarray(mean_facematrix).reshape(original_shape).astype(np.uint8)
mean_face_img = Image.fromarray(mean_face, 'L')
mean_face_img.show()
mean_face_img.save("us-mean-airplane.jpg")

