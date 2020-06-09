# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:05:39 2020

@author: Yashraj
"""
import numpy as np
import cv2
import os

import face_recognition as FR
print(FR)

test_img=cv2.imread('G:\Data Science Project\LBPH Face Recongition\Test_image.jpg')

faces_detected,gray_img=FR.faceDetection(test_img)
cv2.imshow
print("Faces Detected",faces_detected)

#Initializing the training
faces,face_ID=FR.labels_for_training_data(r'G:\Data Science Project\LBPH Face Recongition\capture\0')
face_recognizer=FR.train_classifier(faces,face_ID)
face_recognizer.save(r'G:\Data Science Project\LBPH Face Recongition\trainingData.yml')

name={0:'Yashraj/nData Scientist'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print(label)
    print(confidence)
    FR.draw_rect(test_img,face)
    predict_name=name[label]
    FR.put_text1(test_img,predict_name,x,y)
    
resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows


    
    
    
    
    
    
    










