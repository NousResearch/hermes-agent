@echo off
title COSMOS - Local Host IP Discovery
color 0B

echo.
echo ========================================================
echo       COSMOS LOCAL HOST SETUP - GET YOUR PHONE URL
echo ========================================================
echo.
echo Step 1: Ensure you have started COSMOS via START.bat (Option 13)
echo Step 2: Make sure your phone is connected to the SAME Wi-Fi as this PC.
echo.
echo Your Local IP Address is:
echo.

:: Get IPv4 address
for /f "tokens=14" %%a in ('ipconfig ^| findstr IPv4') do set IP=%%a

echo    ----------------------------------------
echo    TYPE THIS INTO YOUR PHONE'S BROWSER:
echo    http://%IP%:8081
echo    ----------------------------------------
echo.
echo Note: 
echo - If it doesn't load, check your Windows Firewall. 
echo - You may need to allow Python/Node through the firewall on port 8081.
echo.
pause
