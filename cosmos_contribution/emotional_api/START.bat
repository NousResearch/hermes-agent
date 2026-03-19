@echo off
title 12D CST Emotional State API - MediaPipe Enhanced
cd /d %~dp0

:menu
cls
echo.
echo  ================================================================
echo       12D CST EMOTIONAL STATE API v4.0
echo       MediaPipe 468-Landmark Face Mesh + Action Units
echo  ================================================================
echo.
echo  *** RECOMMENDED ***
echo    1. FULL SYSTEM (Camera + API + cosmos) [BEST]
echo.
echo  VISUAL MODES:
echo    2. HD Visual Mode (Face Mesh + Audio + Data)
echo    3. Visual Display Only (No Audio)
echo.
echo  LIVE MODES:
echo    4. Live Demo (Camera + Mic - Terminal)
echo    5. Demo Mode (No Devices - Simulation)
echo.
echo  API SERVER:
echo    6. Emotional Token Server Only (Port 8765)
echo    7. Analyze Files (Audio + Image)
echo.
echo  UTILITIES:
echo    8. Check Devices (Camera/Mic Test)
echo    9. Test MediaPipe Installation
echo    0. Exit
echo.
echo  ================================================================
set /p choice="Select option: "

if "%choice%"=="1" goto full_system
if "%choice%"=="2" goto visual_hd
if "%choice%"=="3" goto visual
if "%choice%"=="4" goto live
if "%choice%"=="5" goto demo
if "%choice%"=="6" goto server
if "%choice%"=="7" goto files
if "%choice%"=="8" goto check
if "%choice%"=="9" goto test_mp
if "%choice%"=="0" exit

echo Invalid option. Press any key to try again.
pause >nul
goto menu

:full_system
cls
echo.
echo  ================================================================
echo    cosmos FULL SYSTEM - COMPLETE INTEGRATION
echo  ================================================================
echo.
echo  Checking for existing processes on port 8765...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8765 ^| findstr LISTENING') do (
    echo  Killing process %%a...
    taskkill /F /PID %%a >nul 2>&1
)
echo.
echo  Starting all components:
echo.
echo    [x] MediaPipe 468-Landmark Face Tracking
echo    [x] Real-time Action Unit Detection
echo    [x] Emotional Token Server (Port 8765)
echo    [x] WebSocket API for cosmos AI
echo.
echo  Connect cosmos to: ws://localhost:8765/ws
echo.
echo  Press Q in the video window to quit
echo.
python full_system.py
pause
goto menu

:visual_hd
cls
echo.
echo  ============================================
echo    HD VISUAL MODE - MediaPipe Face Mesh
echo  ============================================
echo.
echo  Features:
echo    - 468 3D facial landmarks
echo    - Real-time Action Unit detection
echo    - Upper/Lower Tensor visualization
echo    - Audio spectral analysis
echo.
echo  Press 'Q' to quit the visual window
echo.
python live_demo.py
pause
goto menu

:visual
cls
echo.
echo  Starting Visual Display Only...
echo  (Press 'q' in the window to quit)
echo.
python visual_display.py
pause
goto menu

:live
cls
echo.
echo  Starting LIVE Emotional Detection...
echo.
python live_demo.py
pause
goto menu

:demo
cls
echo.
echo  Running CST Emotional State Demo (Simulation)...
echo.
python emotional_state_api.py
pause
goto menu

:server
cls
echo.
echo  ============================================
echo    EMOTIONAL TOKEN SERVER
echo  ============================================
echo.
echo  Cleaning up any existing processes on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
echo.
echo  Starting on http://localhost:8765
echo.
echo  Endpoints:
echo    GET  /state         - cosmos_packet JSON
echo    GET  /stream        - SSE token stream
echo    WS   /ws            - WebSocket tokens
echo    GET  /system_prompt - LLM steering prompt
echo.
python emotion_server.py
pause
goto menu

:files
cls
echo.
echo  Analyze Audio/Image Files
echo.
set /p audio="Enter audio file path (or 'none'): "
set /p image="Enter image file path (or 'none'): "
python emotional_state_api.py %audio% %image%
pause
goto menu

:check
cls
echo.
echo  Checking capture devices...
echo.
python live_capture.py
pause
goto menu

:test_mp
cls
echo.
echo  Testing MediaPipe Installation...
echo.
python -c "from emotional_state_api import MEDIAPIPE_AVAILABLE, get_mediapipe_tracker; print('MediaPipe Available:', MEDIAPIPE_AVAILABLE); t = get_mediapipe_tracker(); print('Tasks API:', t.use_tasks_api); print('Face Landmarker Ready:', t.face_landmarker is not None); print(); print('SUCCESS! Ready for 468-landmark tracking!' if t.face_landmarker else 'FALLBACK: Using Haar Cascade (still works)')"
pause
goto menu
