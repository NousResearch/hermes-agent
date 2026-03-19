@echo off
title COSMOS - Full Experience
cd /d %~dp0

:: ============================================
:: API KEYS (loaded from .env)
:: ============================================
:: Parse .env file: skip blank lines and comments
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    if not "%%A"=="" set "%%A=%%B"
)

:menu
cls
echo.
echo  ================================================================
echo       COSMOS - FULL EXPERIENCE LAUNCHER v2.9
echo       12D CST Physics + Emotional Intelligence + AI Companion
echo  ================================================================
echo.
echo  *** RECOMMENDED ***
echo   13. FULL SYSTEM (Camera + API + AI) [BEST]
echo.
echo  CORE MODES:
echo    1. Interactive CLI (Full Features)
echo    2. Web Chat Interface + Emotional API (Port 8081 + 8765)
echo    3. Streamlit Dashboard
echo.
echo  12D CST EMOTIONAL API:
echo    4. Emotional Visual Mode (HD Face + Audio + Data)
echo    5. Visual Display Only
echo    6. CST Demo (Physics Test)
echo   12. Emotional Token Server (Standalone API on 8765)
echo.
echo  NETWORK + ADVANCED:
echo    7. P2P Network Node
echo    8. P2P Node + Dashboard
echo    9. Health Dashboard
echo.
echo  *** NEW: DIGITAL ORGANISM ***
echo   14. CNS MODE (Bio-Digital Central Nervous System)
echo.
echo  UTILITIES:
echo   15. Host Web Server via Ngrok (Internet Access)
echo   10. Run All Tests
echo   11. First-Time Setup
echo    0. Exit
echo.
echo  ================================================================
set /p choice="Select option: "

if "%choice%"=="1" goto cli
if "%choice%"=="2" goto web
if "%choice%"=="3" goto ui
if "%choice%"=="4" goto emotional_visual
if "%choice%"=="5" goto visual
if "%choice%"=="6" goto cst_demo
if "%choice%"=="7" goto p2p
if "%choice%"=="8" goto p2p_dashboard
if "%choice%"=="9" goto health
if "%choice%"=="10" goto tests
if "%choice%"=="11" goto setup
if "%choice%"=="12" goto emotional_server
if "%choice%"=="13" goto full_system
if "%choice%"=="14" goto cns_mode
if "%choice%"=="15" goto ngrok_host
if "%choice%"=="0" exit

echo Invalid option. Press any key to try again.
pause >nul
goto menu

:ngrok_host
cls
echo.
echo  ============================================
echo    COSMOS INTERNET TUNNEL (NGROK)
echo  ============================================
echo.
echo  Installing dependencies and initializing secure tunnel...
python -m pip install -q pyngrok python-dotenv
python scripts\start_ngrok.py
pause
goto menu

:cns_mode
cls
echo.
echo  ============================================
echo    COSMOS CNS - DIGITAL ORGANISM MODE
echo  ============================================
echo.
echo  Starting Ollama...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    start "Ollama" /min cmd /c "ollama serve"
    timeout /t 2 /nobreak >nul
) else (
    echo  WARNING: Ollama not found. Daemons will use stubs.
)
echo.
echo  Igniting Central Nervous System...
cd /d "%~dp0..\Cosmic Genesis A.Lmi Cybernetic Bio Resonance Core\cosmosynapse"
python start_cns.py
pause
cd /d %~dp0
goto menu

:cli
cls
echo.
echo  Starting COSMOS CLI Mode...
echo  Type 'help' for commands, 'exit' to quit.
echo.
python main.py --cli
pause
goto menu

:web
cls
echo.
echo  ============================================
echo    COSMOS WEB + FULL SENSORY SYSTEM
echo  ============================================
echo.
echo  Cleaning up any lingering processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081 ^| findstr LISTENING') do taskkill /F /T /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do taskkill /F /T /PID %%a >nul 2>&1
echo.
echo  Starting Full Sensory System (Camera + Mic + Audio)...
start "Emotional API + Sensors" cmd /c "cd emotional_api && python full_system.py"
timeout /t 3 /nobreak >nul

:: Start Ollama if installed
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo  Starting Ollama Server...
    start "Ollama" /min cmd /c "ollama serve"
) else (
    echo  ⚠️  Ollama not found in PATH. Local models may fail.
)

echo.
    echo  Starting Matrix Console...
    start "COSMOS Matrix Protocol" cmd /c "python scripts/matrix_console.py"

echo.
echo  Starting Web Interface (GEMINI VERIFYING)...
echo.
echo  ========================================
echo    SERVERS RUNNING:
echo  ========================================
echo    Web Interface (PC):    http://localhost:8081
echo    Emotional API:         http://localhost:8765
echo    Matrix Console:        [Running in separate window]
echo.
echo  ========================================
echo    📱 PHONE UI CONNECTION
echo  ========================================
for /f "tokens=4" %%a in ('route print ^| find " 0.0.0.0 "') do set LOCAL_IP=%%a
echo    Open Safari/Chrome on your phone and go to:
echo    http://%LOCAL_IP%:8081
echo.
echo    Sensory Features:
echo      [x] Camera - Face tracking (MediaPipe 468-landmarks)
echo      [x] Microphone - Audio analysis
echo      [x] Real-time emotion detection
echo      [x] CST Phase mapping
echo      [x] Cosmo's 54D CST Transformer (local swarm brain)
echo.
echo    Press Q in the camera window to stop sensors
echo  ========================================
echo.
set COSMOS_WEB_PORT=8081
python run_web.py
pause
goto menu

:ui
cls
echo.
echo  Starting Streamlit Dashboard...
echo.
python main.py --ui
pause
goto menu

:emotional_visual
cls
echo.
echo  ============================================
echo    CST EMOTIONAL API - VISUAL + DATA MODE
echo  ============================================
echo.
echo  This will open TWO windows:
echo    1. Live camera feed with emotion detection
echo    2. Live data tokenization for COSMOS
echo.
echo  Controls: [R] Record  [S] Send  [C] Capture  [Q] Quit
echo.
python emotional_api/live_demo.py
pause
goto menu

:visual
cls
echo.
echo  Starting Visual Display Mode...
echo  Press 'Q' in the window to quit.
echo.
python emotional_api/visual_display.py
pause
goto menu

:cst_demo
cls
echo.
echo  Running CST Physics Demo...
echo.
python demo_cst.py
pause
goto menu

:emotional_server
cls
echo.
echo  ============================================
echo    EMOTIONAL TOKEN SERVER - 12D CST ENGINE
echo  ============================================
echo.
echo  Cleaning up any existing processes on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
echo.
echo  Starting API server on http://localhost:8765
echo.
echo  Endpoints for COSMOS AI:
echo    GET  /state  - Current emotional state
echo    GET  /stream - SSE token stream (for continuous feed)
echo    WS   /ws     - WebSocket for real-time tokens
echo.
echo  Press Ctrl+C to stop the server.
echo.
python emotional_api/emotion_server.py --port 8765
pause
goto menu

:p2p
cls
echo.
echo  Starting P2P Network Node...
echo.
python main.py --node
pause
goto menu

:p2p_dashboard
cls
echo.
echo  Starting P2P Node with Dashboard...
echo.
python main.py --node --dashboard
pause
goto menu

:health
cls
echo.
echo  Starting Health Dashboard on port 8081...
echo.
python main.py --health
pause
goto menu

:tests
cls
echo.
echo  Running Test Suites...
echo.
echo  === CST Tests ===
python -m pytest tests/test_cst.py -v
echo.
echo  === Emotional API Test ===
python emotional_api/emotional_state_api.py
echo.
pause
goto menu

:setup
cls
echo.
echo  Running First-Time Setup Wizard...
echo.
python main.py --setup
pause
goto menu

:full_system
cls
echo.
echo  ================================================================
echo    COSMOS FULL SYSTEM - COMPLETE INTEGRATION
echo  ================================================================
echo.
echo  Starting all components together:
echo.
echo    [x] MediaPipe 468-Landmark Face Tracking
echo    [x] Real-time Action Unit Detection  
echo    [x] Emotional Token Server (Port 8765)
echo    [x] WebSocket API for COSMOS AI
echo.
echo  Connect COSMOS AI to: ws://localhost:8765/ws
echo.
echo  Cleaning up any existing processes on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
echo.
echo  Press Q in the video window to quit
echo.
cd emotional_api
python full_system.py
cd ..
pause
goto menu
