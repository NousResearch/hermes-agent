@echo off
title COSMOS Setup
echo.
echo  ============================================
echo       COSMOS SETUP
echo  ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.10+
    pause
    exit /b
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo  Python Version: %PYTHON_VERSION%

echo.
echo  NOTE: If you have Python 3.12+, some optional voice features
echo        (TTS, Whisper) are disabled. Core features work fine!
echo.

REM Create Virtual Environment if not exists
if not exist "venv" (
    echo  Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Upgrade pip
echo.
echo  Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install core dependencies first
echo.
echo  Installing core dependencies...
pip install numpy scipy loguru pyyaml python-dotenv pydantic --quiet

REM Install Emotional API dependencies
echo  Installing Emotional API dependencies...
pip install opencv-python --quiet

REM Try to install pyaudio (may fail without build tools)
pip install pyaudio --quiet 2>nul
if %errorlevel% neq 0 (
    echo  NOTE: PyAudio needs Visual C++ Build Tools to compile.
    echo        Camera-only mode will still work.
)

REM Install remaining requirements
echo  Installing remaining dependencies...
echo  This may take a few minutes...
pip install -r requirements.txt 2>nul

echo.
echo  ============================================
echo       SETUP COMPLETE
echo  ============================================
echo.
echo  Next steps:
echo    1. Ensure Ollama is installed: https://ollama.ai
echo    2. Run: ollama pull llama3.2:3b
echo    3. Run: START.bat to launch COSMOS
echo.

REM Keep venv activated
cmd /k "venv\Scripts\activate"
