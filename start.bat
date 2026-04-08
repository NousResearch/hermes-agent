@echo off
SETLOCAL EnableDelayedExpansion

echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.10 or newer from https://www.python.org/
    pause
    exit /b 1
)

echo Checking dependencies...
python -c "import fastapi, uvicorn, jinja2" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing missing dependencies...
    python -m pip install fastapi uvicorn jinja2 pydantic
)

echo Starting Hermes Dashboard...
set PYTHONPATH=%PYTHONPATH%;.
start "" http://localhost:8000
python dashboard/app.py
pause
