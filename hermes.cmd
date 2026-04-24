@echo off
rem Hermes Agent native Windows launcher
rem Prioritizes the virtual environment Python if available

setlocal
set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%~dp0hermes_cli\main.py" %*
) else (
    python "%~dp0hermes_cli\main.py" %*
)
endlocal
