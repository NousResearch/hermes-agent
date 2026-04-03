@echo off
setlocal

REM Build foureleven.exe on Windows using PyInstaller.
REM Run from repo root after installing deps in a Windows venv.

if "%VIRTUAL_ENV%"=="" (
  echo Activate a virtualenv first.
  exit /b 1
)

python -m pip install pyinstaller >nul
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name foureleven ^
  bootstrap_recovery\foureleven_bootstrap.py

if errorlevel 1 exit /b 1

echo Built dist\foureleven.exe
