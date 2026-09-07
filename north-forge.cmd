@echo off
REM north-forge.cmd - double-click launcher. Bootstraps on first run, then starts `hermes`.
REM Minimal, single-drive: venv + data are siblings of this checkout (see scripts\bootstrap-north-forge.ps1).
REM Not the hardened install - that waits on DECISION-2026-09-06-003. Pass args straight through: north-forge.cmd gateway
setlocal
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
for %%I in ("%REPO%")     do set "LEAF=%%~nxI"
for %%I in ("%REPO%\..")  do set "PARENT=%%~fI"
set "VENV=%PARENT%\%LEAF%-venv"
set "DATA=%PARENT%\%LEAF%-data"

if not exist "%VENV%\Scripts\hermes.exe" (
  echo [north-forge] first run - bootstrapping ^(one time^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\bootstrap-north-forge.ps1"
  if errorlevel 1 (
    echo.
    echo [north-forge] bootstrap failed - see the output above.
    pause
    exit /b 1
  )
)

if not exist "%DATA%" mkdir "%DATA%"
set "HERMES_HOME=%DATA%"
"%VENV%\Scripts\hermes.exe" %*
