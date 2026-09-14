@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Hermes Workstation - Dogfood Launcher

echo ============================================================
echo   HERMES WORKSTATION - ONE-CLICK DOGFOOD
echo ============================================================
echo.
echo [1/3] Installing and validating Hermes Workstation...
call "%~dp0workstation\install.cmd"
if errorlevel 1 goto :install_failed

echo.
echo [2/3] Running Hermes Workstation Doctor...
call "%~dp0workstation\doctor.cmd" -Strict
if errorlevel 1 goto :doctor_failed

echo.
echo [3/3] Starting Hermes Workstation...
findstr /c:"[switch]$SkipInstall" "%~dp0workstation\start.ps1" >nul 2>&1
if errorlevel 1 (
  rem Backwards-compatible fallback for a checkout older than the one-click launcher support.
  call "%~dp0workstation\start.cmd"
) else (
  call "%~dp0workstation\start.cmd" -SkipInstall
)
set "HERMES_EXIT=%ERRORLEVEL%"
if not "%HERMES_EXIT%"=="0" goto :start_failed
exit /b 0

:install_failed
set "HERMES_EXIT=%ERRORLEVEL%"
echo.
echo [FAIL] Installation/validation failed with exit code %HERMES_EXIT%.
echo Fix the error shown above and double-click this file again.
pause
exit /b %HERMES_EXIT%

:doctor_failed
set "HERMES_EXIT=%ERRORLEVEL%"
echo.
echo [FAIL] Doctor failed with exit code %HERMES_EXIT%.
echo Fix the error shown above and double-click this file again.
pause
exit /b %HERMES_EXIT%

:start_failed
echo.
echo [FAIL] Hermes Workstation exited with code %HERMES_EXIT%.
echo Review the output above, then double-click this file to retry.
pause
exit /b %HERMES_EXIT%
