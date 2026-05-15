@echo off
REM Hermes Agent - Start All Services (Production)
REM Launches gateway, dashboard, workspace (production server), and watchdog

set HERMES_DIR=%~dp0..
set WORKSPACE_DIR=%HERMES_DIR%\..\hermes-workspace

echo Starting Hermes Agent services...

REM Kill any existing Hermes instances
taskkill /F /IM hermes.exe >nul 2>&1
REM Only kill node processes running Hermes Workspace on port 3000
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Start Gateway
echo [1/4] Starting Gateway...
start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" gateway run
timeout /t 8 /nobreak >nul

REM Start Dashboard
echo [2/4] Starting Dashboard...
start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" dashboard
timeout /t 5 /nobreak >nul

REM Start Workspace (production server - stable, no crashes)
echo [3/4] Starting Workspace (production)...
start /B "" cmd.exe /c "cd /d %WORKSPACE_DIR% && set HERMES_API_URL=http://127.0.0.1:8642 && set HERMES_AGENT_PATH=%HERMES_DIR% && set PORT=3000 && node server-entry.js"
timeout /t 8 /nobreak >nul

REM Start Watchdog
echo [4/4] Starting Watchdog...
schtasks /Run /TN "HermesAgentWatchdog" >nul 2>&1

echo.
echo All services started.
echo   Gateway:   http://127.0.0.1:8642
echo   Dashboard: http://127.0.0.1:9119
echo   Workspace: http://localhost:3000
echo   Watchdog:  runs every 2 minutes, auto-restarts any downed service
