@echo off
set HERMES_DIR=%~dp0..
set WORKSPACE_DIR=%HERMES_DIR%\..\hermes-workspace

taskkill /F /IM hermes.exe >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" gateway run
timeout /t 8 /nobreak >nul

start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" dashboard
timeout /t 5 /nobreak >nul

start /B "" cmd.exe /c "cd /d %WORKSPACE_DIR% && set HERMES_API_URL=http://127.0.0.1:8642 && set HERMES_AGENT_PATH=%HERMES_DIR% && set PORT=3000 && node server-entry.js"
timeout /t 8 /nobreak >nul
