@echo off
REM Hermes Agent Services Watchdog - Production version
REM Checks gateway, dashboard, and workspace every 2 minutes
REM Uses production server (not dev server) for stability

set HERMES_DIR=%~dp0..
set WORKSPACE_DIR=%HERMES_DIR%\..\hermes-workspace
set LOG=%TEMP%\hermes-watchdog.log

echo. >> "%LOG%"
echo [%date% %time%] === Watchdog check === >> "%LOG%"

REM Check Gateway (port 8642) - real HTTP check
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8642/v1/health' -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo [%date% %time%] GATEWAY DOWN - restarting >> "%LOG%"
    taskkill /F /IM hermes.exe >nul 2>&1
    timeout /t 3 /nobreak >nul
    start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" gateway run
    timeout /t 10 /nobreak >nul
    echo [%date% %time%] Gateway restarted >> "%LOG%"
) else (
    echo [%date% %time%] Gateway OK >> "%LOG%"
)

REM Check Dashboard (port 9119) - real HTTP check
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:9119/' -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo [%date% %time%] DASHBOARD DOWN - restarting >> "%LOG%"
    taskkill /F /FI "IMAGENAME eq hermes.exe" /FI "CMDLINE eq *dashboard*" >nul 2>&1
    timeout /t 2 /nobreak >nul
    start /B "" "%HERMES_DIR%\.venv\Scripts\hermes.exe" dashboard
    timeout /t 12 /nobreak >nul
    echo [%date% %time%] Dashboard restarted >> "%LOG%"
) else (
    echo [%date% %time%] Dashboard OK >> "%LOG%"
)

REM Check Workspace (port 3000) - real HTTP check
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3000/' -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
    echo [%date% %time%] WORKSPACE DOWN - restarting >> "%LOG%"
    REM Only kill the node process running Hermes Workspace (port 3000), NOT all node.exe
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 3 /nobreak >nul
    start /B "" "%WORKSPACE_DIR%\start-workspace.bat"
    timeout /t 10 /nobreak >nul
    echo [%date% %time%] Workspace restarted >> "%LOG%"
) else (
    echo [%date% %time%] Workspace OK >> "%LOG%"
)

echo [%date% %time%] === Check complete === >> "%LOG%"
