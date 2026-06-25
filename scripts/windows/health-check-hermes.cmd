@echo off
setlocal enabledelayedexpansion

set HERMES_HOME=C:\Users\aliyf\AppData\Local\hermes
set LOGFILE=%HERMES_HOME%\logs\health-check.log

:: Check if gateway is listening on port 8642
netstat -ano 2>nul | findstr "127.0.0.1:8642" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    :: Gateway is alive — silent exit
    exit /b 0
)

:: Gateway is DOWN — log and recover
echo [%date% %time%] HEALTH CHECK FAILED: Port 8642 not listening! Starting recovery... >> "%LOGFILE%"

:: Step 1: Kill zombie processes
taskkill /f /im hermes.exe 2>nul
echo [%date% %time%] Killed zombie hermes.exe (exit code %ERRORLEVEL%) >> "%LOGFILE%"

:: Step 2: Clean lock files
del /f /q "%HERMES_HOME%\gateway.lock" 2>nul
del /f /q "%HERMES_HOME%\gateway.pid" 2>nul

:: Step 3: Restart Gateway
set API_SERVER_ENABLED=true
cd /d "C:\Users\aliyf\hermes-agent"
start "" /MIN "C:\Users\aliyf\hermes-agent\venv\Scripts\hermes.exe" gateway run --replace

:: Step 4: Wait for port 8642
set GATEWAY_READY=0
for /L %%i in (1,1,30) do (
    timeout /t 2 /nobreak >nul
    netstat -ano 2>nul | findstr "127.0.0.1:8642" | findstr "LISTENING" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set GATEWAY_READY=1
        goto :gateway_up
    )
)
:gateway_up

:: Step 5: Restart Web UI if needed
netstat -ano 2>nul | findstr "0.0.0.0:8648" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    start "" /MIN "C:\nodejs\node_global\hermes-web-ui.cmd" start --port 8648
    echo [%date% %time%] Web UI restarted >> "%LOGFILE%"
)

if %GATEWAY_READY% equ 1 (
    echo [%date% %time%] Auto-recovery SUCCESS: Gateway listening on 8642 >> "%LOGFILE%"
) else (
    echo [%date% %time%] Auto-recovery FAILED: Gateway did not start within 60s >> "%LOGFILE%"
)

exit /b 0
