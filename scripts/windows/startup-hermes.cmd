@echo off
setlocal enabledelayedexpansion
:: Hermes Agent — Windows startup script (template)
::
:: Customize the paths below to match your installation, then copy this
:: script to your Windows Startup folder for auto-launch at boot time.
::
::   copy startup-hermes.cmd "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\hermes-startup.cmd"

:: ── CONFIGURE THESE PATHS ──────────────────────────────────────
set HERMES_HOME=%USERPROFILE%\AppData\Local\hermes
set HERMES_INSTALL=%USERPROFILE%\hermes-agent
set HERMES_BIN=%HERMES_INSTALL%\venv\Scripts\hermes.exe
set WEBUI_CMD=%USERPROFILE%\AppData\Roaming\npm\hermes-web-ui.cmd
:: ────────────────────────────────────────────────────────────────

set LOGFILE=%HERMES_HOME%\logs\startup.log

if not exist "%HERMES_HOME%\logs" mkdir "%HERMES_HOME%\logs"
echo [%date% %time%] === Hermes startup begin === >> "%LOGFILE%"

:: Clean stale lock/pid files from previous unclean shutdown
del /f /q "%HERMES_HOME%\gateway.lock" 2>nul
del /f /q "%HERMES_HOME%\gateway.pid" 2>nul
echo [%date% %time%] Cleaned stale lock files >> "%LOGFILE%"

:: Wait for network to be ready before launching gateway
echo [%date% %time%] Waiting for network... >> "%LOGFILE%"
set NETWORK_OK=0
for /L %%i in (1,1,30) do (
    nslookup google.com >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set NETWORK_OK=1
        goto :network_ready
    )
    timeout /t 2 /nobreak >nul
)
:network_ready
if !NETWORK_OK! equ 1 (
    echo [%date% %time%] Network is ready >> "%LOGFILE%"
) else (
    echo [%date% %time%] WARNING: Network not detected after 60s, proceeding anyway >> "%LOGFILE%"
)

:: Launch hermes-agent gateway (API on 127.0.0.1:8642)
echo [%date% %time%] Starting hermes gateway... >> "%LOGFILE%"
set API_SERVER_ENABLED=true
cd /d "%HERMES_INSTALL%"
start "" /MIN "%HERMES_BIN%" gateway run --replace
echo [%date% %time%] hermes gateway launched, waiting for port 8642... >> "%LOGFILE%"

:: Wait for gateway API server to actually listen on port 8642
set GATEWAY_READY=0
for /L %%i in (1,1,60) do (
    netstat -ano 2>nul | findstr "127.0.0.1:8642" | findstr "LISTENING" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set GATEWAY_READY=1
        goto :gateway_ready
    )
    timeout /t 1 /nobreak >nul
)
:gateway_ready
if !GATEWAY_READY! equ 1 (
    echo [%date% %time%] Gateway is listening on port 8642 >> "%LOGFILE%"
) else (
    echo [%date% %time%] WARNING: Gateway did not start within 60s >> "%LOGFILE%"
)

:: Extra buffer for gateway platform connections (Feishu/Weixin) to stabilize
timeout /t 3 /nobreak >nul

:: Launch hermes-web-ui (Web panel on 127.0.0.1:8648)
echo [%date% %time%] Starting hermes web ui... >> "%LOGFILE%"
if exist "%WEBUI_CMD%" (
    start "" /MIN "%WEBUI_CMD%" start --port 8648
    echo [%date% %time%] hermes web ui launched >> "%LOGFILE%"
) else (
    echo [%date% %time%] WARNING: Web UI not found at %WEBUI_CMD% >> "%LOGFILE%"
)

:: Launch watchdog (background health monitor, checks port 8642 every 5min)
echo [%date% %time%] Starting hermes watchdog... >> "%LOGFILE%"
set WATCHDOG_PS1=%HERMES_INSTALL%\hermes-watchdog.ps1
if exist "%WATCHDOG_PS1%" (
    start "" /MIN powershell.exe -WindowStyle Hidden -ExecutionPolicy Bypass -File "%WATCHDOG_PS1%"
    echo [%date% %time%] hermes watchdog launched >> "%LOGFILE%"
) else (
    echo [%date% %time%] WARNING: Watchdog not found at %WATCHDOG_PS1% >> "%LOGFILE%"
)

echo [%date% %time%] === Hermes startup complete === >> "%LOGFILE%"
