@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
:: Hermes Agent — Windows emergency recovery script
::
:: Double-click to recover a zombie/unresponsive Hermes gateway.
:: Kills stale processes, cleans lock files, restarts gateway + web UI.

title Hermes One-Click Recovery
color 0E

:: ── CONFIGURE THESE PATHS ──────────────────────────────────────
set HERMES_HOME=%USERPROFILE%\AppData\Local\hermes
set HERMES_INSTALL=%USERPROFILE%\hermes-agent
set HERMES_BIN=%HERMES_INSTALL%\venv\Scripts\hermes.exe
set WEBUI_CMD=%USERPROFILE%\AppData\Roaming\npm\hermes-web-ui.cmd
:: ────────────────────────────────────────────────────────────────

set LOGFILE=%HERMES_HOME%\logs\recovery.log

if not exist "%HERMES_HOME%\logs" mkdir "%HERMES_HOME%\logs"
echo [%date% %time%] === Recovery started === >> "%LOGFILE%"

echo.
echo ╔══════════════════════════════════════════════╗
echo ║       Hermes Gateway Recovery Tool           ║
echo ╚══════════════════════════════════════════════╝
echo.

:: ── Step 1: Kill all zombie Hermes processes ──────────────────
echo [1/4] Cleaning zombie processes...
echo [%date% %time%] Killing stale hermes processes... >> "%LOGFILE%"

taskkill /f /im hermes.exe 2>nul
if %ERRORLEVEL% equ 0 (
    echo    [OK] hermes.exe terminated
    echo [%date% %time%] hermes.exe killed >> "%LOGFILE%"
) else (
    echo    --  hermes.exe not running (OK)
)

echo [%date% %time%] Web UI processes cleaned >> "%LOGFILE%"

:: ── Step 2: Clean stale lock files ────────────────────────────
echo [2/4] Cleaning lock files...
del /f /q "%HERMES_HOME%\gateway.lock" 2>nul
del /f /q "%HERMES_HOME%\gateway.pid" 2>nul

if exist "%HERMES_HOME%\*.lock" (
    del /f /q "%HERMES_HOME%\*.lock" 2>nul
    echo    [OK] Lock files cleaned
) else (
    echo    --  No stale lock files
)
echo [%date% %time%] Lock files cleaned >> "%LOGFILE%"

:: ── Step 3: Launch Hermes Gateway ─────────────────────────────
echo [3/4] Starting Hermes Gateway (port 8642)...

set API_SERVER_ENABLED=true
cd /d "%HERMES_INSTALL%"
start "" /MIN "%HERMES_BIN%" gateway run --replace

set GATEWAY_READY=0
for /L %%i in (1,1,60) do (
    netstat -ano 2>nul | findstr "127.0.0.1:8642" | findstr "LISTENING" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set GATEWAY_READY=1
        goto :gateway_up
    )
    <nul set /p =.
    timeout /t 1 /nobreak >nul
)
:gateway_up
echo.
if %GATEWAY_READY% equ 1 (
    echo    [OK] Gateway listening on 127.0.0.1:8642
    echo [%date% %time%] Gateway listening on 8642 >> "%LOGFILE%"
) else (
    echo    [WARN] Gateway not ready after 60s - check logs
    echo [%date% %time%] WARNING: Gateway not ready after 60s >> "%LOGFILE%"
)

:: ── Step 4: Launch Hermes Web UI ──────────────────────────────
echo [4/4] Starting Hermes Web UI (port 8648)...

netstat -ano 2>nul | findstr "0.0.0.0:8648" | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo    --  Web UI already running
) else (
    if exist "%WEBUI_CMD%" (
        start "" /MIN "%WEBUI_CMD%" start --port 8648
        timeout /t 3 /nobreak >nul
        netstat -ano 2>nul | findstr "0.0.0.0:8648" | findstr "LISTENING" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo    [OK] Web UI listening on 0.0.0.0:8648
        ) else (
            echo    --  Web UI launching, check back shortly
        )
    ) else (
        echo    --  Web UI not found, skipping
    )
)
echo [%date% %time%] Web UI launched >> "%LOGFILE%"

:: ── Done ──────────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════╗
echo ║  [OK] Hermes Recovery Complete!              ║
echo ║                                            ║
echo ║  Gateway API : http://127.0.0.1:8642       ║
echo ║  Web UI      : http://127.0.0.1:8648       ║
echo ╚══════════════════════════════════════════════╝
echo.
echo [%date% %time%] === Recovery complete === >> "%LOGFILE%"

timeout /t 2 /nobreak >nul
start http://127.0.0.1:8648

pause
