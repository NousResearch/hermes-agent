@echo off
setlocal
REM ============================================================================
REM start-hermes-stack.cmd — one-command bring-up for the Hermes local stack.
REM
REM Boots the gateway and the dashboard together: two detached python
REM processes with separate stdout/stderr logs, then waits for the dashboard
REM to answer HTTP 200 on 127.0.0.1:%PORT%.
REM
REM  * Works from any checkout — the repo root is derived from this file's
REM    own location (%~dp0), so no path editing is needed after cloning.
REM  * Idempotent: if the dashboard already answers, nothing is started.
REM  * One optional argument: dashboard port (default 9119).
REM  * All output lands in %LOCALAPPDATA%\hermes\logs\.
REM  * For auto-start on logon (Windows), put a shortcut to this script in
REM    shell:startup — the script's idempotence makes that safe.
REM ============================================================================

set "PORT=%~1"
if "%PORT%"=="" set "PORT=9119"

rem Repo root = the directory this script lives in (strip trailing backslash).
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"

set "LOGDIR=%LOCALAPPDATA%\hermes\logs"
set "GATEWAY_LOG=%LOGDIR%\gateway-stdio.log"
set "DASH_LOG=%LOGDIR%\dashboard-stdio.log"

if not exist "%REPO%\" (
  echo [start-hermes-stack] repo not found: %REPO%
  exit /b 1
)
if not exist "%LOGDIR%\" mkdir "%LOGDIR%" >nul 2>&1

echo [start-hermes-stack] waiting for dashboard on 127.0.0.1:%PORT% ...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$port=%PORT%;" ^
  "function Test-Dash { try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri \"http://127.0.0.1:$port/\"; return $r.StatusCode -eq 200 } catch { return $false } }" ^
  "if (Test-Dash) { Write-Host '[start-hermes-stack] dashboard already up on '$port' - nothing to do'; exit 0 }" ^
  "Write-Host '[start-hermes-stack] starting gateway...';" ^
  "Start-Process -FilePath 'python.exe' -ArgumentList '-m','hermes_cli.main','gateway','run' -WorkingDirectory '%REPO%' -RedirectStandardOutput '%GATEWAY_LOG%' -RedirectStandardError '%GATEWAY_LOG%.err' -WindowStyle Hidden | Out-Null;" ^
  "Write-Host '[start-hermes-stack] starting dashboard (cold build can take 40-90s)...';" ^
  "Start-Process -FilePath 'python.exe' -ArgumentList '-m','hermes_cli.main','dashboard','--port',$port,'--no-open' -WorkingDirectory '%REPO%' -RedirectStandardOutput '%DASH_LOG%' -RedirectStandardError '%DASH_LOG%.err' -WindowStyle Hidden | Out-Null;" ^
  "$deadline=(Get-Date).AddSeconds(150);" ^
  "while((Get-Date) -lt $deadline){ if(Test-Dash){ Write-Host '[start-hermes-stack] READY: http://127.0.0.1:'$port'/'; exit 0 }; Start-Sleep -Seconds 3 }" ^
  "Write-Host '[start-hermes-stack] TIMEOUT: dashboard did not answer in 150s';" ^
  "Write-Host '  gateway log: %GATEWAY_LOG%';" ^
  "Write-Host '  dashboard log: %DASH_LOG%';" ^
  "exit 1"

endlocal & exit /b %ERRORLEVEL%
