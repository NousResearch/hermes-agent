@echo off
REM north-forge.cmd - double-click launcher. Bootstraps on first run, then starts `hermes`.
REM Minimal, single-drive: venv + data are siblings of this checkout (see scripts\bootstrap-north-forge.ps1).
REM Not the hardened install - that waits on DECISION-2026-09-06-003. Pass args straight through: north-forge.cmd gateway
REM Every launch also self-heals two things against the CURRENT drive letter/path:
REM   - the North Forge CLI skin copy in HERMES_HOME\skins\  (CHG-2026-09-07-012)
REM   - "<drive>:\Start North Forge.lnk", a drive-root double-click launcher  (CHG-2026-09-07-013)
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

REM --- self-healing, refreshed every launch (cheap, idempotent, never fatal) ---
REM Keep HERMES_HOME's North Forge skin in sync with the checkout so a `git pull`
REM that updates the splash art takes effect without a re-bootstrap.
if not exist "%DATA%\skins" mkdir "%DATA%\skins"
if exist "%REPO%\skins\north-forge.yaml" copy /Y "%REPO%\skins\north-forge.yaml" "%DATA%\skins\north-forge.yaml" >nul 2>&1
REM (Re)write "<drive>:\Start North Forge.lnk" against the path the checkout is at
REM right now, so it stays correct even if Windows re-letters the drive.
if exist "%REPO%\scripts\make-drive-root-shortcut.ps1" powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\make-drive-root-shortcut.ps1" -RepoRoot "%REPO%" -Quiet >nul 2>&1

"%VENV%\Scripts\hermes.exe" %*
