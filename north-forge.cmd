@echo off
REM north-forge.cmd - double-click launcher. Bootstraps on first run, then starts `hermes`.
REM Minimal, single-drive: venv + data are siblings of this checkout (see scripts\bootstrap-north-forge.ps1).
REM Not the hardened install - that waits on DECISION-2026-09-06-003. Pass args straight through: north-forge.cmd gateway
REM Every launch also self-heals, against the CURRENT machine + drive letter/path:
REM   - the run environment itself: a real readiness probe (does the venv's
REM     python run? does it import hermes_cli from THIS checkout? does the
REM     .nf-bootstrapped marker match?) + a silent venv rebuild if not - a venv
REM     is not portable between machines  (ERR-2026-09-07-006 / CHG-2026-09-07-020)
REM   - the North Forge CLI skin copy in HERMES_HOME\skins\  (CHG-2026-09-07-012)
REM   - "<drive>:\Start North Forge.lnk", a drive-root double-click launcher  (CHG-2026-09-07-013)
setlocal
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
for %%I in ("%REPO%")     do set "LEAF=%%~nxI"
for %%I in ("%REPO%\..")  do set "PARENT=%%~fI"
set "VENV=%PARENT%\%LEAF%-venv"
set "DATA=%PARENT%\%LEAF%-data"
set "PREFLIGHT=%REPO%\scripts\nf-preflight.ps1"
set "LAUNCHLOG=%PARENT%\%LEAF%-launcher.log"

if exist "%PREFLIGHT%" (
  REM Readiness probe + silent self-heal. Writes ONE line to "%LAUNCHLOG%" per
  REM launch BEFORE Python is ever started, so a broken interpreter cannot stop
  REM the log line from existing. Rebuilds the venv - never the data folder - if
  REM any check fails. Non-zero exit = not ready and the auto-repair did not fix it.
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PREFLIGHT%" -RepoRoot "%REPO%" -VenvDir "%VENV%" -DataDir "%DATA%" -LogFile "%LAUNCHLOG%"
  if errorlevel 1 (
    echo.
    echo [north-forge] the run environment is not ready and automatic repair failed.
    echo [north-forge] see "%LAUNCHLOG%" and the output above.
    pause
    exit /b 1
  )
) else (
  REM Fallback only if nf-preflight.ps1 is missing from the checkout: still leave
  REM a launcher-log line, then degrade to the legacy first-run existence check.
  >>"%LAUNCHLOG%" echo %DATE% %TIME% ^| host=%COMPUTERNAME% ^| repo=%REPO% ^| checks: nf-preflight.ps1=MISSING ^| action=legacy-existence-check
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

REM --- North Forge tier check (CHG-2026-09-07-022) --------------------------------
REM If this drive carries a provisioning record and it has been tampered with,
REM stop here with a clear message rather than letting the in-process gate refuse
REM every subcommand. A missing record is normal (un-provisioned drive) -> exit 0.
"%VENV%\Scripts\python.exe" -m hermes_cli.nf_tier verify >nul 2>&1
if errorlevel 2 (
  echo.
  echo [north-forge] this drive's North Forge provisioning is invalid or was modified
  echo [north-forge] after Setup Run. An admin must repair it:  scripts\nf-setup.ps1 -Force
  "%VENV%\Scripts\python.exe" -m hermes_cli.nf_tier show
  pause
  exit /b 2
)

"%VENV%\Scripts\hermes.exe" %*
