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
REM Drive-letter changes are safe (venv + data are siblings that move together). A
REM CHECKOUT-FOLDER RENAME is not: the data-folder name is derived from the leaf
REM folder name, so a rename points HERMES_HOME at a fresh empty folder and the old
REM conversations/config/credentials look lost. The block below detects that and
REM asks - it never silently adopts or silently ignores the old folder (Codex audit Fix 2).
setlocal
set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
for %%I in ("%REPO%")     do set "LEAF=%%~nxI"
for %%I in ("%REPO%\..")  do set "PARENT=%%~fI"
set "VENV=%PARENT%\%LEAF%-venv"
set "DATA=%PARENT%\%LEAF%-data"
set "PREFLIGHT=%REPO%\scripts\nf-preflight.ps1"
set "LAUNCHLOG=%PARENT%\%LEAF%-launcher.log"

REM --- checkout-rename recovery (Codex audit Fix 2) ------------------------------
REM Only when the expected data folder does not exist yet. Look for exactly one
REM sibling "*-data" folder from a previous checkout name; if found, make the
REM operator choose - reuse it, or start fresh. No timeout, no default: this must
REM not be decided automatically.
if not exist "%DATA%\" call :find_prior_data
if not exist "%DATA%\" if "%PRIORCOUNT%"=="1" (
  echo.
  echo [north-forge] The data folder for this checkout name does not exist:
  echo [north-forge]     %DATA%
  echo [north-forge] but a data folder from a previous checkout name was found:
  echo [north-forge]     %PRIORDATA%
  echo [north-forge] Renaming the checkout folder changed the expected data-folder name.
  echo.
  echo   [R] Reuse "%PRIORDATA%"
  echo       ^(keep your existing conversations, config, memories, credentials^)
  echo   [N] Start fresh with a new empty "%DATA%"
  echo.
  choice /c RN /n /m "Choose [R/N]: "
  if errorlevel 255 (
    echo.
    echo [north-forge] Cannot prompt for a choice here. Re-run north-forge.cmd from a
    echo [north-forge] console, or rename the data folder yourself:
    echo [north-forge]     "%PRIORDATA%"  -^>  "%DATA%"
    exit /b 1
  )
  if errorlevel 2 (
    echo [north-forge] Starting fresh. The old folder is left untouched.
  ) else (
    set "DATA=%PRIORDATA%"
    echo [north-forge] Reusing "%PRIORDATA%".
  )
)

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

REM --- data folder MUST exist before we hand HERMES_HOME to hermes (Codex audit Fix 3) ---
REM `mkdir` failing silently (read-only drive, no media, full disk, damaged FS)
REM would set HERMES_HOME to a folder that isn't there and hermes would fail deep
REM in its own startup, looking to the operator like their saved data vanished.
if not exist "%DATA%\" mkdir "%DATA%" 2>nul
if not exist "%DATA%\" (
  echo.
  echo [north-forge] Could not create or open the drive-local data folder:
  echo [north-forge]     %DATA%
  echo [north-forge] The drive may be read-only, full, disconnected, or damaged.
  echo [north-forge] North Forge will not start without it. No data has been lost.
  pause
  exit /b 1
)
set "HERMES_HOME=%DATA%"

REM --- self-healing, refreshed every launch (cheap, idempotent, never fatal) ---
REM Keep HERMES_HOME's North Forge skin in sync with the checkout so a `git pull`
REM that updates the splash art takes effect without a re-bootstrap.
if not exist "%DATA%\skins\" mkdir "%DATA%\skins" 2>nul
if not exist "%DATA%\skins\" (
  echo [north-forge] warning: could not create "%DATA%\skins" - the North Forge skin may not load.
)
if exist "%REPO%\skins\north-forge.yaml" copy /Y "%REPO%\skins\north-forge.yaml" "%DATA%\skins\north-forge.yaml" >nul 2>&1
REM (Re)write "<drive>:\Start North Forge.lnk" against the path the checkout is at
REM right now, so it stays correct even if Windows re-letters the drive.
if exist "%REPO%\scripts\make-drive-root-shortcut.ps1" powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\scripts\make-drive-root-shortcut.ps1" -RepoRoot "%REPO%" -Quiet >nul 2>&1

REM --- North Forge tier check (CHG-2026-09-07-022) --------------------------------
REM If this drive carries a provisioning record and it has been tampered with,
REM stop here with a clear message rather than letting the in-process gate refuse
REM every subcommand. A missing record is normal (un-provisioned drive) -> exit 0.
REM Exit code 2 means EXACTLY "record present but its signature doesn't verify".
REM Any other non-zero (interpreter missing DLLs, module not importable, ...) is a
REM venv problem the preflight above owns, not a tamper - don't misreport it.
"%VENV%\Scripts\python.exe" -m hermes_cli.nf_tier verify >nul 2>&1
if "%ERRORLEVEL%"=="2" (
  echo.
  echo [north-forge] this drive's North Forge provisioning is invalid or was modified
  echo [north-forge] after Setup Run. An admin must repair it:  scripts\nf-setup.ps1 -Force
  "%VENV%\Scripts\python.exe" -m hermes_cli.nf_tier show
  pause
  exit /b 2
)

"%VENV%\Scripts\hermes.exe" %*
exit /b %ERRORLEVEL%

REM ===========================================================================
:find_prior_data
REM Sets PRIORCOUNT (count of sibling *-data folders that are NOT %DATA%) and
REM PRIORDATA (the last such folder). Called only when %DATA% is absent.
set "PRIORDATA="
set "PRIORCOUNT=0"
for /d %%D in ("%PARENT%\*-data") do call :consider_prior "%%~fD"
goto :eof

:consider_prior
if /I "%~1"=="%DATA%" goto :eof
set /a PRIORCOUNT+=1
set "PRIORDATA=%~1"
goto :eof
