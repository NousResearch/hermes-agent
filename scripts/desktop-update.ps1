# desktop-update.ps1 -- repo-owned Windows Desktop update hand-off.
#
# WHY THIS EXISTS (the frozen-binary problem): the Desktop's Update button
# used to hand off exclusively to the staged Tauri binary
# (%HERMES_HOME%\hermes-setup.exe). That binary has no self-update path --
# copy_self_to_hermes_home deliberately no-ops during --update -- so every
# updater-side fix (cache refresh #67369, marker self-adopt #74782, straggler
# handling) only reaches users when a new installer is built, signed, and
# published. In practice binaries go months stale and users hit long-fixed
# bugs on every update (the 2026-08-09 incident chain).
#
# This script inverts that: it lives in the repo checkout, so EVERY
# `hermes update` refreshes the very code that drives the next update. The
# Desktop spawns it detached (see resolveUpdateScriptHandoff in
# apps/desktop/electron/main.ts) and exits; only PowerShell itself -- an OS
# component -- is "frozen".
#
# CONTRACT (keep in sync with apps/desktop/electron/main.ts):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\desktop-update.ps1
#     -InstallRoot <path>   repo checkout (HERMES_HOME\hermes-agent)
#     -Branch <ref>         branch to update against
#     -DesktopPid <pid>     the Electron main process to wait out
#     [-RelaunchExe <path>] Hermes.exe to start when done (omit = no relaunch)
#     [-NoMarkerCleanup]    leave .hermes-update-in-progress in place (tests)
#
# The Desktop pre-writes HERMES_HOME\.hermes-update-in-progress with THIS
# process's pid before quitting. Contracts that already exist make that safe:
#   * hermes_cli/update_lock.py `acquire` treats a live marker owned by a
#     process ANCESTOR as its own orchestrator -- our `hermes update` child
#     adopts the claim instead of refusing (no HANDOFF_PID_ENV needed).
#   * electron/update-marker.ts gates backend startup on the marker, so a
#     relaunched Desktop parks instead of spawning a venv-locking backend
#     into the update window.
# We delete the marker on every exit path; a crash self-heals via the
# 20-minute staleness ceiling both readers enforce.

param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [string]$Branch = "main",
    [int]$DesktopPid = 0,
    [string]$RelaunchExe = "",
    [switch]$NoMarkerCleanup
)

$ErrorActionPreference = "Continue"
$HermesHome = Split-Path -Parent $InstallRoot
$MarkerPath = Join-Path $HermesHome ".hermes-update-in-progress"
$LogPath = Join-Path $HermesHome "logs\desktop-update-handoff.log"

function Write-HandoffLog([string]$Message) {
    $line = "{0:yyyy-MM-ddTHH:mm:ssK} {1}" -f (Get-Date), $Message
    try { Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8 } catch {}
    Write-Host $line
}

function Remove-Marker {
    if ($NoMarkerCleanup) { return }
    try {
        if (Test-Path -LiteralPath $MarkerPath) {
            Remove-Item -LiteralPath $MarkerPath -Force -ErrorAction SilentlyContinue
            Write-HandoffLog "removed update marker"
        }
    } catch {}
}

try {
    Write-HandoffLog "hand-off start: root=$InstallRoot branch=$Branch desktopPid=$DesktopPid pid=$PID"

    # -- 1. Wait for the Desktop to actually exit -------------------------
    # The Desktop quits right after spawning us, but Electron teardown is
    # asynchronous. Bounded wait; a Desktop that never exits is a bug we
    # surface rather than fight.
    if ($DesktopPid -gt 0) {
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $proc = Get-Process -Id $DesktopPid -ErrorAction SilentlyContinue
            if (-not $proc) { break }
            Start-Sleep -Milliseconds 300
        }
        $still = Get-Process -Id $DesktopPid -ErrorAction SilentlyContinue
        if ($still) {
            Write-HandoffLog "WARNING: desktop pid $DesktopPid still alive after 30s; proceeding (hermes update has its own guards)"
        } else {
            Write-HandoffLog "desktop exited"
        }
    }

    # -- 2. Wait for the venv shim to unlock ------------------------------
    # Mirrors the Rust updater's is_locked(): a running exe refuses O_RDWR.
    $shim = Join-Path $InstallRoot "venv\Scripts\hermes.exe"
    if (Test-Path -LiteralPath $shim) {
        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline) {
            try {
                $fs = [System.IO.File]::Open($shim, 'Open', 'ReadWrite', 'None')
                $fs.Close()
                Write-HandoffLog "venv shim unlocked"
                break
            } catch {
                Start-Sleep -Milliseconds 400
            }
        }
    }

    # -- 3. Run the update from the CURRENT checkout ----------------------
    # hermes update handles everything downstream: gateway pause, venv-holder
    # guard (with orphan reap), dep sync, desktop rebuild, skills/config sync,
    # gateway restart. --force skips only the hermes.exe shim guard, which by
    # this point is provably unlocked (step 2); the venv-python holder guard
    # stays active. Our marker claim is adopted by the child via the
    # process-ancestry rule in hermes_cli/update_lock.py.
    $hermesExe = Join-Path $InstallRoot "venv\Scripts\hermes.exe"
    if (-not (Test-Path -LiteralPath $hermesExe)) {
        Write-HandoffLog "ERROR: $hermesExe missing - install too broken for the script hand-off; falling back is the Desktop's job"
        exit 3
    }
    Write-HandoffLog "running: hermes update --yes --gateway --force --branch $Branch"
    & $hermesExe update --yes --gateway --force --branch $Branch 2>&1 | ForEach-Object {
        Write-HandoffLog ("update| " + $_)
    }
    $updateExit = $LASTEXITCODE
    Write-HandoffLog "hermes update exit code: $updateExit"

    if ($updateExit -ne 0) {
        # One retry for the update-boundary class (fresh code on disk, stale
        # code in memory -- same rationale as the Tauri updater's retry).
        # Skip for exit 2: "close all Hermes windows" is not retryable.
        if ($updateExit -ne 2) {
            Write-HandoffLog "first attempt failed; retrying once (freshly pulled fix loads on the second run)"
            & $hermesExe update --yes --gateway --force --branch $Branch 2>&1 | ForEach-Object {
                Write-HandoffLog ("update| " + $_)
            }
            $updateExit = $LASTEXITCODE
            Write-HandoffLog "retry exit code: $updateExit"
        }
    }

    # -- 4. Relaunch the Desktop ------------------------------------------
    # Marker must be gone BEFORE relaunch or the new Desktop parks on it.
    Remove-Marker
    if ($RelaunchExe -and (Test-Path -LiteralPath $RelaunchExe)) {
        Write-HandoffLog "relaunching desktop: $RelaunchExe"
        Start-Process -FilePath $RelaunchExe -WorkingDirectory (Split-Path -Parent $RelaunchExe) | Out-Null
    }

    exit $updateExit
} finally {
    Remove-Marker
}
