# Start Go-based Hermes Desktop<->backend watchdog (operator-only; NOT agent-reachable).
param(
    [int]$IntervalSec = 20,
    [int]$FailThreshold = 2,
    [switch]$Once,
    [switch]$NoPrewarm,
    [switch]$NoTsnet,
    [string]$Listen = "127.0.0.1:9920",
    [string]$HermesRoot = "",
    [string]$HermesHome = "",
    [switch]$BuildIfMissing,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = if ($HermesRoot) { $HermesRoot } else { (Resolve-Path (Join-Path $ScriptDir "..\..")).Path }
if (-not $HermesHome) { $HermesHome = Join-Path $env:USERPROFILE ".hermes" }

$Exe = Join-Path $ScriptDir "watchdog-go\dist\hermes-watchdog.exe"
if (-not (Test-Path -LiteralPath $Exe)) {
    if ($BuildIfMissing) {
        & (Join-Path $ScriptDir "Build-HermesGoWatchdog.ps1")
    } else {
        throw "Missing $Exe — run Build-HermesGoWatchdog.ps1 first or pass -BuildIfMissing"
    }
}

$DataDir = Join-Path $env:LOCALAPPDATA "HermesWatchdog"
$LockPath = Join-Path $DataDir "watchdog.lock"

function Test-GoWatchdogAlive {
    if (-not (Test-Path -LiteralPath $LockPath)) { return $false }
    try {
        $obj = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
        $pidLock = [int]$obj.pid
        if ($pidLock -le 0) { return $false }
        $proc = Get-Process -Id $pidLock -ErrorAction SilentlyContinue
        return [bool]$proc
    } catch {
        return $false
    }
}

function Stop-GoWatchdog {
    if (Test-GoWatchdogAlive) {
        try {
            $obj = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
            Stop-Process -Id ([int]$obj.pid) -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        } catch {}
    }
    Get-Process -Name hermes-watchdog -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}

function Stop-PsDesktopBackendWatchdog {
    # PS and Go watchdogs use different lock files — running both causes dual
    # Hermes.exe relaunch loops. Prefer Go; stop the legacy PS mutual watchdog.
    $psLock = Join-Path $HermesHome "logs\desktop-backend-watchdog.lock"
    if (Test-Path -LiteralPath $psLock) {
        try {
            $obj = Get-Content -LiteralPath $psLock -Raw | ConvertFrom-Json
            if ($obj.pid) {
                Stop-Process -Id ([int]$obj.pid) -Force -ErrorAction SilentlyContinue
            }
        } catch {}
        Remove-Item -LiteralPath $psLock -Force -ErrorAction SilentlyContinue
    }
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'Start-HermesDesktopBackendWatchdog\.ps1'
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Stop-PsDesktopBackendWatchdog

if ($ForceRestart -or $Once) {
    Stop-GoWatchdog
} elseif (Test-GoWatchdogAlive) {
    Write-Host "Go watchdog already running (lock=$LockPath)"
    exit 0
}

$argList = @(
    "-interval=$IntervalSec",
    "-fail-threshold=$FailThreshold",
    "-hermes-root=`"$RepoRoot`"",
    "-hermes-home=`"$HermesHome`"",
    "-listen=$Listen"
)
if ($Once) { $argList += "-once" }
if ($NoPrewarm) { $argList += "-prewarm-backend=false" }
if (-not $NoTsnet -and ($env:HERMES_WATCHDOG_TS_AUTHKEY -or $env:TS_AUTHKEY)) {
    $argList += "-tsnet"
}

$env:HERMES_HOME = $HermesHome
$workDir = Split-Path -Parent $Exe
Write-Host "Starting Go watchdog detached: $Exe $($argList -join ' ')"

# UseShellExecute detaches from Cursor/agent job objects (see Start-HermesDesktopBackendWatchdog.ps1 header).
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $Exe
$startInfo.WorkingDirectory = $workDir
$startInfo.Arguments = ($argList -join ' ')
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$startInfo.UseShellExecute = $true
[void][System.Diagnostics.Process]::Start($startInfo)

Start-Sleep -Seconds 2
if (Test-GoWatchdogAlive) {
    Write-Host "Go watchdog launched (logs: $(Join-Path $HermesHome 'logs\hermes-go-watchdog.log'))"
} else {
    Write-Warning "Go watchdog may still be starting — check $(Join-Path $HermesHome 'logs\hermes-go-watchdog.log')"
}
