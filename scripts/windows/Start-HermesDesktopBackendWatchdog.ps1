# Mutual watchdog: packaged Hermes Desktop <-> desktop-spawned hermes serve backend.
# - If Hermes.exe dies → relaunch packaged Desktop (which respawns serve).
# - If Desktop lives but /api/status on its serve child fails → restart Desktop.
# - Orphan HERMES_DESKTOP serve processes without a parent Hermes.exe are reaped.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Start-HermesDesktopBackendWatchdog.ps1
#   ... -Once          # single probe (exit after one cycle)
#   ... -IntervalSec 20
#
# Detach from IDE/agent job objects (required under Cursor terminals):
#   cmd /c start "HermesWd" /MIN powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this> ...
# Do not rely on Start-Process from a Cursor-managed shell alone — the child may die with the job.

[CmdletBinding()]
param(
    [int]$IntervalSec = 20,
    [int]$FailThreshold = 2,
    [switch]$Once,
    [string]$HermesRoot = "",
    [string]$HermesHome = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = if ($HermesRoot) { $HermesRoot } else { (Resolve-Path (Join-Path $ScriptDir "..\..")).Path }
if (-not $HermesHome) { $HermesHome = Join-Path $env:USERPROFILE ".hermes" }

$LogDir = Join-Path $HermesHome "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "desktop-backend-watchdog.log"
$LockPath = Join-Path $LogDir "desktop-backend-watchdog.lock"
$StatePath = Join-Path $LogDir "desktop-backend-watchdog.state.json"

$PackagedExe = Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe"
if (-not (Test-Path -LiteralPath $PackagedExe)) {
    $PackagedExe = Join-Path $RepoRoot "apps\desktop\release\win-unpacked\Hermes.exe"
}

function Write-WdLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-WatchdogLock {
    if (-not (Test-Path -LiteralPath $LockPath)) { return $false }
    try {
        $raw = Get-Content -LiteralPath $LockPath -Raw -ErrorAction Stop
        $obj = $raw | ConvertFrom-Json
        $pidLock = [int]$obj.pid
        $proc = Get-Process -Id $pidLock -ErrorAction SilentlyContinue
        if ($proc) { return $true }
    } catch {}
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    return $false
}

function Enter-WatchdogLock {
    if (Test-WatchdogLock) {
        Write-WdLog "another watchdog already holds $LockPath — exiting"
        exit 0
    }
    @{
        pid       = $PID
        startedAt = (Get-Date).ToString("o")
        repoRoot  = $RepoRoot
    } | ConvertTo-Json | Set-Content -LiteralPath $LockPath -Encoding UTF8
}

function Exit-WatchdogLock {
    try {
        if (Test-Path -LiteralPath $LockPath) {
            $obj = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
            if ([int]$obj.pid -eq $PID) {
                Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}
}

function Get-DesktopProcesses {
    return @(Get-Process -Name Hermes -ErrorAction SilentlyContinue)
}

function Get-DesktopBackendCandidates {
    # Children launched by Desktop: hermes serve / dashboard --no-open with HERMES_DESKTOP.
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        if (-not $_.CommandLine) { return $false }
        $cl = $_.CommandLine
        if ($cl -notmatch 'hermes_cli\.main|\\hermes\.exe|Scripts\\hermes\.exe') { return $false }
        if ($cl -match '\bserve\b') { return $true }
        if ($cl -match 'dashboard' -and $cl -match '--no-open') { return $true }
        return $false
    }
}

function Get-ListeningPortsForPid([int]$ProcessId) {
    @(Get-NetTCPConnection -OwningProcess $ProcessId -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty LocalPort -Unique)
}

# Stack-owned ports — never treat as Desktop ephemeral hermes serve / never reap.
$script:ReservedOpsPorts = @(8080, 8081, 8646, 8765, 8787, 9119, 9120, 9920, 18794)

function Test-ReservedOpsPort([int]$Port) {
    return $script:ReservedOpsPorts -contains $Port
}

function Test-BackendStatus([int]$Port) {
    if (Test-ReservedOpsPort -Port $Port) { return $false }
    try {
        $code = & curl.exe -s -m 3 -o NUL -w "%{http_code}" "http://127.0.0.1:$Port/api/status"
        return ($code -eq "200")
    } catch {
        return $false
    }
}

function Find-HealthyDesktopBackend {
    foreach ($proc in (Get-DesktopBackendCandidates)) {
        foreach ($port in (Get-ListeningPortsForPid -ProcessId $proc.ProcessId)) {
            if ($port -and (Test-BackendStatus -Port $port)) {
                return [PSCustomObject]@{
                    Pid  = $proc.ProcessId
                    Port = [int]$port
                    Cmd  = $proc.CommandLine
                }
            }
        }
    }
    return $null
}

function Stop-OrphanDesktopBackends {
    $desktop = Get-DesktopProcesses
    if ($desktop.Count -gt 0) { return 0 }
    $n = 0
    foreach ($proc in (Get-DesktopBackendCandidates)) {
        $ports = @(Get-ListeningPortsForPid -ProcessId $proc.ProcessId)
        $skip = $false
        foreach ($port in $ports) {
            if (Test-ReservedOpsPort -Port ([int]$port)) {
                Write-WdLog "skip reap pid=$($proc.ProcessId) (ops port $port)"
                $skip = $true
                break
            }
        }
        if ($skip) { continue }
        Write-WdLog "reaping orphan backend pid=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $n++
    }
    return $n
}

function Start-PackagedDesktop {
    if (-not (Test-Path -LiteralPath $PackagedExe)) {
        Write-WdLog "Hermes.exe missing at $PackagedExe"
        return $false
    }
    $env:HERMES_HOME = $HermesHome
    $env:HERMES_DESKTOP_HERMES_ROOT = $RepoRoot
    $env:HERMES_DESKTOP_CWD = $RepoRoot
    $work = Split-Path -Parent $PackagedExe
    Start-Process -FilePath $PackagedExe -WorkingDirectory $work | Out-Null
    Write-WdLog "launched $PackagedExe"
    return $true
}

function Restart-PackagedDesktop {
    Write-WdLog "restarting Desktop (force backend respawn)"
    Get-DesktopProcesses | ForEach-Object {
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
    Stop-OrphanDesktopBackends | Out-Null
    Start-Sleep -Seconds 1
    return Start-PackagedDesktop
}

function Save-WatchdogState($state) {
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Invoke-WatchdogCycle([ref]$failCount) {
    $desktop = Get-DesktopProcesses
    $backend = Find-HealthyDesktopBackend

    if ($desktop.Count -eq 0) {
        Stop-OrphanDesktopBackends | Out-Null
        Write-WdLog "Desktop DOWN — relaunch"
        Start-PackagedDesktop | Out-Null
        $failCount.Value = 0
        return @{ desktop = "relaunched"; backend = "pending" }
    }

    if (-not $backend) {
        $failCount.Value++
        Write-WdLog "Desktop UP (pids=$(($desktop | ForEach-Object Id) -join ',')) but backend DOWN (fail=$($failCount.Value)/$FailThreshold)"
        if ($failCount.Value -ge $FailThreshold) {
            Restart-PackagedDesktop | Out-Null
            $failCount.Value = 0
            return @{ desktop = "restarted"; backend = "respawning" }
        }
        return @{ desktop = "up"; backend = "down" }
    }

    $failCount.Value = 0
    Write-WdLog "OK desktop=$(($desktop | ForEach-Object Id) -join ',') backend=pid:$($backend.Pid) port:$($backend.Port)"
    return @{
        desktop = "up"
        backend = "up"
        backendPid = $backend.Pid
        backendPort = $backend.Port
    }
}

Enter-WatchdogLock
try {
    Write-WdLog "watchdog start interval=${IntervalSec}s threshold=$FailThreshold exe=$PackagedExe"
    $fails = 0
    do {
        $result = Invoke-WatchdogCycle -failCount ([ref]$fails)
        Save-WatchdogState @{
            updatedAt = (Get-Date).ToString("o")
            watchdogPid = $PID
            result = $result
            consecutiveBackendFails = $fails
        }
        if ($Once) { break }
        Start-Sleep -Seconds $IntervalSec
    } while ($true)
}
finally {
    Exit-WatchdogLock
    Write-WdLog "watchdog stop"
}
