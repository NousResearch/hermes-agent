# Mutual watchdog: packaged Hermes Desktop <-> desktop-spawned hermes serve backend.
# Prefer Start-HermesGoWatchdog.ps1 (managed :9118 + desktop-backend.json). This script
# remains for environments without the Go binary and mirrors its backend discovery order.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\Start-HermesDesktopBackendWatchdog.ps1
#   ... -Once
#   ... -IntervalSec 20
#
# Detach from IDE/agent job objects (required under Cursor terminals):
#   cmd /c start "" /MIN powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this> ...

[CmdletBinding()]
param(
    [int]$IntervalSec = 20,
    [int]$FailThreshold = 2,
    [int]$StartupGraceSec = 45,
    [int]$ManagedBackendPort = 9118,
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
$WatchdogDataDir = Join-Path $env:LOCALAPPDATA "HermesWatchdog"
$ManifestPath = Join-Path $WatchdogDataDir "desktop-backend.json"
$DesktopLogPath = Join-Path $LogDir "desktop.log"

$PackagedExe = Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe"
if (-not (Test-Path -LiteralPath $PackagedExe)) {
    $PackagedExe = Join-Path $RepoRoot "apps\desktop\release\win-unpacked\Hermes.exe"
}

# Stack-owned listeners — never reap; skip when scanning ephemeral Desktop serve.
$script:ReservedOpsPorts = @(8080, 8081, 8646, 8765, 8787, 9119, 9120, 9920, 18794)

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

function Test-ReservedOpsPort([int]$Port) {
    return $script:ReservedOpsPorts -contains $Port
}

function Test-BackendStatus([int]$Port) {
    if ($Port -le 0) { return $false }
    if (Test-ReservedOpsPort -Port $Port) { return $false }
    try {
        $code = & curl.exe -s -m 3 -o NUL -w "%{http_code}" "http://127.0.0.1:$Port/api/status"
        return ($code -eq "200")
    } catch {
        return $false
    }
}

function Get-ListeningPortsForPid([int]$ProcessId) {
    $ports = [System.Collections.Generic.HashSet[int]]::new()
    try {
        $out = & netstat.exe -ano -p tcp 2>$null
        if (-not $out) { return @() }
        $target = [string]$ProcessId
        foreach ($line in $out) {
            $trimmed = $line.Trim()
            if ($trimmed -notmatch '\sLISTENING\s') { continue }
            $fields = $trimmed -split '\s+', 0, 'SimpleMatch'
            if ($fields.Count -lt 5) { continue }
            if ($fields[-1] -ne $target) { continue }
            $hostPort = $fields[1]
            $idx = $hostPort.LastIndexOf(':')
            if ($idx -lt 0) { continue }
            $port = 0
            if ([int]::TryParse($hostPort.Substring($idx + 1), [ref]$port) -and $port -gt 0) {
                [void]$ports.Add($port)
            }
        }
    } catch {}
    return @($ports)
}

function Test-DesktopBackendCommandLine([string]$CommandLine) {
    if (-not $CommandLine) { return $false }
    $cl = $CommandLine
    $lower = $cl.ToLowerInvariant()
    if ($cl -notmatch 'hermes_cli\.main|\\hermes\.exe|Scripts\\hermes\.exe') { return $false }
    if ($lower -match '\s gateway|\s harness|\s cron') { return $false }
    if ($cl -match '--port\s+9120|--port=9120|--port\s+8787|--port=8787') { return $false }
    if ($cl -match '\bserve\b') { return $true }
    if ($cl -match 'dashboard' -and $cl -match '--no-open') { return $true }
    return $false
}

function Get-DesktopBackendCandidates {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        Test-DesktopBackendCommandLine -CommandLine $_.CommandLine
    }
}

function Find-ManifestBackend {
    foreach ($path in @($ManifestPath, (Join-Path $LogDir "desktop-backend.json"))) {
        if (-not (Test-Path -LiteralPath $path)) { continue }
        try {
            $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
            $port = [int]$manifest.port
            if ($port -le 0 -and $manifest.baseUrl) {
                if ($manifest.baseUrl -match ':(\d+)\s*$') {
                    $port = [int]$Matches[1]
                }
            }
            if ($port -le 0 -or (Test-ReservedOpsPort -Port $port)) { continue }
            if ($manifest.pid -and -not (Get-Process -Id ([int]$manifest.pid) -ErrorAction SilentlyContinue)) { continue }
            if (Test-BackendStatus -Port $port) {
                return [PSCustomObject]@{
                    Pid  = if ($manifest.pid) { [int]$manifest.pid } else { 0 }
                    Port = $port
                    Cmd  = "manifest:$path"
                }
            }
        } catch {}
    }
    return $null
}

function Find-LatestDesktopLogBackendPort {
    if (-not (Test-Path -LiteralPath $DesktopLogPath)) { return $null }
    try {
        $tail = Get-Content -LiteralPath $DesktopLogPath -Tail 400 -ErrorAction Stop
        for ($i = $tail.Count - 1; $i -ge 0; $i--) {
            if ($tail[$i] -match 'HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)') {
                $port = [int]$Matches[1]
                if ($port -gt 0 -and -not (Test-ReservedOpsPort -Port $port) -and (Test-BackendStatus -Port $port)) {
                    return $port
                }
            }
        }
    } catch {}
    return $null
}

function Find-HealthyDesktopBackend {
    $manifest = Find-ManifestBackend
    if ($manifest) { return $manifest }

    if ($ManagedBackendPort -gt 0 -and -not (Test-ReservedOpsPort -Port $ManagedBackendPort)) {
        if (Test-BackendStatus -Port $ManagedBackendPort) {
            return [PSCustomObject]@{
                Pid  = 0
                Port = $ManagedBackendPort
                Cmd  = "managed-port"
            }
        }
    }

    $logPort = Find-LatestDesktopLogBackendPort
    if ($logPort) {
        return [PSCustomObject]@{
            Pid  = 0
            Port = $logPort
            Cmd  = "desktop.log"
        }
    }

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
    $desktop = @(Get-DesktopProcesses)
    if ($desktop.Count -gt 0) { return 0 }
    $n = 0
    foreach ($proc in (Get-DesktopBackendCandidates)) {
        $ports = @(Get-ListeningPortsForPid -ProcessId $proc.ProcessId)
        $skip = $false
        foreach ($port in $ports) {
            if ([int]$port -eq $ManagedBackendPort) {
                Write-WdLog "skip reap pid=$($proc.ProcessId) (managed port $port)"
                $skip = $true
                break
            }
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
    $manifest = Find-ManifestBackend
    if ($manifest -and (Test-Path -LiteralPath $ManifestPath)) {
        try {
            $raw = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
            if ($raw.baseUrl -and $raw.token) {
                $env:HERMES_DESKTOP_REMOTE_URL = [string]$raw.baseUrl
                $env:HERMES_DESKTOP_REMOTE_TOKEN = [string]$raw.token
            }
        } catch {}
    }
    $work = Split-Path -Parent $PackagedExe
    Start-Process -FilePath $PackagedExe -WorkingDirectory $work | Out-Null
    Write-WdLog "launched $PackagedExe"
    return $true
}

function Stop-DesktopProcessTrees {
    # Tree-kill so Electron helpers and desktop-spawned hermes serve children die
    # with the main process (plain Stop-Process skips before-quit cleanup).
    foreach ($proc in @(Get-DesktopProcesses)) {
        Write-WdLog "tree-killing Hermes.exe pid=$($proc.Id)"
        & taskkill.exe /PID $proc.Id /T /F 2>$null | Out-Null
    }
    & taskkill.exe /IM Hermes.exe /T /F 2>$null | Out-Null
}

function Restart-PackagedDesktop {
    Write-WdLog "restarting Desktop (force backend respawn)"
    Stop-DesktopProcessTrees
    Start-Sleep -Seconds 2
    Stop-OrphanDesktopBackends | Out-Null
    Start-Sleep -Seconds 1
    return Start-PackagedDesktop
}

function Load-WatchdogState {
    if (-not (Test-Path -LiteralPath $StatePath)) { return @{} }
    try {
        return (Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json)
    } catch {
        return @{}
    }
}

function Save-WatchdogState($state) {
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Test-InStartupGrace {
    param([object]$State)
    if (-not $State.lastDesktopLaunch) { return $false }
    try {
        $launched = [datetime]$State.lastDesktopLaunch
        return ((Get-Date) - $launched).TotalSeconds -lt $StartupGraceSec
    } catch {
        return $false
    }
}

function Invoke-WatchdogCycle([ref]$failCount, [ref]$state) {
    $desktop = @(Get-DesktopProcesses)
    $backend = Find-HealthyDesktopBackend

    if ($desktop.Count -eq 0) {
        Stop-OrphanDesktopBackends | Out-Null
        Write-WdLog "Desktop DOWN — relaunch"
        if (Start-PackagedDesktop) {
            $state.Value.lastDesktopLaunch = (Get-Date).ToString("o")
        }
        $failCount.Value = 0
        return @{ desktop = "relaunched"; backend = "pending" }
    }

    if (-not $backend) {
        if (Test-InStartupGrace -State $state.Value) {
            Write-WdLog "Desktop UP (pids=$(($desktop | ForEach-Object Id) -join ',')) backend pending (startup grace ${StartupGraceSec}s)"
            return @{ desktop = "up"; backend = "warming" }
        }
        $failCount.Value++
        Write-WdLog "Desktop UP (pids=$(($desktop | ForEach-Object Id) -join ',')) but backend DOWN (fail=$($failCount.Value)/$FailThreshold)"
        if ($failCount.Value -ge $FailThreshold) {
            Restart-PackagedDesktop | Out-Null
            $state.Value.lastDesktopLaunch = (Get-Date).ToString("o")
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
    Write-WdLog "watchdog start interval=${IntervalSec}s threshold=$FailThreshold grace=${StartupGraceSec}s managedPort=$ManagedBackendPort exe=$PackagedExe"
    $fails = 0
    $cycleState = Load-WatchdogState
    if (-not $cycleState) { $cycleState = @{} }
    do {
        $result = Invoke-WatchdogCycle -failCount ([ref]$fails) -state ([ref]$cycleState)
        Save-WatchdogState @{
            updatedAt = (Get-Date).ToString("o")
            watchdogPid = $PID
            result = $result
            consecutiveBackendFails = $fails
            lastDesktopLaunch = $cycleState.lastDesktopLaunch
        }
        if ($Once) { break }
        Start-Sleep -Seconds $IntervalSec
    } while ($true)
}
finally {
    Exit-WatchdogLock
    Write-WdLog "watchdog stop"
}
