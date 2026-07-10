# Idempotent start for Obsidian memory-graph HTTP server (Go, LINE ngrok pattern).
# Serves output/ on 0.0.0.0 via bin/memory-graph-server.exe, regenerates HTML, exits quickly for Task Scheduler.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/start-obsidian-memory-graph-server.ps1
#   ... -Port 8765 -NoRegenerate -Watchdog -Rebuild

[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$NoRegenerate,
    [switch]$Watchdog,
    [switch]$Rebuild,
    [int]$WatchdogIntervalSec = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
. (Join-Path $ScriptDir "Resolve-CanonicalHermesHome.ps1")

$HermesHome = Resolve-CanonicalHermesHome -RepoRoot $RepoRoot
$OutputDir = Join-Path $RepoRoot "output"
$BinDir = Join-Path $RepoRoot "bin"
$ServerExe = Join-Path $BinDir "memory-graph-server.exe"
$BuildScript = Join-Path $ScriptDir "build-memory-graph-server.ps1"
$LogDir = Join-Path $HermesHome "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$StdoutLog = Join-Path $LogDir "memory-graph-server.log"
$StderrLog = Join-Path $LogDir "memory-graph-server.err.log"
$PidFile = Join-Path $LogDir "memory-graph-server.pid"

function Get-PythonExe {
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) { return @("py", "-3") }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) { return @("python") }
    throw "Python not found (py -3 or python)"
}

function Ensure-MemoryGraphServerBinary {
    if ($Rebuild -or -not (Test-Path -LiteralPath $ServerExe)) {
        if ($Rebuild) {
            & $BuildScript -Force
        } else {
            & $BuildScript
        }
    }
    if (-not (Test-Path -LiteralPath $ServerExe)) {
        throw "memory-graph-server.exe missing after build: $ServerExe"
    }
}

function Test-MemoryGraphServerRunning {
    param([int]$ListenPort)

    try {
        $conn = Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue
        if ($null -ne $conn) { return $true }
    } catch {}

    $running = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        if (-not $_.CommandLine) { return $false }
        if ($_.CommandLine -notmatch "memory-graph-server") { return $false }
        return $_.CommandLine -match [regex]::Escape(":$ListenPort")
    }
    return ($null -ne $running)
}

function Get-TailscaleDnsName {
    $tailscale = Get-Command tailscale.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if (-not $tailscale) { return $null }
    try {
        $json = & $tailscale status --json | ConvertFrom-Json
        $dns = [string]$json.Self.DNSName
        if ($dns) { return $dns.TrimEnd('.') }
    } catch {}
    return $null
}

function Sync-TailscaleServeScript {
    $src = Join-Path $ScriptDir "Update-HermesTailscaleServe.ps1"
    $destDir = Join-Path $env:LOCALAPPDATA "HermesWebUI"
    if (-not (Test-Path -LiteralPath $src)) { return }
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    Copy-Item -LiteralPath $src -Destination (Join-Path $destDir "Update-HermesTailscaleServe.ps1") -Force
}

function Start-MemoryGraphServer {
    param([int]$ListenPort)

    Ensure-MemoryGraphServerBinary

    $env:HERMES_MEMORY_GRAPH_ROOT = $OutputDir
    $argList = "-addr 0.0.0.0:$ListenPort"

    $proc = Start-Process `
        -FilePath $ServerExe `
        -ArgumentList $argList `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog

    Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
    Write-Host "Started memory-graph-server (Go) pid=$($proc.Id) port=$ListenPort dir=$OutputDir"
}

if (-not $NoRegenerate) {
    $graphScript = Join-Path $RepoRoot "scripts\obsidian_memory_graph.py"
    if (Test-Path -LiteralPath $graphScript) {
        Write-Host "Regenerating obsidian-memory-graph.html ..."
        $py = Get-PythonExe
        $genArgs = @()
        if ($py.Length -gt 1) { $genArgs += $py[1..($py.Length - 1)] }
        $genArgs += $graphScript
        & $py[0] @genArgs | Out-Host
    }
}

Sync-TailscaleServeScript

if (Test-MemoryGraphServerRunning -ListenPort $Port) {
    Write-Host "Already listening on port $Port"
} else {
    Start-MemoryGraphServer -ListenPort $Port
}

$tsDns = Get-TailscaleDnsName
Write-Host "Local:  http://127.0.0.1:$Port/obsidian-memory-graph.html"
Write-Host "Health: http://127.0.0.1:$Port/health"
Write-Host "LAN:    http://<pc-ip>:$Port/obsidian-memory-graph.html"
if ($tsDns) {
    Write-Host "Tailscale: https://$tsDns/memory-graph/obsidian-memory-graph.html"
}
Write-Host "Dashboard: http://127.0.0.1:9120/memory-graph/obsidian-memory-graph.html"

if (-not $Watchdog) {
    exit 0
}

Write-Host "Watchdog active (every ${WatchdogIntervalSec}s). Ctrl+C to stop."
while ($true) {
    Start-Sleep -Seconds $WatchdogIntervalSec
    if (-not (Test-MemoryGraphServerRunning -ListenPort $Port)) {
        Write-Host "[watchdog] port $Port down — restarting"
        Start-MemoryGraphServer -ListenPort $Port
    }
}
