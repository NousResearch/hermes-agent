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
    [switch]$BuildIfMissing
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

$args = @(
    "-interval=$IntervalSec",
    "-fail-threshold=$FailThreshold",
    "-hermes-root=`"$RepoRoot`"",
    "-hermes-home=`"$HermesHome`"",
    "-listen=$Listen"
)
if ($Once) { $args += "-once" }
if ($NoPrewarm) { $args += "-prewarm-backend=false" }
if (-not $NoTsnet -and ($env:HERMES_WATCHDOG_TS_AUTHKEY -or $env:TS_AUTHKEY)) {
    $args += "-tsnet"
}

$env:HERMES_HOME = $HermesHome
Write-Host "Starting Go watchdog: $Exe $($args -join ' ')"
Start-Process -FilePath $Exe -ArgumentList $args -WindowStyle Hidden -WorkingDirectory (Split-Path -Parent $Exe) | Out-Null
Write-Host "Go watchdog launched (logs: $(Join-Path $HermesHome 'logs\hermes-go-watchdog.log'))"
