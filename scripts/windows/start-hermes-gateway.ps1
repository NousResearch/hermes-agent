param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

$DelaySeconds = 30
$WindowStyle = "Normal"
if ($env:HERMES_STARTUP_DELAY_SECONDS -and $env:HERMES_STARTUP_DELAY_SECONDS.Trim()) {
    $parsedDelay = 0
    if ([int]::TryParse($env:HERMES_STARTUP_DELAY_SECONDS, [ref]$parsedDelay) -and $parsedDelay -ge 0) {
        $DelaySeconds = $parsedDelay
    }
}
if ($env:HERMES_GATEWAY_WINDOW_STYLE -and $env:HERMES_GATEWAY_WINDOW_STYLE.Trim()) {
    $candidate = $env:HERMES_GATEWAY_WINDOW_STYLE.Trim()
    if ($candidate -in @("Normal", "Minimized", "Maximized", "Hidden")) {
        $WindowStyle = $candidate
    }
}

$HermesHome = if ($env:HERMES_HOME -and $env:HERMES_HOME.Trim()) {
    $env:HERMES_HOME
} else {
    Join-Path $env:USERPROFILE ".hermes"
}

$LogDir = Join-Path $HermesHome "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Avoid duplicate launches.
$isRunning = "0"
try {
    $isRunning = & py -3 -c "from gateway.status import is_gateway_running; print('1' if is_gateway_running() else '0')" 2>$null
} catch {
    $isRunning = "0"
}
if ($isRunning -eq "1") {
    exit 0
}

if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
}

$llamaScript = Join-Path $ScriptDir "start-hermes-llama-fallback-rtx3080.ps1"
if (-not (Test-Path -LiteralPath $llamaScript)) {
    $llamaScript = Join-Path $ScriptDir "start-hermes-llama-fallback.ps1"
}
if (Test-Path -LiteralPath $llamaScript) {
    try {
        & $llamaScript | Out-Null
    } catch {
        Write-Warning "llama.cpp fallback autostart failed: $_"
    }
}

Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList "/c", "set PYTHONIOENCODING=utf-8&& set PYTHONUTF8=1&& py -3 -m hermes_cli gateway run" `
    -WorkingDirectory $RepoRoot `
    -WindowStyle $WindowStyle
