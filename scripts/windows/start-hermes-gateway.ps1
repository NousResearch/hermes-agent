param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe) -or -not (Test-Path -LiteralPath (Join-Path $RepoRoot ".venv\pyvenv.cfg"))) {
    $PythonExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonExe) -or -not (Test-Path -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PythonExe)) "pyvenv.cfg"))) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}

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

. (Join-Path $ScriptDir "Resolve-CanonicalHermesHome.ps1")

$HermesHome = Resolve-CanonicalHermesHome -RepoRoot $RepoRoot

$LogDir = Join-Path $HermesHome "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$env:HERMES_HOME = $HermesHome
$envFile = Join-Path $HermesHome ".env"
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        if ($line -match '^\s*#' -or $line -notmatch '=') {
            continue
        }
        $parts = $line -split '=', 2
        $name = $parts[0].Trim()
        $value = $parts[1]
        $preferHostEnvironment = $name -match '^(TELEGRAM|DISCORD|LINE)_' -or $name -match '^(TELEGRAM|DISCORD|LINE).*(TOKEN|SECRET|KEY|WEBHOOK)'
        if ($name -and $preferHostEnvironment -and [Environment]::GetEnvironmentVariable($name, "Process")) {
            continue
        }
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Avoid duplicate launches.
$isRunning = "0"
try {
    $isRunning = & $PythonExe -c "from gateway.status import is_gateway_running; print('1' if is_gateway_running() else '0')" 2>$null
} catch {
    $isRunning = "0"
}
if ($isRunning -eq "1") {
    exit 0
}

if ($DelaySeconds -gt 0) {
    Start-Sleep -Seconds $DelaySeconds
}

$llamaScript = Join-Path $ScriptDir "start-hermes-llama-fallback-rtx3060.ps1"
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

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$stdoutLog = Join-Path $LogDir "gateway-stdout.log"
$stderrLog = Join-Path $LogDir "gateway-stderr.log"

Start-Process `
    -FilePath $PythonExe `
    -ArgumentList @("-m", "hermes_cli.main", "gateway", "run") `
    -WorkingDirectory $RepoRoot `
    -WindowStyle $WindowStyle `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog
