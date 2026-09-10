#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Canonical test runner for hermes-agent on Windows.
    Run this instead of calling pytest directly to match CI behavior.

.DESCRIPTION
    Enforces:
      * Per-file subprocess isolation via run_tests_parallel.py (matches CI)
      * TZ=UTC, LANG=C.UTF-8 (approximation on Windows)
      * Credential env vars blanked
      * Proper venv activation

    Usage:
      scripts\run_tests.ps1                          # full suite
      scripts\run_tests.ps1 tests\agent\              # one directory
      scripts\run_tests.ps1 --tb=long -v              # pass-through pytest args
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$pytestArgs
)

$ErrorActionPreference = "Stop"

# Activate venv (prefer .venv, fall back to venv)
$venvPaths = @(
    Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
    Join-Path $PSScriptRoot "..\venv\Scripts\Activate.ps1"
)

$activated = $false
foreach ($vp in $venvPaths) {
    if (Test-Path $vp) {
        Write-Host "[run_tests.ps1] Activating $vp" -ForegroundColor Cyan
        . $vp
        $activated = $true
        break
    }
}

if (-not $activated) {
    Write-Host "[run_tests.ps1] No virtualenv found. Creating one..." -ForegroundColor Yellow
    $venvDir = Join-Path $PSScriptRoot "..\.venv"
    python -m venv $venvDir
    . (Join-Path $venvDir "Scripts\Activate.ps1")
    pip install uv
    uv sync --locked --extra all --extra dev
}

# Enforce deterministic environment
$env:TZ = "UTC"
$env:PYTHONHASHSEED = "0"
$env:LANG = "C.UTF-8"

# Blank credential env vars (belt-and-suspenders with conftest.py)
$credSuffixes = @("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_KEY", "_ID", "_CLIENT_ID", "_CLIENT_SECRET")
$env.Keys | Where-Object {
    foreach ($suffix in $credSuffixes) {
        if ($_ -like "*$suffix") { return $true }
    }
    return $false
} | ForEach-Object {
    Set-Item -Path "env:$_" -Value ""
}

# Ensure pytest is installed
$check = python -m pytest --version 2>&1
if ($LASTEXITCODE -ne 0) {
    pip install pytest pytest-asyncio
}

# Use run_tests_parallel.py for per-file subprocess isolation (matches CI).
# No xdist — fresh interpreter per file prevents cross-file state leakage.
$runner = Join-Path $PSScriptRoot "run_tests_parallel.py"

# Build args for the parallel runner
$runnerArgs = @("--jobs", "4", "--tb=short", "-v")
if ($pytestArgs) {
    $runnerArgs += $pytestArgs
}

Write-Host "[run_tests.ps1] Running: python $runner $($runnerArgs -join ' ')" -ForegroundColor Cyan
Write-Host "[run_tests.ps1] TZ=$env:TZ PYTHONHASHSEED=$env:PYTHONHASHSEED" -ForegroundColor Cyan

# Emit output directly — do not capture in a variable, so failing runs
# preserve their traceback and pytest summary.
python $runner @runnerArgs
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host "[run_tests.ps1] All tests passed." -ForegroundColor Green
} else {
    Write-Host "[run_tests.ps1] Tests failed with exit code $exitCode" -ForegroundColor Red
}

# Deactivate venv
if (Test-Path function:deactivate) {
    deactivate
}

exit $exitCode
