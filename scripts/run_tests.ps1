param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArgs
)

# Native Windows companion to run_tests.sh. Keep environment policy here in
# sync with that launcher; run_tests_parallel.py owns discovery and execution.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = $null
$SkippedVenvs = @()

function Test-PytestAvailable([string]$CandidatePython) {
    # Windows PowerShell 5.1 promotes native stderr to NativeCommandError when
    # ErrorActionPreference is Stop, even when stderr is redirected. Probe
    # under Continue and decide solely from the native process exit code.
    $PreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $CandidatePython -c "import pytest" 2>$null
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $PreviousPreference
    }
}

$Candidates = @(
    (Join-Path $RepoRoot ".venv"),
    (Join-Path $RepoRoot "venv")
)
if ($env:LOCALAPPDATA) {
    $Candidates += Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\venv"
}
if ($env:HOME) {
    $Candidates += Join-Path $env:HOME ".hermes\hermes-agent\venv"
}

foreach ($Candidate in ($Candidates | Select-Object -Unique)) {
    $CandidatePython = Join-Path $Candidate "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $CandidatePython -PathType Leaf)) {
        continue
    }
    if (Test-PytestAvailable $CandidatePython) {
        $Python = $CandidatePython
        break
    }
    $SkippedVenvs += $Candidate
}

if (-not $Python -and $env:HERMES_PYTHON -and
    (Test-Path -LiteralPath $env:HERMES_PYTHON -PathType Leaf)) {
    if (Test-PytestAvailable $env:HERMES_PYTHON) {
        $Python = $env:HERMES_PYTHON
        Write-Host "> no local venv - using HERMES_PYTHON: $Python"
    }
}

foreach ($Skipped in $SkippedVenvs) {
    [Console]::Error.WriteLine("> skipping venv without pytest: $Skipped")
}
if (-not $Python) {
    [Console]::Error.WriteLine(
        "error: no Windows virtualenv with pytest found under .venv, venv, or the managed Hermes install"
    )
    exit 1
}

# Capture opt-in values before clearing the process environment. PowerShell
# 5.1 has no equivalent of `env -i`, so temporarily replace the process
# environment with the same explicit allowlist inherited by the Python child.
$OriginalEnvironment = @{}
foreach ($Entry in [Environment]::GetEnvironmentVariables("Process").GetEnumerator()) {
    $OriginalEnvironment[[string]$Entry.Key] = [string]$Entry.Value
}

$AllowedNames = @(
    "PATH", "HOME",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA",
    "SYSTEMROOT", "SystemDrive", "ComSpec", "TEMP", "TMP",
    "HERMES_TEST_IMAGE", "HERMES_TEST_WORKERS", "HERMES_TEST_PATHS",
    "HERMES_TEST_FILE_TIMEOUT", "HERMES_TEST_FILE_RETRIES", "HERMES_TEST_SLICE",
    "HERMES_RUN_SLOW_PET_TESTS", "HERMES_E2E_BROWSER"
)
$CleanEnvironment = @{}
foreach ($Name in $AllowedNames) {
    if ($OriginalEnvironment.ContainsKey($Name) -and $OriginalEnvironment[$Name]) {
        $CleanEnvironment[$Name] = $OriginalEnvironment[$Name]
    }
}

$HermesHome = if ($CleanEnvironment.ContainsKey("HOME")) {
    Join-Path $CleanEnvironment["HOME"] ".hermes"
} elseif ($CleanEnvironment.ContainsKey("USERPROFILE")) {
    Join-Path $CleanEnvironment["USERPROFILE"] ".hermes"
} else {
    $null
}
if ($HermesHome) {
    $LiveGuard = Join-Path $HermesHome "pytest_live_guard.py"
    if (Test-Path -LiteralPath $LiveGuard -PathType Leaf) {
        $CleanEnvironment["PYTHONPATH"] = $HermesHome
        $CleanEnvironment["PYTEST_PLUGINS"] = "pytest_live_guard"
    }
}

$CleanEnvironment["TZ"] = "UTC"
$CleanEnvironment["LANG"] = "C.UTF-8"
$CleanEnvironment["LC_ALL"] = "C.UTF-8"
$CleanEnvironment["PYTHONHASHSEED"] = "0"
$CleanEnvironment["PYTHONUTF8"] = "1"

Write-Host "> running per-file parallel test suite via run_tests_parallel.py"
Write-Host "  (TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0; clean env)"

try {
    foreach ($Name in @([Environment]::GetEnvironmentVariables("Process").Keys)) {
        [Environment]::SetEnvironmentVariable([string]$Name, $null, "Process")
    }
    foreach ($Entry in $CleanEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            [string]$Entry.Key, [string]$Entry.Value, "Process"
        )
    }

    Push-Location $RepoRoot
    try {
        & $Python (Join-Path $PSScriptRoot "run_tests_parallel.py") @RunnerArgs
        $RunnerExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    foreach ($Name in @([Environment]::GetEnvironmentVariables("Process").Keys)) {
        [Environment]::SetEnvironmentVariable([string]$Name, $null, "Process")
    }
    foreach ($Entry in $OriginalEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            [string]$Entry.Key, [string]$Entry.Value, "Process"
        )
    }
}

exit $RunnerExitCode
