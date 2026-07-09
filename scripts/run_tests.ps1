param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PytestArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

$candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot "venv\Scripts\python.exe"),
    (Join-Path $HOME ".hermes\hermes-agent\venv\Scripts\python.exe")
)

$python = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
    throw "No virtualenv Python found in .venv\Scripts, venv\Scripts, or ~/.hermes/hermes-agent/venv/Scripts."
}

$env:TZ = "UTC"
$env:LANG = "C.UTF-8"
$env:LC_ALL = "C.UTF-8"
$env:PYTHONHASHSEED = "0"
$env:PYTHONDONTWRITEBYTECODE = "1"

& $python (Join-Path $ScriptDir "run_tests_parallel.py") @PytestArgs
exit $LASTEXITCODE
