#requires -Version 7.0
# Native Windows entry point for the same hermetic per-file runner as run_tests.sh.
# MSYS fork failures must not require changing Windows exploit-protection settings.
$ErrorActionPreference = 'Stop'
$testRepoRoot = Split-Path -Parent $PSScriptRoot
$testUserHome = [Environment]::GetEnvironmentVariable('HOME')
if (-not $testUserHome) { $testUserHome = [Environment]::GetFolderPath('UserProfile') }
$testCandidates = @(
    (Join-Path $testRepoRoot '.venv/Scripts/python.exe'),
    (Join-Path $testRepoRoot 'venv/Scripts/python.exe'),
    (Join-Path $testUserHome '.hermes/hermes-agent/venv/Scripts/python.exe'),
    [Environment]::GetEnvironmentVariable('HERMES_PYTHON')
)

function New-TestProcessInfo([string]$Executable, [string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.WorkingDirectory = $testRepoRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.Environment.Clear()
    # Keep this allowlist in parity with run_tests.sh; never forward credentials.
    foreach ($name in @(
        'PATH', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'LOCALAPPDATA', 'APPDATA',
        'SYSTEMROOT', 'TEMP', 'TMP', 'HERMES_TEST_IMAGE', 'HERMES_TEST_WORKERS',
        'HERMES_TEST_PATHS', 'HERMES_TEST_FILE_TIMEOUT', 'HERMES_TEST_FILE_RETRIES',
        'HERMES_TEST_SLICE', 'HERMES_GATEWAY_LOCK_DIR', 'HERMES_RUN_SLOW_PET_TESTS',
        'HERMES_E2E_BROWSER'
    )) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { $info.Environment[$name] = $value }
    }
    $info.Environment['HOME'] = $testUserHome
    $info.Environment['TZ'] = 'UTC'
    $info.Environment['LANG'] = 'C.UTF-8'
    $info.Environment['LC_ALL'] = 'C.UTF-8'
    $info.Environment['PYTHONHASHSEED'] = '0'
    $info.Environment['PYTHONUTF8'] = '1'
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    return $info
}

$testPython = $null
foreach ($candidate in $testCandidates) {
    if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
    $probeInfo = New-TestProcessInfo $candidate @('-c', 'import pytest')
    $probeInfo.RedirectStandardError = $true
    $probeInfo.RedirectStandardOutput = $true
    $probe = [Diagnostics.Process]::Start($probeInfo)
    $probe.StandardOutput.ReadToEnd() | Out-Null
    $probe.StandardError.ReadToEnd() | Out-Null
    $probe.WaitForExit()
    $probeExit = $probe.ExitCode
    $probe.Dispose()
    if ($probeExit -eq 0) { $testPython = $candidate; break }
}
if (-not $testPython) {
    [Console]::Error.WriteLine('No virtualenv with pytest found; install dev extras or set HERMES_PYTHON.')
    exit 1
}

$runnerArguments = @('-u', (Join-Path $PSScriptRoot 'run_tests_parallel.py')) + $args
$runnerInfo = New-TestProcessInfo $testPython $runnerArguments
$runnerInfo.RedirectStandardOutput = $true
$runnerInfo.RedirectStandardError = $true
$testGuard = Join-Path $testUserHome '.hermes/pytest_live_guard.py'
if (Test-Path -LiteralPath $testGuard -PathType Leaf) {
    $runnerInfo.Environment['PYTHONPATH'] = Split-Path -Parent $testGuard
    $runnerInfo.Environment['PYTEST_PLUGINS'] = 'pytest_live_guard'
}
Write-Host 'Running per-file test suite (native Windows; clean environment, UTC, deterministic hash seed).'
$runner = [Diagnostics.Process]::Start($runnerInfo)
$outputTask = $runner.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
$errorTask = $runner.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
$runner.WaitForExit()
$runnerExit = $runner.ExitCode
[Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($outputTask, $errorTask))
$runner.Dispose()
exit $runnerExit
