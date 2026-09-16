# Behavioral regressions for managed Git PATH repair.
# Dot-source the real installer without running its entry point. No downloads,
# registry writes, or existing Hermes files are touched by these tests.

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$testRoot = Join-Path $env:TEMP ("hermes-git-path-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $testRoot 'checkout'
. (Join-Path $repoRoot 'scripts\install.ps1') -HermesHome $HermesHome -InstallDir $InstallDir

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:Failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
    if ($Expected -cne $Actual) {
        Write-Host "FAIL: $Label (expected [$Expected], got [$Actual])"
        $script:Failures++
    } else {
        Write-Host "PASS: $Label"
    }
}

$gitDir = 'C:\Example User\hermes\git'
$entries = @("$gitDir\cmd", "$gitDir\bin", "$gitDir\usr\bin")
$joined = $entries -join ';'
# The bug: a one-element User PATH unrolled into a string, so `+=` concatenated
# the three Git dirs onto it with no separators.
$legacy = $entries -join ''
$cases = @(
    @{ Name = 'single entry stays delimited'; Before = 'C:\Other'; After = "C:\Other;$joined" },
    @{ Name = 'legacy concatenated entry repaired'; Before = "C:\Other$legacy"; After = "C:\Other;$joined" }
)
foreach ($case in $cases) {
    $actual = Get-ManagedGitUserPath -UserPath $case.Before -GitDir $gitDir
    Assert-Equal $case.After $actual $case.Name
    Assert-Equal $actual (Get-ManagedGitUserPath -UserPath $actual -GitDir $gitDir) "$($case.Name): idempotent"
}

# A managed git.exe that exists but cannot launch (interrupted extraction, AV
# quarantine) must not be promoted ahead of a working system Git.
$brokenGitDir = Join-Path $testRoot 'broken-git'
$brokenGitCmd = Join-Path $brokenGitDir 'cmd'
New-Item -ItemType Directory -Path $brokenGitCmd -Force | Out-Null
Set-Content -LiteralPath (Join-Path $brokenGitCmd 'git.exe') -Value 'not an executable' -Encoding Ascii
$userPathBefore = [Environment]::GetEnvironmentVariable('Path', 'User')
$processPathBefore = $env:Path
try {
    Set-ManagedGitPath -GitDir $brokenGitDir
    Assert-Equal $processPathBefore $env:Path 'unlaunchable managed git leaves process PATH alone'
    Assert-Equal $userPathBefore ([Environment]::GetEnvironmentVariable('Path', 'User')) 'unlaunchable managed git leaves User PATH alone'
} finally {
    $env:Path = $processPathBefore
    if ($userPathBefore -cne [Environment]::GetEnvironmentVariable('Path', 'User')) {
        [Environment]::SetEnvironmentVariable('Path', $userPathBefore, 'User')
    }
    Remove-Item -Recurse -Force $testRoot -ErrorAction SilentlyContinue
}

if ($script:Failures -gt 0) { throw "$script:Failures Git PATH regression assertion(s) failed" }
Write-Host 'All Git PATH regression tests passed.'
