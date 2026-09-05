# Behavioral regressions for managed Git PATH repair and checkout preservation.
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
$legacy = $entries -join ''
$cases = @(
    @{ Name = 'empty PATH'; Before = ''; After = $joined },
    @{ Name = 'single entry'; Before = 'C:\Other'; After = "C:\Other;$joined" },
    @{ Name = 'multiple entries'; Before = 'C:\One;C:\Two'; After = "C:\One;C:\Two;$joined" },
    @{ Name = 'empty segments preserved'; Before = 'C:\Other;;'; After = "C:\Other;;;$joined" },
    @{ Name = 'existing Git entries'; Before = $joined; After = $joined },
    @{ Name = 'single existing Git entry'; Before = $entries[0]; After = $joined },
    @{ Name = 'case insensitive membership'; Before = $joined.ToUpperInvariant(); After = $joined.ToUpperInvariant() },
    @{ Name = 'legacy empty-PATH corruption'; Before = $legacy; After = $joined },
    @{ Name = 'legacy single-entry corruption'; Before = "C:\Other$legacy"; After = "C:\Other;$joined" },
    @{ Name = 'legacy corruption after another stage'; Before = "C:\Example User\hermes\node;$legacy"; After = "C:\Example User\hermes\node;$joined" },
    @{ Name = 'repeated legacy retry'; Before = "C:\Other$legacy$legacy"; After = "C:\Other;$joined" },
    @{ Name = 'legacy case variation'; Before = $legacy.ToUpperInvariant(); After = $joined },
    @{ Name = 'unrelated concatenation untouched'; Before = 'C:\OtherD:\Tools'; After = "C:\OtherD:\Tools;$joined" },
    @{ Name = 'other Git root untouched'; Before = 'D:\Git\cmdD:\Git\binD:\Git\usr\bin'; After = "D:\Git\cmdD:\Git\binD:\Git\usr\bin;$joined" },
    @{ Name = 'partial signature untouched'; Before = "$gitDir\cmd$gitDir\bin"; After = "$gitDir\cmd$gitDir\bin;$joined" }
)
foreach ($case in $cases) {
    $actual = Get-ManagedGitUserPath -UserPath $case.Before -GitDir $gitDir
    Assert-Equal $case.After $actual $case.Name
    Assert-Equal $actual (Get-ManagedGitUserPath -UserPath $actual -GitDir $gitDir) "$($case.Name): idempotent"
}
Assert-Equal $joined (Get-ManagedGitUserPath -UserPath $null -GitDir $gitDir) 'unset PATH'

# Replace the registry-writing setter with an in-memory handoff probe.
function Set-ManagedGitPath {
    param([string]$GitDir)
    $script:FakeUserPath = Get-ManagedGitUserPath -UserPath $script:FakeUserPath -GitDir $GitDir
    $script:GitAvailable = $true
    $script:PathRepairs++
}
function Write-Info { param([string]$Message) }
function Write-Warn { param([string]$Message) }
function Write-Success { param([string]$Message) }
function Set-GitBashEnvVar { $script:GitBashPath = 'synthetic-bash' }
function Test-GitBashCompatibility { param([string]$BashPath) $true }
function Invoke-WebRequest { throw 'unexpected download in Git recovery test' }

$script:GitAvailable = $false
$script:GitMode = 'ok'
$script:PathRepairs = 0
$script:FakeUserPath = ''
function Get-Command {
    [CmdletBinding()]
    param([string]$Name)
    if ($Name -eq 'git') {
        if ($script:GitAvailable) {
            return Microsoft.PowerShell.Core\Get-Command git -CommandType Function
        }
        return $null
    }
    Microsoft.PowerShell.Core\Get-Command $Name
}
function git {
    if (-not $script:GitAvailable -or $script:GitMode -eq 'throw') {
        throw [System.Management.Automation.CommandNotFoundException]::new('synthetic Git unavailable')
    }
    if ($script:GitMode -eq 'nonzero') { $global:LASTEXITCODE = 1; return }
    if ($args.Count -eq 1 -and $args[0] -eq '--version') {
        $global:LASTEXITCODE = 0
        return 'git version test'
    }
    throw 'repository probe reached after successful Git preflight'
}

New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
try {
    # Model a completed Git download with a stale next-stage PATH. The file is
    # only a discovery fixture; the git function above is the executable seam.
    $managedCmd = Join-Path $HermesHome 'git\cmd'
    New-Item -ItemType Directory -Path $managedCmd -Force | Out-Null
    New-Item -ItemType File -Path (Join-Path $managedCmd 'git.exe') | Out-Null
    $managedDir = Join-Path $HermesHome 'git'
    $script:FakeUserPath = "$managedDir\cmd$managedDir\bin$managedDir\usr\bin"
    Assert-Equal $true (Install-Git) 'existing managed Git recovered without a download'
    Assert-Equal 1 $script:PathRepairs 'Git stage repairs existing install'
    Assert-Equal "$managedDir\cmd;$managedDir\bin;$managedDir\usr\bin" $script:FakeUserPath 'Git stage repairs malformed persisted entries'

    # A new repository-stage process must rediscover managed Git too. Abort at
    # the first post-preflight Git call so this never clones or accesses network.
    $script:GitAvailable = $false
    $script:PathRepairs = 0
    $errorText = ''
    try { Install-Repository } catch { $errorText = $_.Exception.Message }
    Assert-Equal 1 $script:PathRepairs 'repository stage repairs managed Git visibility'
    Assert-Equal $true ($errorText -like '*repository probe reached*') 'repository stage reaches Git only after successful preflight'

    # Missing or unlaunchable Git must leave both .git checkouts and ordinary
    # existing directories untouched, rather than moving them to .broken-*.
    Remove-Item -LiteralPath (Join-Path $managedCmd 'git.exe')
    foreach ($mode in @('missing', 'throw', 'nonzero')) {
        foreach ($withDotGit in @($false, $true)) {
            $InstallDir = Join-Path $testRoot ("checkout-$mode-$withDotGit")
            New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
            $sentinel = Join-Path $InstallDir 'keep.txt'
            Set-Content -LiteralPath $sentinel -Value 'preserve me'
            if ($withDotGit) { New-Item -ItemType Directory -Path (Join-Path $InstallDir '.git') | Out-Null }
            $script:GitAvailable = ($mode -ne 'missing')
            $script:GitMode = $mode
            $errorText = ''
            try { Install-Repository } catch { $errorText = $_.Exception.Message }
            Assert-Equal $true ($errorText -like '*checkout has not been moved*') "$mode/${withDotGit}: actionable prerequisite error"
            Assert-Equal $true (Test-Path -LiteralPath $sentinel) "$mode/${withDotGit}: existing files preserved"
            $backups = @(Get-ChildItem -LiteralPath $testRoot -Directory | Where-Object { $_.Name -like '*.broken-*' })
            Assert-Equal 0 $backups.Count "$mode/${withDotGit}: no false broken-checkout backup"
        }
    }
} finally {
    # This is the unique fixture directory created above, never a user home.
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}

if ($script:Failures -gt 0) { throw "$script:Failures Git PATH regression assertion(s) failed" }
Write-Host 'All Git PATH regression tests passed.'
