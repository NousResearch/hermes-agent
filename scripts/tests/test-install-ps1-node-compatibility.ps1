# Behavioral tests for install.ps1 system Node/npm compatibility selection.
#
# The installer is dot-sourced without running its entry point, then external
# commands and downloads are replaced with deterministic in-process stubs.
# This exercises the shipped range parser and Test-Node acceptance gate without
# changing PATH, installing software, or touching the user's Hermes home.

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path $env:TEMP ("hermes-node-compatibility-test-" + [Guid]::NewGuid().ToString('N'))
$HermesHome = Join-Path $testRoot 'home'
$InstallDir = Join-Path $testRoot 'missing-checkout'
. $installScript -HermesHome $HermesHome -InstallDir $InstallDir

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
    if ($Expected -ceq $Actual) {
        Write-Host "PASS: $Label"
    } else {
        Write-Host "FAIL: $Label"
        Write-Host "  expected: [$Expected]"
        Write-Host "  actual:   [$Actual]"
        $script:Failures++
    }
}

Write-Host '-- npm range evaluation --'
$supportedRange = Get-NpmRange
Assert-Equal '<11.10.0 || >=11.17.0' $supportedRange 'fresh-install fallback matches the supported npm range'
Assert-Equal $true (Test-NpmVersionOk '10.9.8') 'bundled npm 10.9.8 is accepted before clone'
Assert-Equal $true (Test-NpmVersionOk '11.9.9') 'lower alternative is accepted'
Assert-Equal $false (Test-NpmVersionOk '11.10.0') 'excluded band starts at 11.10.0'
Assert-Equal $false (Test-NpmVersionOk '11.16.0') 'reported npm 11.16.0 is rejected'
Assert-Equal $true (Test-NpmVersionOk '11.17.0') 'upper alternative starts at 11.17.0'
Assert-Equal $false (Test-NpmVersionOk 'not-a-version') 'malformed version fails closed'
Assert-Equal $false (Test-NpmVersionOk '12.0.0' '^12.0.0') 'unsupported range syntax fails closed'

# Controlled command surface used by the real Test-Node function.
$script:FakeNpmAvailable = $true
$script:FakeNpmVersion = '11.16.0'
$script:FakeNodeVersion = 'v24.18.0'
$script:DownloadAttempts = 0
$script:HasNode = $null
$NodeVersion = '22'

function node { $script:FakeNodeVersion }
function npm.cmd { $script:FakeNpmVersion }
function Get-Command {
    [CmdletBinding()]
    param([string]$Name)

    switch ($Name) {
        'node' {
            return Microsoft.PowerShell.Core\Get-Command node -CommandType Function
        }
        'npm.cmd' {
            if ($script:FakeNpmAvailable) {
                return Microsoft.PowerShell.Core\Get-Command npm.cmd -CommandType Function
            }
            return $null
        }
        'npm' { return $null }
        'winget' { return $null }
        default { return $null }
    }
}
function Ensure-NodeExeOnPath { $true }
function Get-WindowsArch { 'x64' }
function Invoke-WebRequest {
    $script:DownloadAttempts++
    throw 'network disabled by test'
}
function Write-Info { param([string]$Message) }
function Write-Warn { param([string]$Message) }
function Write-Success { param([string]$Message) }

function Invoke-SystemNodeProbe {
    param(
        [string]$NodeVersion,
        [string]$NpmVersion,
        [bool]$NpmAvailable = $true
    )

    $script:FakeNodeVersion = $NodeVersion
    $script:FakeNpmVersion = $NpmVersion
    $script:FakeNpmAvailable = $NpmAvailable
    $script:DownloadAttempts = 0
    $script:HasNode = $null
    [void](Test-Node)
    return [pscustomobject]@{
        HasNode = $script:HasNode
        DownloadAttempts = $script:DownloadAttempts
    }
}

Write-Host ''
Write-Host '-- system Node acceptance --'
$result = Invoke-SystemNodeProbe 'v24.18.0' '11.17.0'
Assert-Equal $true $result.HasNode 'compatible system Node/npm is accepted'
Assert-Equal 0 $result.DownloadAttempts 'compatible system npm avoids managed download'

$result = Invoke-SystemNodeProbe 'v22.22.0' '10.9.8'
Assert-Equal $true $result.HasNode 'minimum Node with bundled npm is accepted'
Assert-Equal 0 $result.DownloadAttempts 'bundled npm avoids managed download'

$result = Invoke-SystemNodeProbe 'v24.18.0' '11.16.0'
Assert-Equal $false $result.HasNode 'incompatible system npm is not accepted'
Assert-Equal 1 $result.DownloadAttempts 'incompatible system npm falls through to managed Node'

$result = Invoke-SystemNodeProbe 'v24.18.0' '' $false
Assert-Equal $false $result.HasNode 'missing system npm is not accepted'
Assert-Equal 1 $result.DownloadAttempts 'missing system npm falls through to managed Node'

Write-Host ''
Write-Host '-- managed npm reuse --'
$managedDir = Join-Path $testRoot 'managed-node'
New-Item -ItemType Directory -Force -Path $managedDir | Out-Null
$managedNpm = Join-Path $managedDir 'npm.cmd'
@'
@echo off
if "%~1"=="--version" (
  echo 10.9.8
  exit /b 0
)
exit /b 42
'@ | Set-Content -LiteralPath $managedNpm -Encoding Ascii
Assert-Equal $true (Update-ManagedNpm $managedDir) 'compatible managed npm skips the upgrade command'

if ($script:Failures -gt 0) {
    Write-Host ''
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ''
Write-Host 'all assertions passed'

if (Test-Path $testRoot) {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
