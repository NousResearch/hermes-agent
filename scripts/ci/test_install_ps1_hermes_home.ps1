# Behavioral regression test for install.ps1's HERMES_HOME environment policy.
#
# Run: pwsh -NoProfile -File scripts/ci/test_install_ps1_hermes_home.ps1
#
# This lifts the shipped Set-PathVariable body from its PowerShell AST and
# replaces only registry access with an in-memory store. It verifies an update
# removes an old persisted value while retaining the installer process value.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path $PSScriptRoot '..' 'install.ps1' | Resolve-Path
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$null, [ref]$null)
$fn = $ast.Find({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Set-PathVariable'
}, $true)

if (-not $fn) {
    throw "Set-PathVariable not found in $installPs1"
}

$definition = $fn.Extent.Text
$homeReads = ([regex]'\[Environment\]::GetEnvironmentVariable\("HERMES_HOME", "User"\)').Matches($definition).Count
$homeWrites = ([regex]'\[Environment\]::SetEnvironmentVariable\("HERMES_HOME", \$null, "User"\)').Matches($definition).Count
if ($homeReads -ne 1 -or $homeWrites -ne 1) {
    throw "expected one User HERMES_HOME read and one removal; found $homeReads read(s), $homeWrites removal(s). Update this harness."
}

$definition = $definition -replace `
    '\[Environment\]::GetEnvironmentVariable\("Path", "User"\)', '$script:FakeUserPath'
$definition = $definition -replace `
    '\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)', '$script:FakeUserPath = $1'
$definition = $definition -replace `
    '\[Environment\]::GetEnvironmentVariable\("HERMES_HOME", "User"\)', '$script:FakeHermesHome'
$definition = $definition -replace `
    '\[Environment\]::SetEnvironmentVariable\("HERMES_HOME", \$null, "User"\)', '$script:FakeHermesHome = $null; $script:HomeWrites++'

function Write-Info { param([string]$Message) }
function Write-Success { param([string]$Message) }
Invoke-Expression $definition

$script:FakeUserPath = 'C:\Windows'
$script:FakeHermesHome = 'C:\Users\local\AppData\Local\hermes'
$script:HomeWrites = 0
$NoVenv = $true
$InstallDir = 'C:\Users\local\AppData\Local\hermes\hermes-agent'
$HermesHome = 'C:\Users\local\AppData\Local\hermes'

Set-PathVariable

if ($script:FakeHermesHome -ne $null) {
    throw 'expected Set-PathVariable to remove persisted User HERMES_HOME'
}
if ($script:HomeWrites -ne 1) {
    throw "expected one persisted HERMES_HOME removal; got $script:HomeWrites"
}
if ($env:HERMES_HOME -ne $HermesHome) {
    throw "expected installer process HERMES_HOME to remain $HermesHome"
}

Write-Host 'all assertions passed'
