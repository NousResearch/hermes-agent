# Behavioral tests for install.ps1 system Node/npm compatibility selection.
#
# The installer itself is not executed. The real shipped functions are lifted
# through the PowerShell AST, then external commands and downloads are replaced
# with deterministic in-process stubs. This exercises the actual range parser
# and Test-Node acceptance gate without changing PATH, installing software, or
# touching the user's Hermes home.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installScript, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "install.ps1 has parse errors: $($parseErrors -join '; ')"
}

foreach ($name in @(
    'ConvertTo-NpmVersion',
    'Test-NpmVersionOk',
    'Test-NodeVersionOk',
    'Test-SystemNodeReady',
    'Test-Node'
)) {
    $fn = $ast.FindAll(
        {
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq $name
        },
        $true
    ) | Select-Object -First 1
    if (-not $fn) { throw "$name not found in install.ps1" }
    . ([scriptblock]::Create($fn.Extent.Text))
}

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
$supportedRange = '<11.10.0 || >=11.17.0'
Assert-Equal $true (Test-NpmVersionOk '11.9.9' $supportedRange) 'lower alternative is accepted'
Assert-Equal $false (Test-NpmVersionOk '11.10.0' $supportedRange) 'excluded band starts at 11.10.0'
Assert-Equal $false (Test-NpmVersionOk '11.16.0' $supportedRange) 'reported npm 11.16.0 is rejected'
Assert-Equal $true (Test-NpmVersionOk '11.17.0' $supportedRange) 'upper alternative starts at 11.17.0'
Assert-Equal $false (Test-NpmVersionOk '11.17.0' '>=12.0.0') 'pre-clone npm floor rejects 11.x'
Assert-Equal $true (Test-NpmVersionOk '12.0.0' '>=12.0.0') 'pre-clone npm floor accepts 12.0.0'
Assert-Equal $false (Test-NpmVersionOk 'not-a-version' $supportedRange) 'malformed version fails closed'
Assert-Equal $false (Test-NpmVersionOk '12.0.0' '^12.0.0') 'unsupported range syntax fails closed'

# Controlled command surface used by the lifted Test-Node function.
$script:FakeNpmAvailable = $true
$script:FakeNpmVersion = '11.16.0'
$script:FakeNpmRange = $supportedRange
$script:DownloadAttempts = 0
$script:HasNode = $null
$HermesHome = Join-Path $env:TEMP ("hermes-node-compatibility-test-" + [Guid]::NewGuid().ToString('N'))
$NodeVersion = '22'

function node { 'v24.18.0' }
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
function Get-NpmRange { $script:FakeNpmRange }
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
    param([string]$NpmVersion, [bool]$NpmAvailable = $true)

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
$result = Invoke-SystemNodeProbe '11.17.0'
Assert-Equal $true $result.HasNode 'compatible system Node/npm is accepted'
Assert-Equal 0 $result.DownloadAttempts 'compatible system npm avoids managed download'

$result = Invoke-SystemNodeProbe '11.16.0'
Assert-Equal $false $result.HasNode 'incompatible system npm is not accepted'
Assert-Equal 1 $result.DownloadAttempts 'incompatible system npm falls through to managed Node'

$result = Invoke-SystemNodeProbe '' $false
Assert-Equal $false $result.HasNode 'missing system npm is not accepted'
Assert-Equal 1 $result.DownloadAttempts 'missing system npm falls through to managed Node'

if ($script:Failures -gt 0) {
    Write-Host ''
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ''
Write-Host 'all assertions passed'
