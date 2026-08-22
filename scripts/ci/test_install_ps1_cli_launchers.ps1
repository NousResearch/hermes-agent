# Behavioral test for install.ps1's dedicated Hermes launcher directory.
#
# Run: powershell.exe -NoProfile -File scripts/ci/test_install_ps1_cli_launchers.ps1
#
# The test lifts the real Install-HermesCommandLaunchers function from the
# PowerShell AST and executes it against a temporary install tree. It never
# reads or changes the user's PATH.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path (Join-Path $PSScriptRoot '..') 'install.ps1' | Resolve-Path
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$null, [ref]$null)

$fn = $ast.Find({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Install-HermesCommandLaunchers'
}, $true)

if (-not $fn) {
    throw "Install-HermesCommandLaunchers not found in $installPs1"
}

Invoke-Expression $fn.Extent.Text

$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$caseRoot = [System.IO.Path]::GetFullPath((Join-Path $tempBase (
    'hermes-cli-launcher-test-' + [guid]::NewGuid().ToString('N')
)))
if (-not $caseRoot.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to create test directory outside the system temp directory: $caseRoot"
}

$script:Failures = 0

function Assert-True {
    param([bool]$Condition, [string]$Name)
    if ($Condition) {
        Write-Host "  PASS  $Name"
    } else {
        Write-Host "  FAIL  $Name"
        $script:Failures++
    }
}

function Assert-BytesEqual {
    param([byte[]]$Expected, [byte[]]$Actual, [string]$Name)
    $same = $Expected.Length -eq $Actual.Length
    if ($same) {
        for ($i = 0; $i -lt $Expected.Length; $i++) {
            if ($Expected[$i] -ne $Actual[$i]) {
                $same = $false
                break
            }
        }
    }
    Assert-True $same $Name
}

try {
    New-Item -ItemType Directory -Force -Path $caseRoot | Out-Null

    $missingThrew = $false
    try {
        Install-HermesCommandLaunchers -Root $caseRoot | Out-Null
    } catch {
        $missingThrew = $_.Exception.Message -like '*required launcher not found*'
    }
    Assert-True $missingThrew 'missing hermes.exe fails the launcher stage'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $caseRoot 'bin'))) `
        'failure does not create an empty PATH directory'

    $scriptsDir = Join-Path $caseRoot 'venv\Scripts'
    New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null
    $hermesV1 = [byte[]](77, 90, 1)
    $hermesV2 = [byte[]](77, 90, 2)
    $acp = [byte[]](77, 90, 3)
    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes.exe'), $hermesV1)

    $binDir = Install-HermesCommandLaunchers -Root $caseRoot
    Assert-BytesEqual $hermesV1 `
        ([System.IO.File]::ReadAllBytes((Join-Path $binDir 'hermes.exe'))) `
        'required launcher is copied into the dedicated bin directory'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $binDir 'hermes-acp.exe'))) `
        'optional ACP launcher may be absent'

    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes.exe'), $hermesV2)
    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes-acp.exe'), $acp)
    Install-HermesCommandLaunchers -Root $caseRoot | Out-Null
    Assert-BytesEqual $hermesV2 `
        ([System.IO.File]::ReadAllBytes((Join-Path $binDir 'hermes.exe'))) `
        'installer refreshes an existing Hermes launcher'
    Assert-BytesEqual $acp `
        ([System.IO.File]::ReadAllBytes((Join-Path $binDir 'hermes-acp.exe'))) `
        'installer copies the optional ACP launcher when present'
} finally {
    if (Test-Path -LiteralPath $caseRoot) {
        $resolvedCase = [System.IO.Path]::GetFullPath($caseRoot)
        if (-not $resolvedCase.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove test directory outside the system temp directory: $resolvedCase"
        }
        Remove-Item -LiteralPath $resolvedCase -Recurse -Force
    }
}

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
