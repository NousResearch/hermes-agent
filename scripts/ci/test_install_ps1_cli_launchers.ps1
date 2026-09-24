# Behavioral test for install.ps1's hermes launcher staging (PR #92092,
# reworked for the managed-binary-dir layout).
#
# Run: powershell.exe -NoProfile -File scripts/ci/test_install_ps1_cli_launchers.ps1
#
# The test lifts the real Install-HermesCommandLaunchers function from the
# PowerShell AST and executes it against a temporary install tree. It never
# reads or changes the user's PATH. The staging destination is passed in by
# the caller (Set-PathVariable passes $HermesHome\bin -- the managed binary
# dir OUTSIDE the git checkout); here it is a sibling temp dir, which also
# proves the function stages wherever it is pointed rather than assuming
# the legacy in-checkout location.

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
    $installRoot = Join-Path $caseRoot 'hermes-agent'
    $binDir = Join-Path $caseRoot 'bin'
    New-Item -ItemType Directory -Force -Path $installRoot | Out-Null

    # Fail-before-PATH-mutation: a missing required source must throw and
    # must not leave an empty destination for the caller to put on PATH.
    $missingThrew = $false
    try {
        Install-HermesCommandLaunchers -Root $installRoot -Destination $binDir | Out-Null
    } catch {
        $missingThrew = $_.Exception.Message -like '*required launcher not found*'
    }
    Assert-True $missingThrew 'missing hermes.exe fails the launcher stage'
    Assert-True (-not (Test-Path -LiteralPath $binDir)) `
        'failure does not create an empty PATH directory'

    $scriptsDir = Join-Path $installRoot 'venv\Scripts'
    New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null
    $hermesV1 = [byte[]](77, 90, 1)
    $hermesV2 = [byte[]](77, 90, 2)
    $acp = [byte[]](77, 90, 3)
    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes.exe'), $hermesV1)
    Set-Content -Path (Join-Path $installRoot 'venv\pyvenv.cfg') `
        -Value "home = X" -Encoding Ascii

    $staged = Install-HermesCommandLaunchers -Root $installRoot -Destination $binDir
    Assert-True ($staged -eq $binDir) 'returns the destination it staged into'
    # PATH launchers are always .cmd text delegators -- never byte-copies of
    # uv's unsigned trampoline (Defender Pomal!rfn quarantine). No .exe copy
    # lands in the destination for any venv kind.
    Assert-True (Test-Path -LiteralPath (Join-Path $binDir 'hermes.cmd')) `
        'normal venv: .cmd delegator staged'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $binDir 'hermes.exe'))) `
        'normal venv: no exe copy staged'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $binDir 'hermes-acp.exe'))) `
        'optional ACP launcher may be absent'
    $cmdBody = [System.IO.File]::ReadAllText((Join-Path $binDir 'hermes.cmd'))
    Assert-True ($cmdBody.Contains((Join-Path $scriptsDir 'hermes.exe')) -and $cmdBody.Contains('%*')) `
        'delegator invokes the in-venv exe and forwards args'

    $expectedBody = "@echo off`r`n`"$(Join-Path $scriptsDir 'hermes.exe')`" %*`r`n"
    Assert-True ($cmdBody -eq $expectedBody) `
        'delegator body matches expected delegator text with trailing CRLF'

    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes.exe'), $hermesV2)
    [System.IO.File]::WriteAllBytes((Join-Path $scriptsDir 'hermes-acp.exe'), $acp)
    Install-HermesCommandLaunchers -Root $installRoot -Destination $binDir | Out-Null
    $refreshed = [System.IO.File]::ReadAllText((Join-Path $binDir 'hermes.cmd'))
    Assert-True ($refreshed.Contains((Join-Path $scriptsDir 'hermes.exe'))) `
        'installer refresh keeps delegating to the in-venv exe'
    Assert-True (Test-Path -LiteralPath (Join-Path $binDir 'hermes-acp.cmd')) `
        'installer stages the optional ACP delegator when present'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $binDir 'hermes-acp.exe'))) `
        'installer stages no ACP exe copy'

    # Stale exe copies are removed when the delegator is already in place.
    [System.IO.File]::WriteAllBytes((Join-Path $binDir 'hermes.exe'), $hermesV1)
    Assert-True (Test-Path -LiteralPath (Join-Path $binDir 'hermes.cmd')) `
        'delegator is already present before stale exe cleanup'
    Install-HermesCommandLaunchers -Root $installRoot -Destination $binDir | Out-Null
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $binDir 'hermes.exe'))) `
        'stale exe copy removed even when delegator already exists'
    Assert-True (Test-Path -LiteralPath (Join-Path $binDir 'hermes.cmd')) `
        'delegator preserved after stale exe cleanup'
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
