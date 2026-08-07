# Unit tests for install.ps1's Playwright command resolution.
# The installer itself is never executed: the helper is extracted through the
# PowerShell AST so tests do not download packages or browsers.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts/install.ps1"

$failures = 0
function Assert-Equal {
    param($Expected, $Actual, [string]$Label)
    if ($Expected -ne $Actual) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}
function Assert-True {
    param($Condition, [string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installScript, [ref]$tokens, [ref]$parseErrors
)
if ($parseErrors.Count -gt 0) {
    throw "install.ps1 has parse errors: $($parseErrors -join '; ')"
}

$fnAst = $ast.FindAll(
    {
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Resolve-PlaywrightInvocation"
    }, $true
) | Select-Object -First 1
if (-not $fnAst) {
    throw "Resolve-PlaywrightInvocation not found in install.ps1"
}
. ([scriptblock]::Create($fnAst.Extent.Text))

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-pw-test-" + [guid]::NewGuid())
try {
    $rootBin = Join-Path $tempRoot "node_modules/.bin"
    $workspaceBin = Join-Path $tempRoot "apps/desktop/node_modules/.bin"
    New-Item -ItemType Directory -Force -Path $rootBin | Out-Null
    New-Item -ItemType Directory -Force -Path $workspaceBin | Out-Null
    $rootPlaywright = Join-Path $rootBin "playwright.cmd"
    $localPlaywright = Join-Path $workspaceBin "playwright.cmd"
    Set-Content -Path $rootPlaywright -Value "@echo off" -Encoding Ascii
    Set-Content -Path $localPlaywright -Value "@echo off" -Encoding Ascii

    $hoisted = Resolve-PlaywrightInvocation -InstallDir $tempRoot -NpxExe "C:\Node\npx.cmd"
    Assert-Equal $rootPlaywright $hoisted.Command "prefers current root-hoisted workspace binary"
    Assert-Equal "install|chromium" ($hoisted.Arguments -join "|") "hoisted binary receives install args"

    Remove-Item -Force $rootPlaywright
    $local = Resolve-PlaywrightInvocation -InstallDir $tempRoot -NpxExe "C:\Node\npx.cmd"
    Assert-Equal $localPlaywright $local.Command "falls back to nested Desktop workspace binary"
    Assert-Equal "install|chromium" ($local.Arguments -join "|") "local binary receives install args"

    Remove-Item -Force $localPlaywright
    $fallback = Resolve-PlaywrightInvocation -InstallDir $tempRoot -NpxExe "C:\Node\npx.cmd"
    Assert-Equal "C:\Node\npx.cmd" $fallback.Command "falls back to resolved npx"
    Assert-Equal "--yes|--package=playwright|playwright|install|chromium" `
        ($fallback.Arguments -join "|") `
        "fallback explicitly injects the Playwright package bin"

    $missing = Resolve-PlaywrightInvocation -InstallDir $tempRoot -NpxExe $null
    Assert-Equal $null $missing "returns null when neither local binary nor npx exists"

    $source = Get-Content $installScript -Raw
    Assert-True ($source.Contains("Resolve-PlaywrightInvocation -InstallDir `$InstallDir -NpxExe `$npxExe")) `
        "Install-NodeDeps uses the shared resolver"
} finally {
    Remove-Item -Recurse -Force $tempRoot -ErrorAction SilentlyContinue
}

if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All Playwright resolution tests passed." -ForegroundColor Green
exit 0
