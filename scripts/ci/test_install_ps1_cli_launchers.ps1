# Behavioral test for install.ps1's hermes launcher staging.
#
# Run: powershell.exe -NoProfile -File scripts/ci/test_install_ps1_cli_launchers.ps1
#
# The test lifts the real Stage-Path function from the PowerShell AST and
# executes it against a temporary install tree. It never reads or changes
# the user's PATH (the registry I/O is stubbed to a capture variable).
#
# Under pm (no-boot-through-venv) Stage-Path must stage launchers that boot
# the pm STORE python with PYTHONPATH=repo;venv-site-packages — never the
# venv interpreter (pyvenv.cfg is inert dead config). The real minting
# machinery (hermes_cli/_launchers.py) runs via the venv python, exactly as
# the installer does.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$installPs1 = Join-Path $repoRoot 'scripts\install.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$null, [ref]$null)

$fn = $ast.Find({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Stage-Path'
}, $true)

if (-not $fn) {
    throw "Stage-Path not found in $installPs1"
}

# Stub the installer's process-level helpers before expanding the function:
# Fail must throw (not exit the test process) and the HKCU Path write must
# land in a capture variable, never in the user's registry.
$script:userPathWrite = $null
$script:fakeUserPath = ''
$body = $fn.Extent.Text
$body = $body.Replace(
    '[Environment]::GetEnvironmentVariable("Path", "User")',
    '$script:fakeUserPath')
$body = $body.Replace(
    '[Environment]::SetEnvironmentVariable("Path", "$binDir;$userPath", "User")',
    '$script:userPathWrite = "$binDir;$userPath"')
Invoke-Expression $body

function Fail([string]$msg) { throw "stage failed: $msg" }
function Log([string]$msg) { Write-Host "[hermes] $msg" }

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

function Get-LauncherContent {
    param([string]$Path)
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    return [System.Text.Encoding]::ASCII.GetString($bytes)
}

# A real python for the launcher-minting step (the installer runs it via
# the venv python). Prefer the repo checkout's venv interpreter — a
# standalone CPython; a bare `python` on PATH may be a store/managed
# python that cannot run when copied outside its own layout.
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
}
if (-not $python) { $python = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $python) {
    Write-Host "SKIP: no python on PATH — launcher minting cannot be exercised"
    exit 0
}

try {
    $installRoot = Join-Path $caseRoot 'hermes-agent'
    # Stage-Path derives its destination from $HermesHome\bin — point
    # $HermesHome at the case root so the asserts below read the same dir.
    $HermesHome = $caseRoot
    $binDir = Join-Path $caseRoot 'bin'
    $InstallDir = $installRoot

    # Case 1: no venv yet — the stage must fail loudly, staging nothing.
    $missingThrew = $false
    try {
        Stage-Path | Out-Null
    } catch {
        $missingThrew = $_.Exception.Message -like '*venv python missing*'
    }
    Assert-True $missingThrew 'missing venv python fails the launcher stage'

    # Build a fake install: venv + the REAL launchers module (copied from
    # the repo, importable from the install root cwd like the installer's
    # Push-Location does) + a fake pm store with a materialized python.
    $venvScripts = Join-Path $installRoot 'venv\Scripts'
    New-Item -ItemType Directory -Force -Path $venvScripts | Out-Null
    Copy-Item $python (Join-Path $venvScripts 'python.exe')
    New-Item -ItemType Directory -Force -Path (Join-Path $installRoot 'hermes_cli') | Out-Null
    Copy-Item (Join-Path $repoRoot 'hermes_cli\_launchers.py') (Join-Path $installRoot 'hermes_cli\_launchers.py')
    Copy-Item (Join-Path $repoRoot 'hermes_constants.py') (Join-Path $installRoot 'hermes_constants.py')

    $store = Join-Path $caseRoot 'store'
    $entryName = 'python-3.11.15+x20260807-win32-arm64'
    New-Item -ItemType Directory -Force -Path (Join-Path $store $entryName) | Out-Null
    Copy-Item $python (Join-Path $store "$entryName\python.exe")
    ('{"schema": 1, "packages": {"python": {"entry": "' + $entryName + '"}}}') |
        Set-Content -Path (Join-Path $store 'facts.json') -Encoding Ascii
    $env:HERMES_RUNTIME_DIR = $store

    Stage-Path | Out-Null

    foreach ($name in @('hermes', 'hermes-acp')) {
        $exe = Join-Path $binDir "$name.exe"
        $cmd = Join-Path $binDir "$name.cmd"
        if (Test-Path -LiteralPath $exe) {
            $content = Get-LauncherContent $exe
        } elseif (Test-Path -LiteralPath $cmd) {
            $content = Get-LauncherContent $cmd
        } else {
            $content = ''
        }
        Assert-True ($content -ne '') "$name launcher staged"
        $venvPython = (Join-Path $venvScripts 'python.exe')
        Assert-True (-not $content.Contains($venvPython)) `
            "$name launcher does not reference the venv python"
        Assert-True $content.Contains((Join-Path $store $entryName)) `
            "$name launcher binds to the pm store interpreter"
    }

    Assert-True ($null -ne $script:userPathWrite) 'user PATH write captured the bin dir'
    Assert-True ($script:userPathWrite -eq "$binDir;") 'user PATH capture has the right shape'

    # The minted launcher must actually boot the STORE interpreter: give
    # the fake repo a hermes_cli.main that prints what it runs under.
    Set-Content -Path (Join-Path $installRoot 'hermes_cli\__init__.py') `
        -Encoding Ascii -Value ''
    Set-Content -Path (Join-Path $installRoot 'hermes_cli\main.py') `
        -Encoding Ascii -Value "import sys`ndef main():`n    print('PROBE', sys.executable)`n    return 0`n"
    $out = (& (Join-Path $binDir 'hermes.exe') 2>&1) -join ' '
    Write-Host "  BOOT-OUT: $out"
    # The exe must boot a python that imports hermes_cli through the
    # composed PYTHONPATH (repo first). (This fixture's fake store python
    # is a copied venv interpreter whose sys.executable reports its base —
    # a real pm store python (python-build-standalone) reports itself;
    # binding to the store interpreter is asserted above from the exe
    # contents.)
    Assert-True ($out.Contains('PROBE')) `
        'minted hermes.exe boots a python that imports via the composed PYTHONPATH'
} finally {
    Remove-Item Env:HERMES_RUNTIME_DIR -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $caseRoot) {
        $resolvedCase = [System.IO.Path]::GetFullPath($caseRoot)
        if (-not $resolvedCase.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove test directory outside the system temp directory: $resolvedCase"
        }
        # The booted launcher's child may outlive the assertion briefly.
        Start-Sleep -Milliseconds 800
        try {
            Remove-Item -LiteralPath $resolvedCase -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Host "  (leftover temp dir could not be removed: $resolvedCase)"
        }
    }
}

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
