# PM owns Node provisioning. Verify the pre-Python bootstrap handoff.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts\install.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("hermes-pm-delegation-" + [Guid]::NewGuid().ToString('N'))
$testHome = Join-Path $testRoot 'home'
$checkout = Join-Path $testRoot 'checkout'
$script:Failures = 0
function Assert-True($Condition, [string]$Label) {
    if ($Condition) { Write-Host "PASS: $Label" }
    else { Write-Host "FAIL: $Label"; $script:Failures++ }
}
# Tripwires exist before dot-sourcing, so a broken guard cannot run an install.
function Invoke-WebRequest { throw 'unexpected download' }
function Invoke-RestMethod { throw 'unexpected download' }
function git { throw 'unexpected git command' }
function uv { throw 'unexpected uv command' }
function node { throw 'unexpected node command' }
function npm { throw 'unexpected npm command' }

try {
    . $installScript -HermesHome $testHome -InstallDir $checkout
    Assert-True (-not (Test-Path $testRoot)) 'dot-source loads definitions without filesystem writes'

    $fakeUv = Join-Path $testRoot 'uv.cmd'
    $fakePython = Join-Path $testRoot 'python.cmd'
    $argsFile = Join-Path $testRoot 'uv-args.txt'
    $pythonArgsFile = Join-Path $testRoot 'python-args.txt'
    New-Item -ItemType Directory -Force -Path (Join-Path $checkout 'pm') | Out-Null
    @"
@echo off
echo %* >> "$argsFile"
if "%~2"=="find" echo $fakePython
exit /b 0
"@ | Set-Content -LiteralPath $fakeUv -Encoding Ascii
    @"
@echo off
echo %* > "$pythonArgsFile"
exit /b 0
"@ | Set-Content -LiteralPath $fakePython -Encoding Ascii
    function Get-Uv { return $fakeUv }

    $failed = $false
    try { Invoke-BootstrapPm } catch { $failed = $true }
    Assert-True $failed 'missing lockfile refuses delegation'
    Assert-True (-not (Test-Path $argsFile)) 'missing lockfile never invokes uv'

    '{"packages":{"python":{"version":"3.13.2+test"}}}' |
        Set-Content -LiteralPath (Join-Path $checkout 'pm\lock.json') -Encoding UTF8
    Invoke-BootstrapPm
    # cmd's `echo %* >> file` keeps the space before `>>` in the recorded line.
    $recorded = @(Get-Content -LiteralPath $argsFile | ForEach-Object { $_.Trim() })
    Assert-True ($recorded.Count -eq 1) 'uv locates the available bootstrap Python without reinstalling it'
    $pyArch = if ((Get-WindowsArch) -eq 'arm64') { 'aarch64' } else { 'x86_64' }
    Assert-True ($recorded[0] -eq "python find --managed-python --no-project cpython-3.13-windows-$pyArch-none") 'Python minor comes from the lockfile, pinned to the machine architecture; lookup ignores ambient project discovery'
    Assert-True ((Get-Content -LiteralPath $pythonArgsFile -Raw).Trim() -eq '-m pm.cli install') 'Python launches PM without a uv parent'

    # The installer owns no node stage: tool and frontend provisioning belongs
    # to pm, driven by the shared completion tail (install.ps1 "products").
} finally {
    if (Test-Path $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
if ($script:Failures) { exit 1 }
Write-Host 'all assertions passed'
