# Behavioral test for install.ps1's managed-uv isolation helpers.
#
# Run:  pwsh -NoProfile -File scripts/ci/test_install_ps1_uv_isolation.ps1
#
# Not wired into the default CI lane -- the Linux runners have no PowerShell
# host. It runs on any machine with pwsh, and on a Windows runner if one is
# ever added.
#
# Same AST-lift methodology as test_install_ps1_path_migration.ps1: parses
# install.ps1, lifts the real function bodies, and exercises the shipped
# logic -- private managed path and legacy bin\uv migration -- for real.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path $PSScriptRoot '..' 'install.ps1' | Resolve-Path

function Find-InstallFunction {
    param([string]$Name)
    $parsed = [System.Management.Automation.Language.Parser]::ParseFile(
        $installPs1, [ref]$null, [ref]$null)
    return $parsed.Find({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq $Name
    }, $true)
}

function Get-RewrittenDefinition {
    param([string]$Name, [int]$ExpectedReads = 0, [int]$ExpectedWrites = 0)
    $fn = Find-InstallFunction -Name $Name
    if (-not $fn) {
        throw "$Name not found in $installPs1"
    }
    $definition = $fn.Extent.Text
    $reads  = ([regex]'\[Environment\]::GetEnvironmentVariable\("Path", "User"\)').Matches($definition).Count
    $writes = ([regex]'\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)').Matches($definition).Count
    if ($reads -ne $ExpectedReads -or $writes -ne $ExpectedWrites) {
        throw "expected $ExpectedReads read(s) and $ExpectedWrites write(s) in ${Name}; found $reads read(s), $writes write(s). Update this harness."
    }
    return $definition
}

Invoke-Expression (Get-RewrittenDefinition -Name 'Get-ManagedUvPath')
Invoke-Expression (Get-RewrittenDefinition -Name 'Move-LegacyManagedUv')
Invoke-Expression (Get-RewrittenDefinition -Name 'Set-UvPythonIsolationEnv')

# install.ps1's write helpers (failure paths only here).
function Write-Info    { Write-Host "[info]  $args" }
function Write-Success { Write-Host "[ok]    $args" }
function Write-Warn    { Write-Host "[warn]  $args" }

# Point $HermesHome (script scope — the lifted functions read it) at a
# temp dir.
$script:HermesHome = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-uv-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $script:HermesHome | Out-Null

$script:Failures = 0

function Assert-Equal {
    param($Expected, $Actual, [string]$Name)
    if ($Expected -ceq $Actual) {
        Write-Host "  PASS  $Name"
    } else {
        Write-Host "  FAIL  $Name"
        Write-Host "        expected: [$Expected]"
        Write-Host "        actual:   [$Actual]"
        $script:Failures++
    }
}

function Assert-True {
    param([bool]$Actual, [string]$Name)
    if ($Actual) { Write-Host "  PASS  $Name" }
    else {
        Write-Host "  FAIL  $Name"
        $script:Failures++
    }
}

# ---------------------------------------------------------------------------
# Install-Uv must set the astral no-PATH-write switch.  UV_INSTALL_DIR alone
# still results in the managed dir being prepended to the user PATH on a
# fresh install: uv's installer honours NoModifyPath only via UV_UNMANAGED_INSTALL
# or UV_NO_MODIFY_PATH, and without either it writes HKCU\Environment\Path --
# exactly the shadowing the whole private layout exists to prevent.  Static
# check: Install-Uv performs a real network install and cannot be lifted and
# executed here.
# ---------------------------------------------------------------------------
$installUvFn = Find-InstallFunction -Name 'Install-Uv'
if (-not $installUvFn) {
    throw 'Install-Uv not found in install.ps1'
}
$installUvText = $installUvFn.Extent.Text
if ($installUvText -notmatch 'UV_UNMANAGED_INSTALL' -and
    $installUvText -notmatch 'UV_NO_MODIFY_PATH') {
    throw "Install-Uv must set UV_UNMANAGED_INSTALL (or UV_NO_MODIFY_PATH=1): " +
        "without it the astral installer prepends the managed dir to " +
        "HKCU\Environment\Path and shadows the user's uv on fresh installs."
}
Write-Host '  PASS  Install-Uv sets the astral no-PATH-write switch (UV_UNMANAGED_INSTALL / UV_NO_MODIFY_PATH)'

# Install-BrowserUseCli must keep the tool store and download cache inside
# Hermes' tree: with only UV_TOOL_BIN_DIR set, `uv tool install browser-use`
# writes the tool into the user's UV_TOOL_DIR (~/.local/share/uv/tools /
# %APPDATA%\uv\tools) and its cache into the user's UV_CACHE_DIR, so the
# tool shows up in the user's own `uv tool list`.  Static check -- the
# function performs a real network install and cannot be lifted and executed.
$browserInstallFn = Find-InstallFunction -Name 'Install-BrowserUseCli'
if (-not $browserInstallFn) {
    throw 'Install-BrowserUseCli not found in install.ps1'
}
$browserInstallText = $browserInstallFn.Extent.Text
if ($browserInstallText -notmatch 'UV_CACHE_DIR' -or
    $browserInstallText -notmatch 'UV_TOOL_DIR') {
    throw "Install-BrowserUseCli must set UV_CACHE_DIR and UV_TOOL_DIR to " +
        "$HermesHome\cache\uv / $HermesHome\uv\tools: otherwise uv tool " +
        "install browser-use writes the tool into the user's own uv tool " +
        "store and cache."
}
Write-Host '  PASS  Install-BrowserUseCli pins UV_CACHE_DIR / UV_TOOL_DIR (tool store + cache stay in Hermes tree)'

Write-Host "install.ps1 Get-ManagedUvPath (private location)"
$managed = Get-ManagedUvPath
Assert-Equal (Join-Path (Join-Path $script:HermesHome "uv") "uv.exe") $managed `
    'managed uv lives in the private uv\ dir, not bin\'

Write-Host ""
Write-Host "install.ps1 Set-UvPythonIsolationEnv (uv python write containment)"

# Save the harness's own env; restore after.
$savedUvInstallDir = $env:UV_PYTHON_INSTALL_DIR
$savedUvBin = $env:UV_PYTHON_INSTALL_BIN
$savedUvRegistry = $env:UV_PYTHON_INSTALL_REGISTRY

# An inherited value must be OVERRIDDEN, not respected -- Hermes never writes
# into a dir the user configured for their own toolchain.
$env:UV_PYTHON_INSTALL_DIR = "C:\Users\me\my-own-pythons"
$env:UV_CACHE_DIR = "C:\Users\me\my-own-cache"
$env:UV_TOOL_DIR = "C:\Users\me\my-own-tools"
Set-UvPythonIsolationEnv
Assert-Equal (Join-Path $script:HermesHome "python") $env:UV_PYTHON_INSTALL_DIR `
    'inherited UV_PYTHON_INSTALL_DIR overridden to Hermes\python'
Assert-Equal "0" $env:UV_PYTHON_INSTALL_BIN 'UV_PYTHON_INSTALL_BIN=0 (no ~/.local/bin shims)'
Assert-Equal "0" $env:UV_PYTHON_INSTALL_REGISTRY 'UV_PYTHON_INSTALL_REGISTRY=0 (no Windows registry)'
Assert-Equal (Join-Path $script:HermesHome "cache\uv") $env:UV_CACHE_DIR `
    'inherited UV_CACHE_DIR overridden to Hermes\cache\uv'
Assert-Equal (Join-Path $script:HermesHome "uv\tools") $env:UV_TOOL_DIR `
    'inherited UV_TOOL_DIR overridden to Hermes\uv\tools'

$env:UV_PYTHON_INSTALL_DIR = $savedUvInstallDir
$env:UV_PYTHON_INSTALL_BIN = $savedUvBin
$env:UV_PYTHON_INSTALL_REGISTRY = $savedUvRegistry
$env:UV_CACHE_DIR = $null
$env:UV_TOOL_DIR = $null

Write-Host ""
$legacy = Join-Path $script:HermesHome "bin\uv.exe"
New-Item -ItemType Directory -Force -Path (Split-Path $legacy -Parent) | Out-Null
Set-Content -Path $legacy -Value "@echo off`r`necho uv 0.1.2" -Encoding Ascii
New-Item -ItemType Directory -Force -Path "$script:HermesHome\uv" | Out-Null
Set-Content -Path $managed -Value "@echo off`r`necho uv 0.9.9" -Encoding Ascii

Write-Host ""
Write-Host "install.ps1 Move-LegacyManagedUv (one-time migration, uv + uvx)"

$legacyUvx = Join-Path $script:HermesHome "bin\uvx.exe"
$managedUvx = Join-Path $script:HermesHome "uv\uvx.exe"

# Legacy bin\uv.exe -> private dir.
Remove-Item $managed -Force
Assert-True (Move-LegacyManagedUv) 'legacy bin\uv.exe moved'
Assert-True (Test-Path $managed) 'managed path exists after migration'
Assert-Equal $false (Test-Path $legacy) 'legacy path gone after migration'

# Legacy bin\uvx.exe -> private dir (the astral installer always dropped
# uvx alongside uv, so a legacy bin/uvx shadows the user's uvx the same way).
Remove-Item $managedUvx -Force -ErrorAction SilentlyContinue
Set-Content -Path $legacyUvx -Value "@echo off`r`necho uvx 0.1.2" -Encoding Ascii
Assert-True (Move-LegacyManagedUv) 'legacy bin\uvx.exe moved'
Assert-True (Test-Path $managedUvx) 'managed uvx exists after migration'
Assert-Equal $false (Test-Path $legacyUvx) 'legacy uvx gone after migration'

# No-op when the managed binary already exists.
Set-Content -Path $legacy -Value "@echo off`r`necho uv 0.1.2" -Encoding Ascii
Assert-Equal $true (Move-LegacyManagedUv) 'legacy duplicate removed when managed already present'
Assert-Equal $false (Test-Path $legacy) 'legacy path gone when managed present'

# uvx duplicate also removed when the managed uvx already present.
Set-Content -Path $legacyUvx -Value "@echo off`r`necho uvx 0.1.2" -Encoding Ascii
Assert-Equal $true (Move-LegacyManagedUv) 'legacy uvx duplicate removed when managed already present'
Assert-Equal $false (Test-Path $legacyUvx) 'legacy uvx gone when managed present'

# No-op when there is no legacy binary (neither uv nor uvx).
Assert-Equal $false (Move-LegacyManagedUv) 'no-op when no legacy binaries'

# Cleanup
Remove-Item -Recurse -Force $script:HermesHome -ErrorAction SilentlyContinue

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
