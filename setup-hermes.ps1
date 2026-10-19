# ============================================================================
# Hermes Agent Setup Script (Windows) — THE dev-environment entry point.
# ============================================================================
# Sets up the pm-managed development environment from a fresh clone:
#   1. Stage the pinned uv from pm/lock.json (sha256-verified, into the pm
#      store slot) - pm needs uv to bootstrap, so it cannot stage uv itself.
#   2. Use uv to install and locate bootstrap Python, then let uv exit.
#      Run `python -m pm.cli install` directly so PM can safely replace uv.
#      PM owns the final interpreter, tool store, and dependency generation.
#   3. Point you at `.\activate.ps1` - the venv-style way to put the pm env
#      (PATH + tool vars) into your current session.
# ============================================================================
# Setup and activation both prepare the isolated test interpreter; installers
# invoke pm.cli directly and do not select it. -TestExtras overrides coverage.
param([switch]$RuntimeOnly, [string]$TestExtras = '')
$ErrorActionPreference = 'Stop'

# The resolver folds HERMES_HOME to its machine root the way pm does
# (<root>\profiles\<name> and anything under the platform default fold back to
# the root). setup runs from a checkout, so it dot-sources the canonical file
# instead of carrying a copy. HERMES_RUNTIME_DIR is the explicit override; a
# stamped runtimeDir is unreadable here (no Python yet), and a sealed payload's
# writable store folds to this same root.
. (Join-Path $PSScriptRoot "scripts\bootstrap\store_root.ps1")

# A relative HERMES_HOME names different roots before and after Push-Location:
# Get-HermesRoot returns its lexical form and the child re-resolves it from the
# checkout. Bind one absolute home first, so the uv-state pins, the store slot
# and PM's child all name the same root. A whitespace-only value trims to '' and
# is unset for pm, so it stays empty instead of reaching GetFullPath('').
#
# BEGIN/END markers: tests/scripts/install/test_bootstrap_store_root_matches_pm.py
# executes this block on its own to prove it binds before Push-Location. The
# markers are the extraction contract, so a behaviour change here cannot silently
# stop being tested (a regex over the body would).
# --- BEGIN explicit-home binding ---
if ($env:HERMES_HOME) {
    $expr = $env:HERMES_HOME.Trim()
    $e = Expand-HermesHomeValue $expr
    # Export the expression when its expansion still carries a reference, or PM
    # expands the twice-expanded value again (Get-HermesHomeExport).
    $export = Get-HermesHomeExport $expr $e
    if (-not $export) {
        if ($e -and -not (Test-HermesFullyQualifiedPath $e)) {
            $e = [System.IO.Path]::GetFullPath($e)
        }
        $export = $e
    }
    $env:HERMES_HOME = $export
}
# --- END explicit-home binding ---

# uv's cache and managed-python dirs, dot-sourced from the canonical file like
# the resolver above.
. (Join-Path $PSScriptRoot "scripts\bootstrap\uv_state_pins.ps1")
Set-UvStatePins

Write-Host ''
Write-Host 'Hermes Agent Setup' -ForegroundColor Cyan
Write-Host ''

$repo = $PSScriptRoot
$lockPath = Join-Path $repo 'pm/lock.json'
if (-not (Test-Path $lockPath)) { throw 'pm/lock.json not found' }
$lock = Get-Content -Raw $lockPath | ConvertFrom-Json

# -ErrorAction SilentlyContinue: a restricted host must fall back to x64 like
# install.ps1's Get-WindowsArch, not die on the registry probe.
$machineArch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -ErrorAction SilentlyContinue).PROCESSOR_ARCHITECTURE
$arch = if ($machineArch -eq 'ARM64') { 'arm64' } else { 'x64' }
$target = "win32-$arch"

# ---------------------------------------------------------------------------
# Stage the pinned uv from pm/lock.json into the pm store slot
# ---------------------------------------------------------------------------
$uvPin = $lock.packages.uv
if (-not $uvPin) { throw 'no uv pin in pm/lock.json' }
$artifact = $uvPin.artifacts.$target
if (-not $artifact) { $artifact = $uvPin.artifacts.any }
if (-not $artifact) { throw "no uv artifact for $target" }

$pyPin = $lock.packages.python
$pyVersion = if ($pyPin) { ($pyPin.version -split '\+')[0] -replace '^(\d+\.\d+).*', '$1' } else { '3.14' }

# A relative HERMES_RUNTIME_DIR names two stores across the Push-Location
# (this script reads it here, PM's child resolves it from the repo dir).
# Anchor it once so both sides bind one store.
if ($env:HERMES_RUNTIME_DIR -and -not (Test-HermesFullyQualifiedPath $env:HERMES_RUNTIME_DIR)) {
    $env:HERMES_RUNTIME_DIR = [System.IO.Path]::GetFullPath($env:HERMES_RUNTIME_DIR)
}
$store = if ($env:HERMES_RUNTIME_DIR) { $env:HERMES_RUNTIME_DIR } else { Join-Path (Get-HermesRoot) 'tools' }
$entry = Join-Path $store "uv-$($uvPin.version)-$target"
$uv = Join-Path $entry 'uv.exe'

if (Test-Path $uv) {
    Write-Host ("pinned uv found: " + (& $uv --version)) -ForegroundColor Green
} else {
    Write-Host "Staging pinned uv $($uvPin.version) ($target) into the pm store..." -ForegroundColor Cyan
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-setup-" + [guid]::NewGuid().ToString('n'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $archive = Join-Path $tmp ([uri]$artifact.url).Segments[-1]
        Invoke-WebRequest -Uri $artifact.url -OutFile $archive -UseBasicParsing -TimeoutSec 600
        $got = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
        if ($got -ne $artifact.sha256) {
            throw "sha256 mismatch for uv (got $got, pinned $($artifact.sha256))"
        }
        $tree = Join-Path $tmp 'tree'
        Expand-Archive -Path $archive -DestinationPath $tree
        # flatten a single wrapping dir
        $inner = @(Get-ChildItem $tree)
        $src = if ($inner.Count -eq 1 -and $inner[0].PSIsContainer) { $inner[0].FullName } else { $tree }
        New-Item -ItemType Directory -Force -Path $store | Out-Null
        if (Test-Path $entry) { Remove-Item -Recurse -Force $entry }
        Move-Item $src $entry
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
    Write-Host ("uv installed: " + (& $uv --version)) -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# PM installs the tools, then the venv. On ARM64 PM prepares the compiler and
# OpenSSL environment for that sync itself (pm/native_build.py), after its own
# git is published, so every install path builds the same way.
# Activation trusts the recorded tool digest. A direct setup re-checks it.
# ---------------------------------------------------------------------------
Write-Host 'Installing python + tools + dependencies via pm (hash-verified via uv.lock)...' -ForegroundColor Cyan
Write-Host '(first run on a fresh checkout can take 1-5 minutes)'
Push-Location $repo
try {
    # PM can replace its uv entry only after the bootstrap uv has exited.
    # A bare version lets uv pick emulated x86_64 on Windows-on-ARM.
    $pyRequest = "cpython-$pyVersion-windows-$(if ($arch -eq 'arm64') { 'aarch64' } else { 'x86_64' })-none"
    & $uv python install --no-bin --no-registry $pyRequest
    if ($LASTEXITCODE -ne 0) { throw 'bootstrap Python installation failed' }
    # Like scripts/install.ps1, decode uv's UTF-8 path independently of the
    # caller's console code page. Keep this scope local: install.ps1 must also
    # work as a standalone download before a checkout (and shared files) exists.
    $previousNativeOutputEncoding = [Console]::OutputEncoding
    try {
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
        $bootPy = (& $uv python find --managed-python $pyRequest) -join "`n"
    } finally {
        [Console]::OutputEncoding = $previousNativeOutputEncoding
    }
    if ($LASTEXITCODE -ne 0 -or -not $bootPy) { throw 'bootstrap Python lookup failed' }
    & $bootPy.Trim() -m pm.cli install $(if ($RuntimeOnly) { '--trust-recorded' }) "--test-environment=$TestExtras"
    if ($LASTEXITCODE -ne 0) { throw 'pm install failed - see output above.' }
} finally {
    Pop-Location
}
Write-Host 'Tools + dependencies installed (hash-verified via pm + uv.lock)' -ForegroundColor Green

if ($RuntimeOnly) { exit 0 }

# ---------------------------------------------------------------------------
# Environment file
# ---------------------------------------------------------------------------
$envFile = Join-Path $repo '.env'
if (-not (Test-Path $envFile)) {
    if (Test-Path (Join-Path $repo '.env.example')) {
        Copy-Item (Join-Path $repo '.env.example') $envFile
        Write-Host 'Created .env from template' -ForegroundColor Green
    }
} else {
    Write-Host '.env exists' -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Seed bundled skills into ~/.hermes/skills/
# ---------------------------------------------------------------------------
$skillsDir = if ($env:HERMES_HOME -and $env:HERMES_HOME.Trim()) {
    $e = Expand-HermesHomeValue $env:HERMES_HOME.Trim()
    if (-not (Test-HermesFullyQualifiedPath $e)) { $e = [System.IO.Path]::GetFullPath($e) }
    Join-Path $e 'skills'
} else {
    # Unset home: the platform default (LOCALAPPDATA, suffix), exactly what the
    # child's get_hermes_home() seeds — $HOME\.hermes is only the POSIX answer.
    Join-Path (Get-HermesRoot) 'skills'
}
New-Item -ItemType Directory -Force -Path $skillsDir | Out-Null
$sync = Join-Path $repo 'tools/skills_sync.py'
$venvPy = Join-Path $repo 'venv/Scripts/python.exe'
if ((Test-Path $sync) -and (Test-Path $venvPy)) {
    & $venvPy $sync 2>$null
    Write-Host 'Skills synced' -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host 'Setup complete!' -ForegroundColor Green
Write-Host ''
Write-Host 'Next steps:'
Write-Host ''
Write-Host '  1. Activate the dev environment (venv-style, in THIS session):'
Write-Host '     .\activate.ps1'
Write-Host ''
Write-Host '  2. Run the setup wizard to configure API keys:'
Write-Host '     hermes setup'
Write-Host ''
Write-Host '  3. Start chatting:'
Write-Host '     hermes'
Write-Host ''
Write-Host 'Other commands:'
Write-Host '  hermes pm install     # Re-run the tool + dependency install'
Write-Host '  hermes status         # Check configuration'
Write-Host '  hermes doctor         # Diagnose issues'
Write-Host '  deactivate            # Undo the activation (restore PATH etc.)'
Write-Host ''
