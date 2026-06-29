# ===========================================================================
# scripts/lib/apexnodes-region-detect.ps1
# ---------------------------------------------------------------------------
# Sourceable ApexNodes overlay: region self-detection + China mirror downgrade.
# Windows twin of scripts/lib/apexnodes-region-detect.sh — keep the two in step
# (region heuristic, precedence rules, env-var names, COS layout are identical).
#
# This is the SINGLE SOURCE OF TRUTH for the "should this install use domestic
# (mainland-China) mirrors?" decision and, when yes, the mirror/COS env it sets.
# It is an ApexNodes overlay seam (see apex_overlay/README.md): the file lives in
# OUR namespace under scripts/lib/, which upstream Hermes never creates (zero
# merge-conflict surface). install.ps1 stays byte-for-byte upstream apart from
# one self-locating dot-source of this lib plus a few one-line call sites.
#
# Dot-source it (NOT call-operator) so its functions land in the caller's scope
# and can read install.ps1's script-scope state ($HermesHome, $InstallDir,
# $Commit, $Branch) and write $script:UvCmd:
#   . "$PSScriptRoot\lib\apexnodes-region-detect.ps1"
#   Resolve-ApexRegion          # sets $env:HERMES_CN_MIRRORS from the region
#   Set-ApexCnMirrorEnv         # exports CN mirror env iff CN
# COS download helpers (also defined here):
#   Install-UvFromCos           # sets $script:UvCmd on success (CN only)
#   Install-RuntimeFromCos      # populates $InstallDir (CN only)
#
# OFF by default: with $env:HERMES_CN_MIRRORS unset/0 every function is a no-op
# and the installer behaves byte-for-byte like upstream.
#
# install.ps1 provides Write-Info / Write-Warn / Write-Success and Get-WindowsArch
# before these run. When this lib is dot-sourced standalone (e.g. a unit test)
# those may be absent, so define no-op-safe fallbacks here — only if missing.
# ===========================================================================

if (-not (Get-Command Write-Info -ErrorAction SilentlyContinue)) {
    function Write-Info    { param([string]$Message) Write-Host $Message }
}
if (-not (Get-Command Write-Warn -ErrorAction SilentlyContinue)) {
    function Write-Warn    { param([string]$Message) Write-Host $Message }
}
if (-not (Get-Command Write-Success -ErrorAction SilentlyContinue)) {
    function Write-Success { param([string]$Message) Write-Host $Message }
}

# ===========================================================================
# ApexNodes China mirror mode (opt-in via HERMES_CN_MIRRORS=1)
# ===========================================================================
# OFF by default: with the flag unset Set-ApexCnMirrorEnv does nothing and the
# installer behaves byte-for-byte like upstream. The packaged ApexNodes desktop
# sets HERMES_CN_MIRRORS=1 (and, once provisioned, HERMES_RUNTIME_COS_BASE) from
# electron/bootstrap-runner.cjs so a fresh mainland-China machine installs
# without reaching github.com / pypi.org / registry.npmjs.org directly. Our
# runtime source + uv (no public CN mirror exists) come from our public-read COS
# bucket (see Install-RuntimeFromCos / Install-UvFromCos); every public third-
# party dep uses an established CN mirror below. Each value only sets when unset
# so an operator can override any single mirror via the real environment.
function Test-CnEnabled { return ($env:HERMES_CN_MIRRORS -eq "1") }

# ===========================================================================
# ApexNodes region detection (decides whether CN mirror mode turns on)
# ===========================================================================
# Twin of the "ApexNodes region detection" block in install.sh. It only ever
# decides a region and, from it, sets $env:HERMES_CN_MIRRORS -- the mirror URLs
# stay defined solely in Set-ApexCnMirrorEnv below. Precedence (highest first):
#   1. $env:HERMES_CN_MIRRORS already set -> respect verbatim, skip detection.
#      (The packaged desktop / ops set this, so existing behavior is unchanged.)
#   2. $env:APEXNODES_REGION = cn|global -> explicit operator/user override.
#   3. neither set -> auto-detect mainland China, defaulting to "global" on any
#      doubt (a wrong guess only slows a download; it must never break install).
# Decision is cached in $HermesHome\.apexnodes-region so per-stage processes
# probe the network at most once. Diagnostics use Write-Info/Write-Warn, which
# write to the PowerShell information stream -- never stdout -- so the manifest /
# stage JSON frames the bootstrap runner parses stay clean.

# Cheap offline gate: is the machine's timezone plausibly mainland China? CST
# (UTC+8) also covers HK/Taiwan/Singapore, so this only gates the network probe.
function Test-TimezoneSuggestsCn {
    try {
        $tz = Get-TimeZone
        if ($tz.Id -match 'China|Taipei' ) { return $true }  # "China Standard Time"
        # Numeric offset fallback (UTC+08:00). Combined with the probe this still
        # isolates the mainland from other +0800 regions.
        if ($tz.BaseUtcOffset -eq ([TimeSpan]::FromHours(8))) { return $true }
    } catch { }
    return $false
}

# Decisive probe: race a domestic endpoint against a foreign one with short
# timeouts. Classify CN only when domestic SUCCEEDS and foreign FAILS -- a
# conservative AND, so transient flakiness biases toward "global" (defaults).
function Test-NetworkSuggestsCn {
    $cnUrl = "https://registry.npmmirror.com/"
    $foreignUrl = "https://registry.npmjs.org/"
    $cnOk = $false
    $foreignOk = $false
    try { Invoke-WebRequest -Uri $cnUrl -Method Head -TimeoutSec 4 -UseBasicParsing | Out-Null; $cnOk = $true } catch { }
    try { Invoke-WebRequest -Uri $foreignUrl -Method Head -TimeoutSec 4 -UseBasicParsing | Out-Null; $foreignOk = $true } catch { }
    return ($cnOk -and -not $foreignOk)
}

function Resolve-ApexRegion {
    # Rule 1: explicit HERMES_CN_MIRRORS wins; do not touch it, do not probe.
    if (-not [string]::IsNullOrEmpty($env:HERMES_CN_MIRRORS)) { return }

    # Rule 2: explicit region knob.
    $region = ("$env:APEXNODES_REGION").Trim().ToLowerInvariant()
    switch ($region) {
        { $_ -in @('cn','china','mainland') } {
            $env:HERMES_CN_MIRRORS = '1'
            Write-Info "ApexNodes region: cn (from APEXNODES_REGION) -- using China mirrors"
            return
        }
        { $_ -in @('global','intl','international','foreign','row') } {
            $env:HERMES_CN_MIRRORS = '0'
            Write-Info "ApexNodes region: global (from APEXNODES_REGION) -- using default sources"
            return
        }
        '' { }  # fall through to auto-detect
        default {
            Write-Warn "Unknown APEXNODES_REGION='$env:APEXNODES_REGION' (expected cn|global) -- auto-detecting"
        }
    }

    # Rule 3: auto-detect. Reuse a cached decision from an earlier stage process.
    $cache = Join-Path $HermesHome ".apexnodes-region"
    if (Test-Path $cache) {
        $cached = (Get-Content $cache -ErrorAction SilentlyContinue | Select-Object -First 1)
        $cached = ("$cached").Trim().ToLowerInvariant()
        if ($cached -eq 'cn')     { $env:HERMES_CN_MIRRORS = '1'; return }
        if ($cached -eq 'global') { $env:HERMES_CN_MIRRORS = '0'; return }
    }

    # No cache: decide now. Gate the probe on the cheap timezone hint so the
    # common (non-CN) case skips the network entirely.
    $detected = 'global'
    if (Test-TimezoneSuggestsCn) {
        if (Test-NetworkSuggestsCn) { $detected = 'cn' }
    }

    # Persist for sibling stage processes (best-effort; never fail the install).
    try {
        if (-not (Test-Path $HermesHome)) { New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null }
        Set-Content -Path $cache -Value $detected -Encoding ASCII -ErrorAction SilentlyContinue
    } catch { }

    if ($detected -eq 'cn') {
        $env:HERMES_CN_MIRRORS = '1'
        Write-Info "ApexNodes region: cn (auto-detected) -- using China mirrors"
        Write-Info "  (override with APEXNODES_REGION=global if this is wrong)"
    } else {
        $env:HERMES_CN_MIRRORS = '0'
        # Quiet on the global path to keep upstream/CI output byte-clean.
    }
}

# Export the CN mirror env (no-op unless CN mode is on). Idempotent: every value
# only sets when unset so a pre-set operator override is preserved.
function Set-ApexCnMirrorEnv {
    if (-not (Test-CnEnabled)) { return }
    # Python package index -> Tsinghua TUNA (PyPI mirror).
    if (-not $env:UV_DEFAULT_INDEX)         { $env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple" }
    if (-not $env:UV_INDEX_URL)             { $env:UV_INDEX_URL = $env:UV_DEFAULT_INDEX }
    if (-not $env:PIP_INDEX_URL)            { $env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple" }
    # uv-managed CPython (astral python-build-standalone) -> npmmirror binary mirror.
    if (-not $env:UV_PYTHON_INSTALL_MIRROR) { $env:UV_PYTHON_INSTALL_MIRROR = "https://registry.npmmirror.com/-/binary/python-build-standalone" }
    # npm registry + Electron binaries -> npmmirror.
    if (-not $env:npm_config_registry)      { $env:npm_config_registry = "https://registry.npmmirror.com" }
    if (-not $env:ELECTRON_MIRROR)          { $env:ELECTRON_MIRROR = "https://npmmirror.com/mirrors/electron/" }
    # Node.js dist tarballs -> npmmirror binary mirror (consumed by Install-Node).
    if (-not $env:HERMES_NODE_DIST_BASE)    { $env:HERMES_NODE_DIST_BASE = "https://registry.npmmirror.com/-/binary/node" }
}

# ===========================================================================
# CN-mode COS download helpers
# ===========================================================================

# CN mode: fetch a prebuilt uv from our public-read COS bucket instead of the
# astral.sh installer (which downloads from github.com, blocked in mainland
# China). Mirrors install.sh's apexnodes_install_uv_from_cos. The publish script
# ships uv-<triple>.zip (astral's Windows uv is a .zip, not .tar.gz). Returns
# $true with $script:UvCmd set on success; $false on any failure so Install-Uv
# falls through to the astral path.
function Install-UvFromCos {
    if (-not (Test-CnEnabled)) { return $false }
    if ([string]::IsNullOrWhiteSpace($env:HERMES_RUNTIME_COS_BASE)) { return $false }

    $arch = Get-WindowsArch
    $triple = if ($arch -eq 'arm64') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
    $base = $env:HERMES_RUNTIME_COS_BASE.TrimEnd('/')
    $url = "$base/uv-$triple.zip"
    $binDir = Join-Path $HermesHome "bin"
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    $tmp = Join-Path $env:TEMP ("hermes-uv-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        Write-Info "Fetching uv from COS mirror: $url"
        Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp "uv.zip") -UseBasicParsing
        Expand-Archive -Path (Join-Path $tmp "uv.zip") -DestinationPath $tmp -Force
        $uvExe = Get-ChildItem -Path $tmp -Recurse -Filter "uv.exe" | Select-Object -First 1
        if (-not $uvExe) {
            Write-Warn "uv.exe not found inside COS archive -- will try the astral.sh installer"
            return $false
        }
        Copy-Item $uvExe.FullName (Join-Path $binDir "uv.exe") -Force
        $uvxExe = Get-ChildItem -Path $tmp -Recurse -Filter "uvx.exe" | Select-Object -First 1
        if ($uvxExe) { Copy-Item $uvxExe.FullName (Join-Path $binDir "uvx.exe") -Force }
        $managedUv = Join-Path $binDir "uv.exe"
        if (-not (Test-Path $managedUv)) { return $false }
        $script:UvCmd = $managedUv
        $version = & $managedUv --version
        Write-Success "Managed uv installed from COS mirror ($version)"
        return $true
    } catch {
        Write-Warn "COS uv install failed ($_) -- will try the astral.sh installer"
        return $false
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

# CN mode: download the pinned runtime source tarball from our public-read COS
# bucket instead of git-cloning github.com (blocked/slow in mainland China).
# Mirrors install.sh's apexnodes_download_runtime_tarball -- the tarball is `git
# archive --prefix=hermes-agent/` of the pinned commit (clean source tree, NO
# .git), keyed by the pinned commit (preferred) or branch so the COS object
# matches the desktop build stamp. Returns $true with $InstallDir populated on
# success; $false (and $InstallDir removed) on any failure so Install-Repository
# falls back to a normal git clone.
function Install-RuntimeFromCos {
    if (-not (Test-CnEnabled)) { return $false }
    if ([string]::IsNullOrWhiteSpace($env:HERMES_RUNTIME_COS_BASE)) { return $false }

    $key = if ($Commit) { $Commit } else { $Branch }
    if ([string]::IsNullOrWhiteSpace($key)) { return $false }

    $base = $env:HERMES_RUNTIME_COS_BASE.TrimEnd('/')
    $url = "$base/hermes-agent-$key.tar.gz"
    $tmp = Join-Path $env:TEMP ("hermes-src-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $tarball = Join-Path $tmp "runtime.tar.gz"
    try {
        Write-Info "Downloading runtime source from COS mirror: $url"
        Invoke-WebRequest -Uri $url -OutFile $tarball -UseBasicParsing
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        # Archive built with --prefix=hermes-agent/, so strip the leading dir.
        # Windows 10 1803+ ships bsdtar (tar.exe) which handles .tar.gz natively.
        & tar -xzf $tarball -C $InstallDir --strip-components=1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "COS runtime tarball could not be extracted -- falling back to git clone"
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            return $false
        }
        if (-not (Test-Path (Join-Path $InstallDir "pyproject.toml"))) {
            Write-Warn "COS runtime tarball missing pyproject.toml -- falling back to git clone"
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            return $false
        }
        Write-Success "Runtime source ready from COS mirror ($key)"
        return $true
    } catch {
        Write-Warn "COS runtime download failed ($_) -- falling back to git clone"
        if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
        return $false
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}
