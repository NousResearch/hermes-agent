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

# The MACHINE-scoped bootstrap root: HERMES_HOME is the PROFILE home, but the
# store is shared by every profile, so this fold mirrors
# get_default_hermes_root() (<root>\profiles\<name> and anything under the
# platform default fold back to the root); byte-identical to the twin in
# scripts/install.ps1 (the mirror-body test compares both with pm's answer). A
# stamped ``runtimeDir`` is not yet readable here (no Python yet); a sealed
# payload's writable store folds to this same root, so one branch covers both.
# HERMES_RUNTIME_DIR remains the explicit override.
# --- BEGIN store-root resolver (mirrored in scripts/install.ps1) ---
function Expand-HermesHomeValue {
    # os.path.expandvars + expanduser for both parameter styles: %VAR% on
    # Windows, $VAR/${VAR} on POSIX (pwsh) -- and an UNSET name stays literal,
    # the way expandvars keeps it. A leading ~ becomes the home.
    param([string]$Value)
    $pattern = '%([^%]+)%|\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
    $Value = [regex]::Replace($Value, $pattern, {
        param($m)
        $name = if ($m.Groups[1].Success) { $m.Groups[1].Value }
                elseif ($m.Groups[2].Success) { $m.Groups[2].Value }
                else { $m.Groups[3].Value }
        $v = [Environment]::GetEnvironmentVariable($name)
        if ($null -eq $v) { $m.Value } else { $v }
    })
    if ($Value -eq '~' -or $Value.StartsWith('~/') -or $Value.StartsWith('~\')) {
        if ($HOME) { $Value = $HOME + $Value.Substring(1) }
    }
    return $Value
}
function _HermesNormPath {
    # Lexical Path normalization: drop "." segments and repeated separators,
    # the way Path does before the fold compares parents. Exactly TWO leading
    # separators are kept (POSIX defines that form; Windows reads it as UNC);
    # three or more collapse to one -- pathlib folds neither differently. ".."
    # is kept: pathlib resolves it lexically to nothing here either.
    param([string]$Value, [string]$Sep)
    $isWin = $Sep -eq '\'
    $prefix = ''
    $body = $Value
    if ($isWin) {
        if ($body -match '^[A-Za-z]:') { $prefix = $body.Substring(0, 2); $body = $body.Substring(2) }
        $stripped = $body.TrimStart('\', '/')
        $leading = $body.Length - $stripped.Length
        if ($leading -ge 1 -and $prefix -eq '') {
            if ($leading -eq 2) { $prefix = '\\' } else { $prefix = '\' }
        }
        $body = $stripped
    } else {
        $stripped = $body.TrimStart('/')
        $leading = $body.Length - $stripped.Length
        if ($leading -ge 1) {
            if ($leading -eq 2) { $prefix = '//' } else { $prefix = '/' }
        }
        $body = $stripped
    }
    # POSIX does not treat a backslash as a separator -- only Windows does.
    $splitPattern = if ($isWin) { '[\\/]' } else { '/' }
    $parts = @($body -split $splitPattern | Where-Object { $_ -and $_ -ne '.' })
    if (-not $parts -or $parts.Count -eq 0) { return $prefix }
    $joined = $parts -join $Sep
    if ($prefix -and -not $prefix.EndsWith('\') -and -not $prefix.EndsWith('/')) {
        return $prefix + $Sep + $joined
    }
    return $prefix + $joined
}
function _HermesResolvePath {
    # What Path.resolve(strict=False) yields: the realpath of the longest
    # EXISTING directory prefix (reparse points followed -- a link that leaves
    # the default home can never pass the containment check), with the
    # remaining components applied lexically and ".." popping the resolved
    # parent. Relative input is joined to the process cwd first, like Path.
    param([string]$Value, [string]$Sep)
    $work = ($Value -replace '/', $Sep)
    if (-not [System.IO.Path]::IsPathRooted($work)) {
        $work = (Join-Path (Get-Location).Path $work) -replace '/', $Sep
    }
    # A bare root resolves to itself.
    if ($work -match '^[A-Za-z]:$' -or $work -match '^[A-Za-z]:[\\/]$' -or $work -eq $Sep) {
        if ($work.EndsWith($Sep)) { return $work }
        return $work + $Sep
    }
    # UNC root (\\server\share): the share is the anchor -- the existence
    # walk must not rise above it (a share may not exist on this host), and
    # ".." must not pop past it either (pathlib keeps it, like a drive root).
    $uncAnchor = if ($work -match '^\\\\[^\\/]+[\\/][^\\/]+') { $matches[0] } else { '' }
    $dir = $work.TrimEnd($Sep)
    $tail = @()
    # Test-Path resolves "."/".." first (missing "x" in "x/.." tests True), so
    # pop lexically while dots remain — a ".." left in $dir flips the
    # containment fold (ellipsis_mid_* rows). Leaf via LastIndexOf: Split-Path
    # -Leaf resolves a trailing ".." first (ellipsis_missing_mid).
    while ($true) {
        if ($uncAnchor -and $dir -eq $uncAnchor) { break }
        $hasDots = $dir -match '(^|[\\/])\.\.?([\\/]|$)'
        if (-not $hasDots -and (Test-Path -LiteralPath $dir -PathType Container)) { break }
        $idx = $dir.LastIndexOf($Sep)
        if ($idx -lt 0) { break }
        $leaf = $dir.Substring($idx + 1)
        if ($leaf -eq '') { break }  # already at a root: nothing left to pop
        $dir = if ($idx -eq 0) { $Sep } else { $dir.Substring(0, $idx) }
        # Bare "C:" is Get-Item's current dir on the drive; keep the root form.
        if ($Sep -eq '\' -and $dir -match '^[A-Za-z]:$') { $dir = $dir + $Sep }
        $tail = , $leaf + $tail
    }
    if (-not $dir) { $dir = $Sep }
    # Follow reparse points on the existing anchor (PS 5.1-compatible).
    $real = $dir
    $item = Get-Item -LiteralPath $real -Force -ErrorAction SilentlyContinue
    while ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        $target = $item.Target
        if (-not $target) { break }
        if (-not [System.IO.Path]::IsPathRooted(($target -replace '/', $Sep))) {
            $target = Join-Path (Split-Path -Parent $real) $target
        }
        $real = $target -replace '/', $Sep
        $item = Get-Item -LiteralPath $real -Force -ErrorAction SilentlyContinue
    }
    $baseComps = @(($real -split '[\\/]' | Where-Object { $_ }))
    # The rebuild restores the full prefix: collapsing to one leading
    # separator would turn \\server\share into \server\share, a different path.
    $uncPrefix = ''
    $prefix = ''
    $skip = 0
    $popFloor = 0
    if ($baseComps.Count -ge 2 -and $real -match '^\\\\') {
        $uncPrefix = '\\' + (@($baseComps[0..1]) -join $Sep)
        $skip = 2
        $popFloor = 2
    } elseif ($baseComps.Count -gt 0 -and $real -match '^[A-Za-z]:') {
        $prefix = $baseComps[0]
        $skip = 1
        $popFloor = 1
    }
    $out = @()
    foreach ($c in $tail) {
        if ($c -eq '.') { continue }
        if ($c -eq '..') {
            if ($out.Count -gt 0) {
                if ($out.Count -eq 1) { $out = @() } else { $out = @($out[0..($out.Count - 2)]) }
            } elseif ($baseComps.Count -gt $popFloor) {
                if ($baseComps.Count -eq 1) { $baseComps = @() } else { $baseComps = @($baseComps[0..($baseComps.Count - 2)]) }
            }
            # ".." below the anchor stays there, like Path("/..") or Path("C:/..")
        } else {
            $out += $c
        }
    }
    $bodyComps = @($baseComps | Select-Object -Skip $skip)
    $joined = (@($bodyComps) + $out) -join $Sep
    if ($uncPrefix) { return ($uncPrefix + $Sep + $joined).TrimEnd($Sep) }
    if ($prefix) { return $prefix + $Sep + $joined }
    return $Sep + $joined
}
function Get-HermesRoot {
    $sep = [string][System.IO.Path]::DirectorySeparatorChar
    $suffix = if ($env:HERMES_DATA_DIR_SUFFIX) { $env:HERMES_DATA_DIR_SUFFIX } else { '' }
    # _get_platform_default_hermes_home(): LOCALAPPDATA first, else the platform
    # default home of the current OS -- AppData\Local on Windows, ~/.hermes on POSIX.
    $default = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "hermes$suffix" }
               elseif ($sep -eq '\') { Join-Path $HOME "AppData/Local/hermes$suffix" }
               else { Join-Path $HOME ".hermes$suffix" }
    # get_default_hermes_root() decides containment on RESOLVED paths (so a
    # ".." or a link leaving the default home cannot pass a prefix check) but
    # folds and RETURNS the lexical form, which keeps ".." verbatim.
    $defaultLex = _HermesNormPath (Expand-HermesHomeValue $default) $sep
    $defaultLex = $defaultLex.TrimEnd('/', '\')
    $defaultRes = _HermesResolvePath $defaultLex $sep
    $rootLex = if ($env:HERMES_HOME) { Expand-HermesHomeValue $env:HERMES_HOME } else { $defaultLex }
    $rootLex = (_HermesNormPath $rootLex $sep).TrimEnd('/', '\')
    # '.' / './' normalize to nothing; answer '.' like Path('.') — an empty
    # root would make the store land on the drive root.
    if (-not $rootLex) { $rootLex = '.' }
    $rootRes = _HermesResolvePath $rootLex $sep
    # Mirror pathlib's own comparison: Windows accepts either separator and folds
    # case, POSIX does neither. A separator- or case-sensitive StartsWith here
    # skips the fold and stages uv inside the home instead of at the machine
    # root; PowerShell's bare -eq is case-insensitive on POSIX, where pathlib is
    # not, so the fold flips by platform the same way resolve()+normcase does.
    $root = $rootLex
    $default = $defaultLex
    if ($sep -eq '\') {
        $rootRes = $rootRes.Replace('/', '\')
        $defaultRes = $defaultRes.Replace('/', '\')
    }
    $cmp = if ($sep -eq '\') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if ($rootRes.Equals($defaultRes, $cmp) -or $rootRes.StartsWith($defaultRes + $sep, $cmp)) { return $default }
    $parent = Split-Path -Parent $root
    # `profiles` stays -ceq: hermes_constants compares the NAME as a str, which is
    # case-sensitive on every platform. Lexical leaf: Split-Path -Leaf resolves a
    # trailing ".." and would misread a "profiles/.." parent (ellipsis_profile_leaf).
    $pleaf = ''
    if ($parent) {
        $pidx = $parent.LastIndexOf($sep)
        $pleaf = if ($pidx -ge 0) { $parent.Substring($pidx + 1) } else { $parent }
    }
    if ($pleaf -ceq 'profiles') {
        $profileRoot = Split-Path -Parent $parent
        if (-not $profileRoot) {
            $profileRoot = if ($root.StartsWith($sep)) { $sep } else { '.' }
        }
        return $profileRoot
    }
    return $root
}
# --- END store-root resolver ---

# --- BEGIN uv state pins (mirrored in scripts/install.ps1) ---
function Set-UvStatePins {
    # uv's default state (%LOCALAPPDATA%\uv and its cache) belongs to the USER's
    # uv (#101269); pin both to the MACHINE root -- not a profile home. The
    # cache keeps its OWN slot -- pm seeds <root>\cache\uv once, skipping
    # entries that already exist, so bootstrap bytes there first would mark a
    # partial seed done -- while nothing seeds the python dir the `find` below
    # reads back.
    $hermesRoot = Get-HermesRoot
    $cache = Join-Path $hermesRoot 'cache'
    $env:UV_CACHE_DIR = Join-Path $cache 'uv-bootstrap'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $cache 'uv-python'
}
# --- END uv state pins ---
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
        Invoke-WebRequest -Uri $artifact.url -OutFile $archive
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
    $bootPy = (& $uv python find --managed-python $pyRequest) -join "`n"
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
$skillsDir = if ($env:HERMES_HOME) { Join-Path $env:HERMES_HOME 'skills' } else { Join-Path $HOME '.hermes/skills' }
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
