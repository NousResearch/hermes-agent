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
    # os.path.expandvars+expanduser for both styles: %VAR% on Windows, $VAR on
    # POSIX; an UNSET name stays literal, and ~/~user follow ntpath's rule.
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
    if ($Value.StartsWith('~')) {
        # ntpath.expanduser: USERPROFILE, else HOMEDRIVE+HOMEPATH, else unchanged;
        # PRESENCE, not truthiness: an empty USERPROFILE still substitutes ('~/x'
        # -> '/x', exactly like Python), and '~user' resolves only for the current
        # account (or a same-named profile dir).
        $idx = $Value.IndexOfAny([char[]]@('/', '\'), 1)
        if ($idx -lt 0) { $idx = $Value.Length }
        $userHome = if ($null -ne $env:USERPROFILE) { $env:USERPROFILE }
                    elseif ($null -ne $env:HOMEPATH -and $null -ne $env:HOMEDRIVE) { Join-Path $env:HOMEDRIVE $env:HOMEPATH }
                    elseif ($null -ne $HOME) { $HOME }
                    else { $null }
        if ($null -ne $userHome) {
            if ($idx -ne 1) {
                $targetUser = $Value.Substring(1, $idx - 1)
                if ($targetUser -ne $env:USERNAME) {
                    if (-not $userHome -or $env:USERNAME -ne (Split-Path -Leaf $userHome)) { return $Value }
                    $userHome = Join-Path (Split-Path -Parent $userHome) $targetUser
                }
            }
            $Value = $userHome + $Value.Substring($idx)
        }
    }
    return $Value
}
function _HermesNormPath {
    # Lexical Path normalization: drop "." and repeated separators; at most TWO
    # leading separators are kept (POSIX form / Windows UNC). ".." stays. A
    # drive-relative form ("C:rest" with NO separator after the colon) keeps
    # that shape: Windows anchors it at the drive's own current directory, and
    # rewriting it to "C:\rest" stages the store on the wrong directory.
    param([string]$Value, [string]$Sep)
    $isWin = $Sep -eq '\'
    $prefix = ''
    $driveRelative = $false
    $body = $Value
    if ($isWin) {
        if ($body -match '^([A-Za-z]:)([\\/]?)(.*)$') {
            $prefix = $matches[1]
            $driveRelative = -not [bool]$matches[2]
            $body = $matches[3]
        }
        $stripped = $body.TrimStart('\', '/')
        $leading = $body.Length - $stripped.Length
        if ($leading -ge 1 -and -not $prefix) {
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
    if (-not $parts -or $parts.Count -eq 0) {
        # A bare drive-rooted input ('C:\') keeps its root separator; a bare
        # drive ('C:') is drive-relative and must not gain one.
        if (-not $driveRelative -and $prefix -match '^[A-Za-z]:$') { return $prefix + $Sep }
        return $prefix
    }
    $joined = $parts -join $Sep
    if ($prefix -and -not $prefix.EndsWith('\') -and -not $prefix.EndsWith('/')) {
        # A bare drive prefix joins the FIRST component without a separator for
        # a drive-relative input ('C:foo'), with one for a drive-rooted input.
        if ($driveRelative) { return $prefix + $joined }
        return $prefix + $Sep + $joined
    }
    return $prefix + $joined
}
function _HermesResolvePath {
    # Path.resolve(strict=False): follow every reparse point along the chain (not
    # only the longest existing prefix), keep missing components, pop ".." off the
    # resolved prefix (never the anchor).
    param([string]$Value, [string]$Sep, [int]$Depth = 0)
    $work = ($Value -replace '/', $Sep)
    # .NET calls a drive-relative form ("C:rest") unrooted, but prepending the
    # process cwd would turn it into "C:\cwd\..." — its real anchor is the
    # drive's own current directory. Keep it as its own anchor instead.
    $driveRelative = $Sep -eq '\' -and $work -match '^[A-Za-z]:[^\\/]'
    if (-not $driveRelative -and -not [System.IO.Path]::IsPathRooted($work)) {
        $work = (Join-Path (Get-Location).Path $work) -replace '/', $Sep
    }
    # Anchor: drive root (C:\), drive-relative (C:), UNC share (\\server\share)
    # or bare root. A share exists as a FORM even when unreachable, so ".."
    # must not rise above it.
    $anchor = ''
    $body = ''
    if ($Sep -eq '\') {
        if ($work -match '^([A-Za-z]:)[\\/](.*)$') {
            $anchor = $matches[1] + $Sep
            $body = $matches[2]
        } elseif ($work -match '^([A-Za-z]:)(.*)$') {
            # Drive-relative: the anchor carries NO separator after the colon.
            $anchor = $matches[1]
            $body = $matches[2]
        } elseif ($work -match '^\\\\+([^\\/]+)[\\/]+([^\\/]+)[\\/]*(.*)$') {
            $anchor = '\\' + $matches[1] + $Sep + $matches[2]
            $body = $matches[3]
        } elseif ($work -match '^[\\/]+(.*)$') {
            $anchor = $Sep
            $body = $matches[1]
        }
    } else {
        # Path.resolve("//foo") is "/foo" on POSIX: leading separators collapse.
        $anchor = '/'
        $body = $work.TrimStart('/')
    }
    $resolved = $anchor
    $rest = $body
    while ($rest) {
        $idx = $rest.IndexOf($Sep)
        if ($idx -lt 0) { $comp = $rest; $rest = '' }
        else { $comp = $rest.Substring(0, $idx); $rest = $rest.Substring($idx + 1) }
        if (-not $comp -or $comp -eq '.') { continue }
        if ($comp -eq '..') {
            # A one-element pop empties the chain (0..-1 would keep the element).
            if ($resolved.Length -gt $anchor.Length) {
                $cut = $resolved.LastIndexOf($Sep)
                if ($cut -lt $anchor.Length) { $cut = $anchor.Length }
                $resolved = $resolved.Substring(0, $cut)
            }
            continue
        }
        # The drive-relative anchor ('C:') joins its first component with no
        # separator; every later join is ordinary.
        if ($resolved.EndsWith($Sep)) { $cand = $resolved + $comp }
        elseif ($driveRelative -and $resolved -eq $anchor) { $cand = $resolved + $comp }
        else { $cand = $resolved + $Sep + $comp }
        $item = Get-Item -LiteralPath $cand -Force -ErrorAction SilentlyContinue
        if ($item -and ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            # Follow the link (capped like realpath).
            if ($Depth -ge 40) { return _HermesNormPath $Value $Sep }
            $target = [string](@($item.Target)[0])
            if (-not $target) { $resolved = $cand; continue }
            $target = $target -replace '/', $Sep
            if ([System.IO.Path]::IsPathRooted($target)) {
                return _HermesResolvePath (($target + $Sep + $rest)) $Sep ($Depth + 1)
            }
            # Same join rule as the walk: a drive-relative anchor ('C:') adds
            # no separator, so the relative link stays drive-relative.
            if ($resolved.EndsWith($Sep) -or ($driveRelative -and $resolved -eq $anchor)) {
                $linkBase = $resolved
            } else {
                $linkBase = $resolved + $Sep
            }
            return _HermesResolvePath (($linkBase + $target + $Sep + $rest)) $Sep ($Depth + 1)
        }
        $resolved = $cand
    }
    return $resolved
}
function Get-HermesDefaultHome {
    # _get_platform_default_hermes_home(): LOCALAPPDATA first, else the platform
    # default -- AppData\Local on Windows, ~/.hermes on POSIX; the suffix is
    # appended LITERALLY. This assembled default must NOT go through
    # Expand-HermesHomeValue: a %/$ variable in the suffix would expand there
    # while pm treats the suffix as a fixed string.
    $suffix = if ($env:HERMES_DATA_DIR_SUFFIX) { $env:HERMES_DATA_DIR_SUFFIX } else { '' }
    $localAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA.Trim() } else { '' }
    $accountHome = if ($HOME) { [string]$HOME } else { '~' }
    $sep = [string][System.IO.Path]::DirectorySeparatorChar
    if ($localAppData) { return (Join-Path $localAppData "hermes$suffix") }
    if ($sep -eq '\') { return (Join-Path $accountHome "AppData/Local/hermes$suffix") }
    return (Join-Path $accountHome ".hermes$suffix")
}
function Get-HermesRoot {
    $sep = [string][System.IO.Path]::DirectorySeparatorChar
    # The default home carries the literal suffix (Get-HermesDefaultHome); only
    # the explicit HERMES_HOME below is expanded, matching _expand_hermes_home.
    $default = Get-HermesDefaultHome
    # Containment is decided on RESOLVED paths but the lexical form (keeping "..")
    # is returned, mirroring get_default_hermes_root().
    $defaultLex = _HermesNormPath $default $sep
    if ($defaultLex -notmatch '^[A-Za-z]:[\\/]$') { $defaultLex = $defaultLex.TrimEnd('/', '\') }
    $defaultRes = _HermesResolvePath $defaultLex $sep
    # pm strips the RAW env home before expanding it; whitespace-only folds to default.
    $rawHome = if ($env:HERMES_HOME) { $env:HERMES_HOME.Trim() } else { '' }
    $rootLex = if ($rawHome) { Expand-HermesHomeValue $rawHome } else { $defaultLex }
    $rootLex = _HermesNormPath $rootLex $sep
    # A bare drive root keeps its separator ('C:\'); every other path loses a
    # trailing one so a later join does not double up.
    if ($rootLex -notmatch '^[A-Za-z]:[\\/]$') { $rootLex = $rootLex.TrimEnd('/', '\') }
    # '.' like Path('.'): an empty root would land the store on the drive root.
    if (-not $rootLex) { $rootLex = '.' }
    $rootRes = _HermesResolvePath $rootLex $sep
    # Mirror pathlib's comparison: Windows accepts either separator and folds case,
    # POSIX does neither (so the fold flips by platform, as resolve()+normcase does).
    $root = $rootLex
    $default = $defaultLex
    if ($sep -eq '\') {
        $rootRes = $rootRes.Replace('/', '\')
        $defaultRes = $defaultRes.Replace('/', '\')
    }
    $cmp = if ($sep -eq '\') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
    if ($rootRes.Equals($defaultRes, $cmp) -or $rootRes.StartsWith($defaultRes + $sep, $cmp)) { return $default }
    $parent = Split-Path -Parent $root
    # `profiles` stays -ceq (hermes_constants compares the str name case-sensitively);
    # a lexical leaf, since Split-Path -Leaf resolves a trailing "..".
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
function Test-HermesFullyQualifiedPath {
    # Whether the path needs no cwd to name one directory. [System.IO.Path]
    # ::IsPathFullyQualified is .NET Core-only (missing from Windows
    # PowerShell 5.1), and IsPathRooted is NOT its answer: a drive-relative
    # ``C:rest`` and a root-relative ``\rest`` are both rooted yet still
    # anchor at a current directory, so both must be bound first.
    param([string]$Value)
    if (-not [System.IO.Path]::IsPathRooted($Value)) { return $false }
    if ([string][System.IO.Path]::DirectorySeparatorChar -ne '\') { return $true }
    if ($Value -match '^[A-Za-z]:(?![\\/])') { return $false }
    if ($Value -match '^[\\/](?![\\/])') { return $false }
    return $true
}
# --- END store-root resolver ---

# A relative HERMES_HOME names different roots before and after Push-Location:
# Get-HermesRoot deliberately returns its lexical form, but the child re-resolves
# it from the checkout. Bind one absolute home first, so the uv-state pins, the
# store slot, and PM's child all name the same root.
# Whitespace-only trims to '' — "unset" for pm (get_hermes_home strips the raw
# value before deciding); fold it to the empty home instead of dying on
# GetFullPath('').
if ($env:HERMES_HOME) {
    $e = Expand-HermesHomeValue $env:HERMES_HOME.Trim()
    if ($e -and -not (Test-HermesFullyQualifiedPath $e)) {
        $e = [System.IO.Path]::GetFullPath($e)
    }
    $env:HERMES_HOME = $e
}

# --- BEGIN uv state pins (mirrored in scripts/install.ps1) ---
function Set-UvStatePins {
    # uv's default state belongs to the USER's uv (#101269), so pin both to the
    # MACHINE root. The cache keeps its own slot — pm seeds <root>\cache\uv once,
    # skipping entries that already exist, so bootstrap bytes there first would
    # mark a partial seed done — while nothing seeds the python dir.
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
