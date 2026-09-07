# bootstrap-north-forge - one-time setup so a fresh clone can launch: makes a venv and a data folder OUTSIDE the checkout, installs North Forge editable from the checkout. Run once by hand (or let north-forge.cmd run it on first launch).
<#
  scripts\bootstrap-north-forge.ps1  [-Force]  [-RepoRoot <path>]  [-VenvDir <path>]  [-DataDir <path>]

  WHAT IT DOES  (minimal, single-drive, single-folder-tree)
    - Creates a Python venv as a SIBLING of the checkout:  <parent>\<checkout-name>-venv
      (never inside the checkout: a venv inside the tree the agent operates on can be
       wiped by one of the agent's own relative-path commands.)
    - Creates a data folder as a SIBLING:                  <parent>\<checkout-name>-data
      and reports it as the HERMES_HOME the launcher will use.
    - Installs this checkout editable into that venv (`uv pip install -e .`, or
      `python -m pip install -e .` if uv is absent).
    - Writes <venv>\.nf-bootstrapped (repo path + timestamp) as the "ready" marker.

  This is intentionally NOT the hardened install: no sealed drive, no
  NORTHFORGE / NORTHFORGE-DATA two-volume split, no certify/verify/audit. That is
  future work, gated on DECISION-2026-09-06-003 being ratified. See BRANDING.md
  and logs/ledger/decisions/DECISION-LOG.md.

  AFTER THIS: run  north-forge.cmd  (repo root) - it sets HERMES_HOME and starts `hermes`.

  EXIT 0 = venv ready.  1 = failed.
  REQUIRES  Python 3.11+ on PATH (or uv, which fetches its own). PowerShell 5.1+.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [string]$RepoRoot,
    [string]$VenvDir,
    [string]$DataDir
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$parent   = Split-Path -Parent $RepoRoot
$leaf     = Split-Path -Leaf   $RepoRoot
if (-not $VenvDir) { $VenvDir = Join-Path $parent ($leaf + '-venv') }
if (-not $DataDir) { $DataDir = Join-Path $parent ($leaf + '-data') }

if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'pyproject.toml'))) {
    Write-Error "No pyproject.toml at '$RepoRoot' - is -RepoRoot correct?"
    exit 1
}
foreach ($d in @($VenvDir, $DataDir)) {
    $full = [System.IO.Path]::GetFullPath($d)
    if ($full.TrimEnd('\').ToLower().StartsWith($RepoRoot.TrimEnd('\').ToLower() + '\')) {
        Write-Error "'$full' is inside the checkout. venv and data MUST be siblings of it. Pick another -VenvDir/-DataDir."
        exit 1
    }
}

$pyExe   = Join-Path $VenvDir 'Scripts\python.exe'
$hermes  = Join-Path $VenvDir 'Scripts\hermes.exe'
$marker  = Join-Path $VenvDir '.nf-bootstrapped'

Write-Host "North Forge bootstrap" -ForegroundColor Cyan
Write-Host "  repo : $RepoRoot"
Write-Host "  venv : $VenvDir"
Write-Host "  data : $DataDir   (HERMES_HOME)"
Write-Host ""

if ((Test-Path -LiteralPath $hermes) -and (Test-Path -LiteralPath $marker) -and -not $Force) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    Write-Host "Already bootstrapped (pass -Force to rebuild). Ready - run north-forge.cmd." -ForegroundColor Green
    exit 0
}

$uv = Get-Command uv -ErrorAction SilentlyContinue

# --- keep uv's cache on the venv's volume --------------------------
# uv installs by HARDLINKING packages from its cache into the venv. When the cache
# is on a different volume than the venv (uv's default cache is under the user's
# %LOCALAPPDATA% on C:, but a drive-native checkout builds its venv on the
# checkout's own drive) the hardlink fails and uv falls back to a full byte copy of
# every package - first-run bootstrap took ~6.5 min instead of seconds on an E:\
# test. Pin the cache to a sibling of the venv so the two always share a volume.
# An operator who has already set UV_CACHE_DIR keeps their choice.
if ($uv -and -not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $parent '.uv-cache'
    Write-Host "  cache: $env:UV_CACHE_DIR   (same volume as the venv -> hardlink, not copy)"
}

# --- venv ------------------------------------------------------------
if ($Force -and (Test-Path -LiteralPath $VenvDir)) {
    Write-Host "removing existing venv (-Force)..."
    Remove-Item -LiteralPath $VenvDir -Recurse -Force
}
if (-not (Test-Path -LiteralPath $pyExe)) {
    Write-Host "creating venv..."
    if ($uv) {
        & uv venv $VenvDir --python 3.11
    } else {
        $sysPy = Get-Command python -ErrorAction SilentlyContinue
        if (-not $sysPy) { Write-Error "No 'uv' and no 'python' on PATH. Install Python 3.11+ or uv, then re-run."; exit 1 }
        & $sysPy.Source -m venv $VenvDir
    }
    if (-not (Test-Path -LiteralPath $pyExe)) { Write-Error "venv creation failed - '$pyExe' not found."; exit 1 }
}

# --- editable install ---------------------------------------------
Write-Host "installing North Forge (editable) into the venv - this can take a minute..."
if ($uv) {
    & uv pip install --python $pyExe -e $RepoRoot
} else {
    & $pyExe -m pip install --upgrade pip
    & $pyExe -m pip install -e $RepoRoot
}
if ($LASTEXITCODE -ne 0) { Write-Error "editable install failed (exit $LASTEXITCODE)."; exit 1 }

# --- data folder + marker ---------------------------------------
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
[System.IO.File]::WriteAllText($marker,
    "repo=$RepoRoot`nbootstrapped=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))`n",
    (New-Object System.Text.UTF8Encoding($false)))

# --- verify -----------------------------------------------------
$env:HERMES_HOME = $DataDir
$ver = (& $pyExe -c "import hermes_cli; print('import ok')" 2>&1 | Out-String).Trim()
Write-Host ""
if (Test-Path -LiteralPath $hermes) {
    Write-Host "READY. venv has 'hermes' ($ver)." -ForegroundColor Green
    Write-Host "Launch:  north-forge.cmd            (repo root - sets HERMES_HOME, runs hermes)"
    Write-Host "     or:  `$env:HERMES_HOME='$DataDir'; & '$hermes'"
    exit 0
} else {
    Write-Warning "install finished but '$hermes' is missing - check the pip output above."
    exit 1
}
