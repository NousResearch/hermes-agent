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

  PowerShell 5.1 note (Codex audit, Fix 1): `$ErrorActionPreference = 'Stop'` plus a
  directly-invoked native process is a trap - a process that merely writes to stderr
  (uv download progress, pip notices) is treated as a terminating error even when it
  exits 0. Every native call here goes through Invoke-Native, which relaxes
  `$ErrorActionPreference` for the duration and leaves success/failure entirely to
  `$LASTEXITCODE`. The whole run is wrapped so `$ErrorActionPreference`, `HERMES_HOME`
  and `UV_CACHE_DIR` are restored to their pre-run values on every exit path.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [string]$RepoRoot,
    [string]$VenvDir,
    [string]$DataDir
)

# --- path-safety helpers (Codex audit F-04) --------------------------------
# A -Force rebuild runs `Remove-Item -Recurse -Force` on $VenvDir. If $VenvDir is
# the checkout itself - or contains it, or sits inside it - that recursively
# deletes the repository. Canonicalize both paths and reject all three
# directions, for BOTH the venv and the data dir. The earlier guard only caught
# "strictly inside" (a StartsWith with a trailing separator), so passing the
# repo's own path as -VenvDir slipped through and -Force ate the checkout.
function Get-CanonicalDir {
    param([Parameter(Mandatory = $true)][string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    $trimmed = $full.TrimEnd([char]'\', [char]'/')
    if ($trimmed -match '^[A-Za-z]:$') { $trimmed += '\' }   # keep a bare drive root ("D:\")
    return $trimmed
}

function Test-PathOverlap {
    param(
        [Parameter(Mandatory = $true)][string]$A,
        [Parameter(Mandatory = $true)][string]$B
    )
    # True when A and B resolve to the same directory, or either is nested in the other.
    $ca = Get-CanonicalDir $A
    $cb = Get-CanonicalDir $B
    $ord = [System.StringComparison]::OrdinalIgnoreCase
    if ([string]::Equals($ca, $cb, $ord)) { return $true }          # equal
    $caSep = $ca.TrimEnd([char]'\') + '\'
    $cbSep = $cb.TrimEnd([char]'\') + '\'
    if ($ca.StartsWith($cbSep, $ord)) { return $true }              # A inside B
    if ($cb.StartsWith($caSep, $ord)) { return $true }              # B inside A
    return $false
}

function Invoke-Native {
    <#
      Run a side-effecting native-command scriptblock with $ErrorActionPreference
      relaxed so that stderr output alone cannot raise a terminating error on
      PowerShell 5.1 (mirrors scripts/install.ps1
      Invoke-NativeWithRelaxedErrorAction). Deliberately NO stream redirection and
      NO inner pipeline - piping a native command through a cmdlet (Out-Host,
      ForEach-Object, ...) with `2>&1` was observed to leave $LASTEXITCODE at 0
      even for a non-zero exit. The block's stdout flows out of this function; the
      one call site that runs this inside a function pipes THAT function through
      Out-Host so nothing contaminates a captured return value. The caller judges
      success by $LASTEXITCODE on the very next line - never by stderr content.
    #>
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Script } finally { $ErrorActionPreference = $prev }
}

function Get-NativeText {
    <#
      Like Invoke-Native, but RETURNS the merged stdout+stderr as one string for
      the caller to parse. Still $LASTEXITCODE-based for success; the returned text
      is data, not a success signal.
    #>
    param([Parameter(Mandatory = $true)][scriptblock]$Script)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { return (& $Script 2>&1 | Out-String) } finally { $ErrorActionPreference = $prev }
}

function Invoke-NfBootstrap {
    param(
        [bool]$Force,
        [string]$RepoRoot,
        [string]$VenvDir,
        [string]$DataDir
    )
    # Reports its result via $script:__nfExit and `return` (never `exit`), so the
    # caller's finally{} - which restores $ErrorActionPreference / HERMES_HOME /
    # UV_CACHE_DIR - always runs. The caller pipes this whole call through Out-Host
    # so native stdout streamed by the body cannot be captured as a value.

    if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $PSScriptRoot }
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
    $parent = Split-Path -Parent $RepoRoot
    $leaf = Split-Path -Leaf   $RepoRoot
    if (-not $VenvDir) { $VenvDir = Join-Path $parent ($leaf + '-venv') }
    if (-not $DataDir) { $DataDir = Join-Path $parent ($leaf + '-data') }

    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'pyproject.toml'))) {
        Write-Error "No pyproject.toml at '$RepoRoot' - is -RepoRoot correct?"
        $script:__nfExit = 1; return
    }
    foreach ($pair in @(
            [pscustomobject]@{ Name = 'venv (-VenvDir)'; Path = $VenvDir },
            [pscustomobject]@{ Name = 'data (-DataDir)'; Path = $DataDir })) {
        if (Test-PathOverlap $pair.Path $RepoRoot) {
            Write-Error ("Refusing to bootstrap: the $($pair.Name) path " +
                "'$(Get-CanonicalDir $pair.Path)' is the checkout itself, is inside it, or " +
                "contains it. The venv and data dirs MUST live OUTSIDE the checkout - a " +
                "-Force rebuild runs 'Remove-Item -Recurse -Force' on the venv dir, so this " +
                "would delete the repository. Pass a different -VenvDir / -DataDir (or run " +
                "from a checkout that has a real parent directory).")
            $script:__nfExit = 1; return
        }
    }

    $pyExe = Join-Path $VenvDir 'Scripts\python.exe'
    $hermes = Join-Path $VenvDir 'Scripts\hermes.exe'
    $marker = Join-Path $VenvDir '.nf-bootstrapped'

    Write-Host "North Forge bootstrap" -ForegroundColor Cyan
    Write-Host "  repo : $RepoRoot"
    Write-Host "  venv : $VenvDir"
    Write-Host "  data : $DataDir   (HERMES_HOME)"
    Write-Host ""

    # --- "already bootstrapped?" - a REAL probe, not an existence check ---------
    # hermes.exe + .nf-bootstrapped being present is NOT proof the venv can run: a
    # venv built on another machine is not portable, and a renamed / copied /
    # re-lettered checkout leaves both files in place while the interpreter fails or
    # imports stale code from a path that no longer exists (ERR-2026-09-07-006). Run
    # the same four-check readiness probe the launcher uses; only skip the rebuild
    # when it actually passes. A failing probe falls through to the -Force path below
    # and rebuilds the VENV ONLY - the data folder is never touched.
    . (Join-Path $PSScriptRoot 'lib\nf-readiness.ps1')

    if (-not $Force -and (Test-Path -LiteralPath $hermes) -and (Test-Path -LiteralPath $marker)) {
        $ready = Test-NfVenvReady -RepoRoot $RepoRoot -VenvDir $VenvDir
        if ($ready.Ready) {
            New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
            Write-Host "Already bootstrapped; venv passes the readiness probe. Ready - run north-forge.cmd." -ForegroundColor Green
            $script:__nfExit = 0; return
        }
        Write-Host "A venv is present but FAILS the readiness probe ($($ready.Summary)):" -ForegroundColor Yellow
        foreach ($k in $ready.Details.Keys) {
            if (-not $ready.Checks[$k]) { Write-Host "  $k : $($ready.Details[$k])" -ForegroundColor Yellow }
        }
        Write-Host "Rebuilding the venv (the data folder is left untouched)." -ForegroundColor Yellow
        $Force = $true
    }

    $uv = Get-Command uv -ErrorAction SilentlyContinue

    # --- keep uv's cache on the venv's volume --------------------------
    # uv installs by HARDLINKING packages from its cache into the venv. When the cache
    # is on a different volume than the venv (uv's default cache is under the user's
    # %LOCALAPPDATA% on C:, but a drive-native checkout builds its venv on the
    # checkout's own drive) the hardlink fails and uv falls back to a full byte copy of
    # every package - first-run bootstrap took ~6.5 min instead of seconds on an E:\
    # test. Pin the cache to a sibling of the venv so the two always share a volume.
    # An operator who has already set UV_CACHE_DIR keeps their choice. (The caller's
    # finally{} restores UV_CACHE_DIR to its pre-run value regardless.)
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
            Invoke-Native { & uv venv $VenvDir --python 3.11 }
        } else {
            $sysPy = Get-Command python -ErrorAction SilentlyContinue
            if (-not $sysPy) { Write-Error "No 'uv' and no 'python' on PATH. Install Python 3.11+ or uv, then re-run."; $script:__nfExit = 1; return }
            Invoke-Native { & $sysPy.Source -m venv $VenvDir }
        }
        if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed (exit $LASTEXITCODE)."; $script:__nfExit = 1; return }
        if (-not (Test-Path -LiteralPath $pyExe)) { Write-Error "venv creation failed - '$pyExe' not found."; $script:__nfExit = 1; return }
    }

    # --- editable install ---------------------------------------------
    Write-Host "installing North Forge (editable) into the venv - this can take a minute..."
    if ($uv) {
        Invoke-Native { & uv pip install --python $pyExe -e $RepoRoot }
    } else {
        Invoke-Native { & $pyExe -m pip install --upgrade pip }
        Invoke-Native { & $pyExe -m pip install -e $RepoRoot }
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "editable install failed (exit $LASTEXITCODE)."; $script:__nfExit = 1; return }

    # --- data folder + marker ---------------------------------------
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
        Write-Error "Could not create the data folder '$DataDir' (drive read-only, full, or disconnected?)."
        $script:__nfExit = 1; return
    }
    [System.IO.File]::WriteAllText($marker,
        "repo=$RepoRoot`nbootstrapped=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))`n",
        (New-Object System.Text.UTF8Encoding($false)))

    # --- North Forge CLI skin -------------------------------------
    # Ship skins\north-forge.yaml into HERMES_HOME\skins\ and make it the active skin
    # on a fresh install - this is what swaps the stock Hermes launch splash for North
    # Forge's own mark (DECISION-2026-09-07-001 / CHG-2026-09-07-012; banner.py is
    # untouched - hermes_cli/banner.py already prefers skin.banner_logo/banner_hero).
    # north-forge.cmd re-copies the file on every launch; this block only runs at
    # bootstrap and will NOT override a skin the operator has since chosen.
    try {
        $skinSrc = Join-Path $RepoRoot 'skins\north-forge.yaml'
        if (Test-Path -LiteralPath $skinSrc) {
            $skinDstDir = Join-Path $DataDir 'skins'
            New-Item -ItemType Directory -Path $skinDstDir -Force | Out-Null
            Copy-Item -LiteralPath $skinSrc -Destination (Join-Path $skinDstDir 'north-forge.yaml') -Force
            if (Test-Path -LiteralPath $hermes) {
                $env:HERMES_HOME = $DataDir
                $curSkin = (Get-NativeText { & $hermes config get display.skin }).Trim()
                if (-not $curSkin -or @('default', 'none', 'null') -contains $curSkin.ToLower()) {
                    Invoke-Native { & $hermes config set display.skin north-forge }
                    Write-Host "  skin : north-forge  (set as active skin)"
                } else {
                    Write-Host "  skin : north-forge available; kept your display.skin = '$curSkin'"
                }
            }
        } else {
            Write-Host "  skin : skins\north-forge.yaml not in checkout - skipped"
        }
    } catch {
        Write-Warning "North Forge skin seed skipped: $($_.Exception.Message)"
    }

    # --- drive-root launcher --------------------------------------
    # Write "<drive>:\Start North Forge.lnk" now so a freshly-bootstrapped drive has a
    # double-click entry point beside the checkout folder (CHG-2026-09-07-013).
    # north-forge.cmd also refreshes it on every launch (drive-letter self-heal).
    try {
        $mk = Join-Path $RepoRoot 'scripts\make-drive-root-shortcut.ps1'
        if (Test-Path -LiteralPath $mk) { & $mk -RepoRoot $RepoRoot }
    } catch {
        Write-Warning "drive-root shortcut skipped: $($_.Exception.Message)"
    }

    # --- tier / pinned-edition provisioning (separate admin step) --------------
    # bootstrap does NOT set a tier. Stamping a drive with Full/Basic + a pinned
    # edition is a deliberate one-time admin action (CHG-2026-09-07-022):
    #   scripts\nf-setup.ps1            (interactive)
    #   scripts\nf-setup.ps1 -NonInteractive -Tier basic -Pin penny-pincher
    # Without it the drive is un-provisioned: no edition lock, behaves like plain Hermes.
    try {
        $st = (Get-NativeText { & $pyExe -m hermes_cli.nf_tier verify --json }).Trim()
        if ($st -match '"state":\s*"active"') {
            Write-Host "  tier   : provisioned  ($st)"
        } else {
            Write-Host "  tier   : not provisioned - run  scripts\nf-setup.ps1  to set Full/Basic + a pinned edition"
        }
    } catch { }

    # --- verify -----------------------------------------------------
    # Judge success by the child's EXIT CODE, not by whether anything landed on
    # stderr (Codex audit Fix 1.3). 2>&1 merges the streams into text we only
    # *display* on failure; the pass/fail decision is $LASTEXITCODE alone.
    $env:HERMES_HOME = $DataDir
    $importText = (Get-NativeText { & $pyExe -c "import hermes_cli; print('import ok')" }).Trim()
    $importCode = $LASTEXITCODE
    Write-Host ""
    if ($importCode -ne 0) {
        Write-Error "post-install check failed: '$pyExe -c ""import hermes_cli""' exited $importCode."
        if ($importText) { Write-Host $importText }
        $script:__nfExit = 1; return
    }
    if (-not (Test-Path -LiteralPath $hermes)) {
        Write-Warning "install finished but '$hermes' is missing - check the pip output above."
        $script:__nfExit = 1; return
    }
    Write-Host "READY. venv has 'hermes' ($importText)." -ForegroundColor Green
    Write-Host "Launch:  north-forge.cmd            (repo root - sets HERMES_HOME, runs hermes)"
    Write-Host "     or:  `$env:HERMES_HOME='$DataDir'; & '$hermes'"
    $script:__nfExit = 0; return
}

# --- env save / run / restore (Codex audit Fix 1.2) --------------------------
# `exit` inside PowerShell skips finally{}, so the body runs as a function that
# `return`s its code and we `exit` exactly once, after restoration.
$script:__origEAP = $ErrorActionPreference
$script:__hadHermesHome = Test-Path Env:\HERMES_HOME
$script:__hadUvCache = Test-Path Env:\UV_CACHE_DIR
$script:__origHermesHome = if ($script:__hadHermesHome) { $env:HERMES_HOME } else { $null }
$script:__origUvCache = if ($script:__hadUvCache) { $env:UV_CACHE_DIR } else { $null }

$script:__nfExit = 1
try {
    $ErrorActionPreference = 'Stop'
    # Pipe the whole call through Out-Host so any native stdout the body streams
    # cannot land in a captured value; the real result comes back via $script:__nfExit.
    Invoke-NfBootstrap -Force:([bool]$Force) -RepoRoot $RepoRoot -VenvDir $VenvDir -DataDir $DataDir | Out-Host
} catch {
    Write-Error $_
    $script:__nfExit = 1
} finally {
    $ErrorActionPreference = $script:__origEAP
    if ($script:__hadHermesHome) { $env:HERMES_HOME = $script:__origHermesHome }
    elseif (Test-Path Env:\HERMES_HOME) { Remove-Item Env:\HERMES_HOME }
    if ($script:__hadUvCache) { $env:UV_CACHE_DIR = $script:__origUvCache }
    elseif (Test-Path Env:\UV_CACHE_DIR) { Remove-Item Env:\UV_CACHE_DIR }
}
exit $script:__nfExit
