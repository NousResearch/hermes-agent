# ============================================================================
# Windows Desktop install + update E2E driver
# ============================================================================
# Proves, on a real Windows machine, that:
#
#   1. INSTALL  - the commit-prior-to-HEAD ("BASE") installs from scratch
#                 through the real scripts/install.ps1 (uv, managed Python,
#                 Node, venv, packaged Desktop Hermes.exe).
#   2. UPDATE 1 - that BASE install updates to HEAD ("CURRENT") through the
#                 real Desktop GUI update path: scripts/desktop-update.ps1,
#                 the exact hand-off script the Update button spawns
#                 (see apps/desktop/electron/main.ts applyUpdates()).
#   3. UPDATE 2 - the now-CURRENT install updates once more to a synthetic
#                 "NEXT" commit staged on top of HEAD, proving the *outgoing*
#                 update path of the commit under test is not broken either.
#                 (before -> current proves we can be updated TO;
#                  current -> next proves we can update FROM.)
#
# HOW THE STAGING WORKS (no MITM proxy, no network fakery):
#   We bare-clone the checkout into <workroot>\serve.git and point every git
#   process at it with url.<file-url>.insteadOf rewrites for the two
#   canonical repo URLs, injected via GIT_CONFIG_{COUNT,KEY_n,VALUE_n}
#   environment variables. install.ps1's `git clone` and `hermes update`'s
#   `git fetch origin` therefore transparently hit OUR bare repo, whose
#   `main` ref we advance between legs: BASE -> CURRENT -> NEXT. The
#   installer and updater code paths run unmodified, byte-for-byte as in
#   production. Everything else (uv, PyPI, npm, PortableGit download) uses
#   the real network, same as a user install.
#
#   NEXT is a same-tree child of CURRENT (git commit-tree), so it exists
#   only in serve.git and can never leak a content change; leg 3 verifies
#   by commit SHA.
#
# WHY NOT AutoHotkey / pixel-driving the installer window (the prior
# attempt, PR #68183): image-search button clicking is resolution- and
# theme-fragile and never stabilized. The GUI Update button's entire effect
# is spawning desktop-update.ps1 with documented flags (its "CONTRACT"
# header); driving that contract directly tests the same production code
# deterministically. -NoUi and the DesktopPid wait gate exist in that script
# precisely for tests.
#
# USAGE (local Windows box or CI):
#   powershell -File tests\install\windows-desktop-e2e.ps1              # all
#   powershell -File tests\install\windows-desktop-e2e.ps1 -Phase stage
#     ... -Phase install / update-to-current / update-to-next
#   Phases share state via <workroot>\shas.json, so CI can run them as
#   separate steps for readable logs.
#
# The workroot defaults to $env:HERMES_E2E_WORKROOT or %TEMP%. Pass -Keep
# to leave everything on disk for inspection.
# ============================================================================

param(
    [ValidateSet("stage", "install", "update-to-current", "update-to-next", "all")]
    [string]$Phase = "all",

    # Repo checkout whose HEAD is the commit under test.
    [string]$RepoRoot = "",

    # Where the bare serve repo + isolated HERMES_HOME live.
    [string]$WorkRoot = $(if ($env:HERMES_E2E_WORKROOT) { $env:HERMES_E2E_WORKROOT } else { Join-Path $env:TEMP "hermes-desktop-e2e" }),

    # Keep the workroot after a successful `-Phase all` run.
    [switch]$Keep
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$ServeRepo  = Join-Path $WorkRoot "serve.git"
$HermesHome = Join-Path $WorkRoot "hermes-home"
$InstallDir = Join-Path $HermesHome "hermes-agent"
$StatePath  = Join-Path $WorkRoot "shas.json"

# The two URL spellings install.ps1 clones from and `hermes update` fetches
# from. insteadOf is prefix-based; we register the exact full forms only, so
# nothing else can accidentally rewrite to a path + ".git" suffix.
$RepoUrlHttps = "https://github.com/NousResearch/hermes-agent.git"
$RepoUrlSsh   = "git@github.com:NousResearch/hermes-agent.git"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host ("=" * 74)
    Write-Host "== $Message"
    Write-Host ("=" * 74)
}

function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        throw "E2E ASSERTION FAILED: $Message"
    }
    Write-Host "  [ok] $Message"
}

function Invoke-Git([string[]]$GitArgs, [string]$WorkDir = $null) {
    $prev = if ($WorkDir) { Get-Location } else { $null }
    # PS 5.1 trap: under $ErrorActionPreference = "Stop", a native command
    # that writes ANYTHING to stderr while merged via 2>&1 throws a
    # NativeCommandError even when it exits 0 (git loves stderr for
    # progress/notices). Relax EAP around the native call only; exit-code
    # checking below is the real error gate.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($WorkDir) { Set-Location $WorkDir }
        $output = & git @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE): $output"
        }
        return ($output | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $prevEap
        if ($prev) { Set-Location $prev }
    }
}

function Set-GitRedirect {
    # Route the canonical repo URLs to the local bare repo for THIS process
    # and every child (install.ps1's git, hermes update's git).
    #
    # MECHANISM: a driver-owned global gitconfig selected via
    # GIT_CONFIG_GLOBAL. Do NOT use GIT_CONFIG_COUNT/KEY_n/VALUE_n env
    # config here -- install.ps1 SETS those itself (GIT_CONFIG_COUNT=1,
    # windows.appendAtomically), silently clobbering any redirect we put
    # there. That exact clobber made the first CI run clone real GitHub
    # main instead of the staged BASE (caught by the HEAD-at-BASE assert).
    # install.ps1's own `git config --global` writes simply land in our
    # file, so its compat settings still apply. Nothing leaks onto the
    # machine: the file lives in the workroot and dies with it.
    $fileUrl = "file:///" + ($ServeRepo -replace "\\", "/")
    $gitCfg = Join-Path $WorkRoot "e2e-gitconfig"
    if (-not (Test-Path -LiteralPath $WorkRoot)) {
        New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
    }
    @"
[url "$fileUrl"]
	insteadOf = $RepoUrlHttps
	insteadOf = $RepoUrlSsh
"@ | Set-Content -LiteralPath $gitCfg -Encoding ASCII
    $env:GIT_CONFIG_GLOBAL = $gitCfg
    Write-Host "  git URL redirect via GIT_CONFIG_GLOBAL=$gitCfg"
    Write-Host "    $RepoUrlHttps -> $fileUrl"
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath)) {
        throw "State file not found: $StatePath -- run '-Phase stage' first."
    }
    return Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
}

function Get-InstalledHead {
    return Invoke-Git @("-C", $InstallDir, "rev-parse", "HEAD")
}

function Get-DesktopExe {
    $candidates = @(
        (Join-Path $InstallDir "apps\desktop\release\win-unpacked\Hermes.exe"),
        (Join-Path $InstallDir "apps\desktop\release\win-arm64-unpacked\Hermes.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    return $null
}

function Test-HermesRuns([string]$Label) {
    $hermesExe = Join-Path $InstallDir "venv\Scripts\hermes.exe"
    Assert-True (Test-Path -LiteralPath $hermesExe) "$Label -- venv\Scripts\hermes.exe exists"
    & $hermesExe --version 2>&1 | ForEach-Object { Write-Host "    hermes --version| $_" }
    Assert-True ($LASTEXITCODE -eq 0) "$Label -- hermes --version exits 0"
}

# ----------------------------------------------------------------------------
# Phase: stage
# ----------------------------------------------------------------------------
function Invoke-PhaseStage {
    Write-Step "STAGE: bare serve repo + BASE/CURRENT/NEXT refs"

    if (Test-Path -LiteralPath $WorkRoot) {
        Remove-Item -LiteralPath $WorkRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
    # The purge above deleted the redirect gitconfig; re-arm it so the
    # bare-clone below (and everything after) sees the redirect file.
    Set-GitRedirect

    $current = Invoke-Git @("-C", $RepoRoot, "rev-parse", "HEAD")
    $base    = Invoke-Git @("-C", $RepoRoot, "rev-parse", "HEAD~1")
    Write-Host "  CURRENT (commit under test): $current"
    Write-Host "  BASE    (its parent):        $base"

    # Bare-clone the checkout: this is the repo the installer and updater
    # will actually talk to. Local-path clone hardlinks objects, so it's
    # fast even for full history.
    Invoke-Git @("clone", "--bare", "--quiet", $RepoRoot, $ServeRepo) | Out-Null

    # Synthesize NEXT inside the bare repo: a same-tree child of CURRENT.
    # It exists nowhere else, which is the point -- leg 3 proves the commit
    # under test can update to a future main it has never seen.
    $env:GIT_AUTHOR_NAME = "Hermes E2E"; $env:GIT_AUTHOR_EMAIL = "e2e@nousresearch.com"
    $env:GIT_COMMITTER_NAME = "Hermes E2E"; $env:GIT_COMMITTER_EMAIL = "e2e@nousresearch.com"
    $next = Invoke-Git @("-C", $ServeRepo, "commit-tree", "$current^{tree}", "-p", $current, "-m", "e2e: synthetic next commit (same tree as CURRENT)")
    Write-Host "  NEXT    (synthetic):         $next"

    # Serve BASE as `main` first; update legs advance this ref.
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $base) | Out-Null
    Invoke-Git @("-C", $ServeRepo, "symbolic-ref", "HEAD", "refs/heads/main") | Out-Null

    @{ base = $base; current = $current; next = $next } |
        ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    Write-Host "  state written: $StatePath"
}

# ----------------------------------------------------------------------------
# Phase: install (BASE, via BASE's own install.ps1)
# ----------------------------------------------------------------------------
function Invoke-PhaseInstall {
    $state = Read-State
    Write-Step "INSTALL: BASE ($($state.base)) via its own install.ps1 -IncludeDesktop"

    # The honest test runs the installer *as it existed at BASE* -- a user
    # installing yesterday used yesterday's script.
    $baseInstaller = Join-Path $WorkRoot "install-base.ps1"
    & git -C $ServeRepo show "$($state.base):scripts/install.ps1" |
        Set-Content -LiteralPath $baseInstaller -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "could not extract scripts/install.ps1 at BASE" }

    $env:HERMES_HOME = $HermesHome
    New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null

    # Windows PowerShell 5.1 on purpose: it is what the production
    # `irm | iex` one-liner and the desktop bootstrap both run under.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $baseInstaller `
        -NonInteractive -SkipSetup -IncludeDesktop `
        -HermesHome $HermesHome -InstallDir $InstallDir
    Assert-True ($LASTEXITCODE -eq 0) "install.ps1 (BASE) exited 0"

    Assert-True ((Get-InstalledHead) -eq $state.base) "installed checkout is at BASE"
    Test-HermesRuns "post-install"
    $desktopExe = Get-DesktopExe
    Assert-True ($null -ne $desktopExe) "packaged Desktop Hermes.exe exists ($desktopExe)"
}

# ----------------------------------------------------------------------------
# Update legs (shared): the real Desktop GUI update path
# ----------------------------------------------------------------------------
function Invoke-DesktopUpdateLeg([string]$TargetSha, [string]$LegName) {
    Write-Step "UPDATE ($LegName): advance served main -> $TargetSha, run desktop-update.ps1"

    $env:HERMES_HOME = $HermesHome
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $TargetSha) | Out-Null

    # The hand-off script from the INSTALLED checkout drives the update --
    # exactly the production contract: each `hermes update` refreshes the
    # script that drives the NEXT update.
    $handoff = Join-Path $InstallDir "scripts\desktop-update.ps1"
    Assert-True (Test-Path -LiteralPath $handoff) "installed checkout ships scripts/desktop-update.ps1"

    # Stand in for the exiting Electron main process: desktop-update.ps1
    # FAIL-CLOSED waits for this pid to exit before touching the install.
    # A short-lived real process exercises that gate.
    $dummy = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-Command", "Start-Sleep -Seconds 3" `
        -WindowStyle Hidden -PassThru

    $resultPath = Join-Path $HermesHome ".hermes-update-result.json"
    $markerPath = Join-Path $HermesHome ".hermes-update-in-progress"
    Remove-Item -LiteralPath $resultPath -Force -ErrorAction SilentlyContinue

    # -NoUi: headless (no WinForms in CI). No -RelaunchExe: contract says
    # omit = no relaunch, so no orphaned Electron process on the runner.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $handoff `
        -InstallRoot $InstallDir -Branch main -DesktopPid $dummy.Id -NoUi
    $handoffExit = $LASTEXITCODE

    # Surface the hand-off log before asserting, so failures are debuggable
    # straight from the CI step output.
    $logPath = Join-Path $HermesHome "logs\desktop-update-handoff.log"
    if (Test-Path -LiteralPath $logPath) {
        Write-Host "  --- desktop-update-handoff.log (tail) ---"
        Get-Content -LiteralPath $logPath -Tail 40 | ForEach-Object { Write-Host "  | $_" }
    }

    Assert-True ($handoffExit -eq 0) "$LegName -- desktop-update.ps1 exited 0"

    Assert-True (Test-Path -LiteralPath $resultPath) "$LegName -- update result JSON written"
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    Assert-True ([bool]$result.ok) "$LegName -- result JSON reports ok=true ('$($result.message)')"

    Assert-True (-not (Test-Path -LiteralPath $markerPath)) "$LegName -- update marker cleaned up"
    Assert-True ((Get-InstalledHead) -eq $TargetSha) "$LegName -- checkout landed on target commit"
    Test-HermesRuns $LegName
    Assert-True ($null -ne (Get-DesktopExe)) "$LegName -- Desktop Hermes.exe still present after update"
}

function Invoke-PhaseUpdateToCurrent {
    $state = Read-State
    Invoke-DesktopUpdateLeg $state.current "BASE -> CURRENT"
}

function Invoke-PhaseUpdateToNext {
    $state = Read-State
    Invoke-DesktopUpdateLeg $state.next "CURRENT -> NEXT"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
Write-Host "Windows Desktop E2E driver"
Write-Host "  phase:    $Phase"
Write-Host "  repo:     $RepoRoot"
Write-Host "  workroot: $WorkRoot"

Set-GitRedirect

switch ($Phase) {
    "stage"             { Invoke-PhaseStage }
    "install"           { Invoke-PhaseInstall }
    "update-to-current" { Invoke-PhaseUpdateToCurrent }
    "update-to-next"    { Invoke-PhaseUpdateToNext }
    "all" {
        Invoke-PhaseStage
        Invoke-PhaseInstall
        Invoke-PhaseUpdateToCurrent
        Invoke-PhaseUpdateToNext
        if (-not $Keep) {
            Write-Step "CLEANUP"
            Remove-Item -LiteralPath $WorkRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host ""
Write-Host "Phase '$Phase' completed successfully."
