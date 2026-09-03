# ============================================================================
# Windows Desktop GUI install + update E2E driver (the REAL user flow)
# ============================================================================
# Proves, on a real Windows machine, that a user who installs Hermes the way
# the website tells them to can then update to the commit under test through
# a real update surface -- with every leg driven through the GUI a user
# actually touches:
#
#   INSTALL   - downloads the production Hermes-Setup.exe from the website,
#               launches it HEADED, and AutoHotkey clicks Install, waits,
#               then clicks Launch. The real Electron Hermes.exe must appear.
#               The exe runs EXACTLY as shipped against serve.git, whose
#               `main` is parked at OLD (-InstallRef, default: the newest
#               release tag) -- so the install lands on OLD the same way a
#               user's install landed on whatever main served that day.
#   UPDATE    - OLD -> HEAD through the route selected by -Route:
#                 desktop    (implemented) launch the installed Hermes.exe
#                            under Playwright's Electron driver and CLICK
#                            Settings -> About -> "Update now". The
#                            production hand-off chain runs untouched:
#                            marker, app quit, detached updater, `hermes
#                            update`, desktop rebuild, RELAUNCH. Asserts
#                            target sha, marker cleanup, result JSON (when
#                            the script path wrote one), working hermes,
#                            and the relaunched app window.
#                 update     run `hermes update` from the installed venv
#                            (the CLI route a GUI user might take).
#                 installer  re-run the bootstrap installer over the
#                            existing install (download Hermes-Setup.exe
#                            again, AHK clicks Install; lands on HEAD).
#
# HOW THE STAGING WORKS (no MITM proxy, no network fakery):
#   We bare-clone the checkout into <workroot>\serve.git and point every git
#   process at it with url.<file-url>.insteadOf rewrites for the two
#   canonical repo URLs, via a driver-owned gitconfig selected with
#   GIT_CONFIG_GLOBAL. (NOT GIT_CONFIG_COUNT/KEY_n/VALUE_n env config --
#   install.ps1 sets those itself and silently clobbers them.) The
#   installer's `git clone` and `hermes update`'s `git fetch origin`
#   transparently hit OUR bare repo. Its `main` serves OLD for the install
#   phase; the update phase advances it to HEAD -- an update becomes
#   available exactly the way it does for a real user. Installer and
#   updater run byte-for-byte as shipped; everything else (uv, PyPI, npm,
#   the installer's raw.githubusercontent install.ps1 download) uses the
#   real network, same as a user install.
#
# PROOF: screenshots at every renderer step (Playwright), full-desktop
# screenshots around the installer/AHK phases, a rolling desktop capture
# (every 3s) plus a continuous ffmpeg screen recording (recording.mkv) for
# both phases, ahk.log, and the hand-off log. All uploaded as CI artifacts.
#
# DEVIATIONS FROM PRODUCTION (each one deliberate and small):
#   * the git URL redirect itself
#   * serve.git gets uploadpack.allowAnySHA1InWant=true so the installer's
#     baked -Commit pin can be fetched from the redirected clone the same
#     way GitHub's upload-pack allows it.
#   * A dummy provider key is seeded after install so the update leg sees
#     the ready app shell instead of the onboarding overlay (a real
#     updating user has a configured provider).
#   * we don't set .skip_upstream_prompt, but we shim `git` so it returns
#     the real upstream url for 'git remote get-url origin' 
#
# USAGE (local Windows box or CI):
#   powershell -File tests\install\windows-e2e.ps1 -Phase all
#   ... -Phase stage / install / update
#   Phases share state via <workroot>\shas.json, so CI can run them as
#   separate steps for readable logs. -InstallMethod and -Route are
#   orthogonal axes: the install phase dispatches on -InstallMethod, the
#   update phase on -Route, and install writes what update needs (paths,
#   how OLD landed) into the shared state - so any implemented update can
#   follow any implemented install.
# ============================================================================

param(
    [ValidateSet("stage", "install", "update", "all")]
    [string]$Phase = "all",

    # How OLD gets installed, named by the same ids the combination
    # generator (scripts/sandbox/generate-e2e-matrix.mjs) declares.
    [ValidateSet("desktop-installer@latest", "installer-script", "installer-script+desktop")]
    [string]$InstallMethod = "desktop-installer@latest",

    # Update method to exercise in the update phase, same id namespace.
    # open-app-update (from a desktop-installer install) and hermes-update /
    # installer-script / installer-script+desktop (from script installs) are
    # implemented; the rest are declared arms so the surface is stable when
    # they land.
    [ValidateSet("open-app-update", "hermes-desktop-app-update", "hermes-update", "desktop-installer@latest", "installer-script", "installer-script+desktop")]
    [string]$Route = "open-app-update",

    # The OLD version: the ref served as `main` while the installer runs,
    # i.e. what the user starts on. The published Hermes-Setup.exe carries
    # no commit pin (Pin { commit: None, branch: "main" }) -- it installs
    # whatever `main` points at, so staging OLD means serving it there.
    # Empty or "auto" = newest release tag in the checkout (the "user on
    # the current release" starting point, same philosophy as the linux
    # axis's tag matrix). "auto" exists because `powershell -File` silently
    # swallows an empty-string argument ('Missing an argument for
    # parameter'), so the workflow cannot pass "".
    [string]$InstallRef = "auto",

    # Repo checkout whose HEAD is the update target.
    [string]$RepoRoot = "",

    [string]$WorkRoot = $(if ($env:HERMES_E2E_WORKROOT) { $env:HERMES_E2E_WORKROOT } else { Join-Path $env:TEMP "hermes-desktop-gui-e2e" }),

    [string]$SetupExeUrl = "https://hermes-assets.nousresearch.com/Hermes-Setup.exe",

    # Pinned @playwright/test for the update-gui driver. Installed fresh
    # into a scratch dir every run -- never resolved from the installed
    # tree -- so the driver behaves identically for every OLD ref. Bump
    # deliberately; keep roughly in step with the repo's own lockfile.
    [string]$PlaywrightVersion = "1.58.2"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

$ServeRepo   = Join-Path $WorkRoot "serve.git"
$HermesHome  = Join-Path $WorkRoot "hermes-home"
$InstallDir  = Join-Path $HermesHome "hermes-agent"
$StatePath   = Join-Path $WorkRoot "shas.json"
$ProofRoot   = Join-Path $WorkRoot "proof"
$AhkDir      = Join-Path $WorkRoot "ahk"
$AssetsDir   = Join-Path $PSScriptRoot "e2e-assets"

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

function Invoke-Git([string[]]$GitArgs) {
    # PS 5.1 trap: under $ErrorActionPreference = "Stop", a native command
    # that writes ANYTHING to stderr while merged via 2>&1 throws a
    # NativeCommandError even when it exits 0 (git loves stderr for
    # progress/notices). Relax EAP around the native call only; exit-code
    # checking below is the real error gate.
    #
    # ALWAYS the real git.exe, never the shim we ship.
    # annoying bug where .bat files eat ^ args.
    # if hermes ever adds a git command that calls something with ^ this will break, lol.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $script:RealGitExe @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE): $output"
        }
        return ($output | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Set-GitRedirect {
    # we redirect to our own repo so we can play around with what commit hermes thinks we're on.
    # MECHANISM: a driver-owned global gitconfig selected via
    # GIT_CONFIG_GLOBAL. Do NOT use GIT_CONFIG_COUNT/KEY_n/VALUE_n env
    # config here -- install.ps1 SETS those itself (GIT_CONFIG_COUNT=1,
    # windows.appendAtomically), silently clobbering any redirect we put
    # there. install.ps1's own `git config --global` writes simply land in
    # our file, so its compat settings still apply. Nothing leaks onto the
    # machine: the file lives in the workroot and dies with it.
    $fileUrl = "file:///" + ($ServeRepo -replace "\\", "/")
    $gitCfg = Join-Path $WorkRoot "e2e-gitconfig"
    if (-not (Test-Path -LiteralPath $WorkRoot)) {
        New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
    }
    # first, get the set origin url
    $actualGitUrl = Invoke-Git @("-C", $RepoRoot, "remote", "get-url", "origin")
    # then override it 
    @"
[url "$fileUrl"]
	insteadOf = $actualGitUrl
    insteadOf = $RepoUrlHttps
    insteadOf = $RepoUrlSsh
"@ | Set-Content -LiteralPath $gitCfg -Encoding ASCII
    $env:GIT_CONFIG_GLOBAL = $gitCfg

    # check it worked
    $actualGitUrl = Invoke-Git @("-C", $RepoRoot, "remote", "get-url", "origin")
    Assert-True ($actualGitUrl -eq $fileUrl) "git URL redirect: origin resolves to '$actualGitUrl', expected '$fileUrl'."
    Write-Host "  git URL redirect via GIT_CONFIG_GLOBAL=$gitCfg"
    Write-Host "    $RepoUrlHttps -> $fileUrl"

    # shim git and make 'git remote get-url origin' report the actual HA upstream

    # insteadOf is transparent for transport but `git remote get-url origin` gives you the
    # replacement, so _get_origin_url() sees file://$SERVE_REPO and _is_fork() would return true.
    # we check for the arguments "remote get-url origin" in order in any position
    # to allow for e.g. -c with some config being passed.
    # if we didn't do this, we'd need the  .skip_upstream_prompt file to prevent a hang in headless,"add the
    # official repo as upstream?" prompt would hang a headless run. But we don't anymore :D

    $realGit = (Get-Command git.exe -ErrorAction Stop).Source
    $shimDir = Join-Path $WorkRoot "shim"
    New-Item -ItemType Directory -Path $shimDir -Force | Out-Null
    $shimPath = Join-Path $shimDir "git.bat"
    @"
@echo off
setlocal enabledelayedexpansion
set prev2=
set prev1=
:loop
if "%~1"=="" goto passthrough
if /I "!prev2!"=="remote" if /I "!prev1!"=="get-url" if /I "%~1"=="origin" (
    echo $RepoUrlHttps
    exit /b 0
)
set prev2=!prev1!
set prev1=%~1
shift
goto loop
:passthrough
`"$realGit`" %*
exit /b %ERRORLEVEL%
"@ | Set-Content -LiteralPath $shimPath -Encoding ASCII

    $env:PATH = "$shimDir;$env:PATH"

    # Check it worked THROUGH the shim - deliberately not Invoke-Git, which
    # pins the real git.exe. `git` via PATH here is exactly how the
    # product's callers resolve it. Probe the intercepted verb AND plain
    # passthrough.
    #
    # KNOWN HOLE, accepted: cmd parses the command line before the bat sees
    # %*, and callers only quote args containing whitespace (PowerShell
    # native binding and python's list2cmdline alike) - so a caret arg like
    # rev-parse HEAD^{commit} loses its caret THROUGH ANY .bat, unfixably.
    # A .ps1 shim would dodge cmd but PATHEXT-resolving callers (python -
    # the shim's entire audience) never see .ps1 files, so .bat it stays.
    # The driver's own git plumbing therefore pins git.exe (Invoke-Git),
    # and the product's shimmed flows (fork detection: remote get-url) use
    # no caret revs. If a product path ever sends carets through the shim,
    # the leg fails loudly on a bad-revision error naming the mangled arg.
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $observedGitUrl = (& git -C $RepoRoot remote get-url origin 2>&1 | Out-String).Trim()
    $passthroughProbe = (& git -C $RepoRoot rev-parse HEAD 2>&1 | Out-String).Trim()
    $passthroughExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Assert-True ($observedGitUrl -eq $RepoUrlHttps) "git remote get-url shim: origin resolves to '$observedGitUrl', expected '$RepoUrlHttps'"
    Assert-True ($passthroughExit -eq 0 -and $passthroughProbe -match '^[0-9a-f]{40}$') "shim passthrough works: rev-parse HEAD -> '$passthroughProbe'"
    Write-Host "  git remote get-url shim: $shimPath -> $realGit"
    Write-Host "    'remote get-url origin' now reports $RepoUrlHttps"
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
    foreach ($c in @(
        (Join-Path $InstallDir "apps\desktop\release\win-unpacked\Hermes.exe"),
        (Join-Path $InstallDir "apps\desktop\release\win-arm64-unpacked\Hermes.exe")
    )) {
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
# Script-install arm: the irm | iex one-liner, headless (the install.ps1
# shipped AT the ref under test, run with flags probed from that ref's own
# script text - older releases reject parameters added later).
# ----------------------------------------------------------------------------
# shellcheck source=../e2e-assets/ts-prefix.ps1
. (Join-Path $PSScriptRoot "e2e-assets\ts-prefix.ps1")

function Write-LogGroup([string]$Title, [string]$LogPath) {
    Write-Host "::group::$Title"
    if (Test-Path -LiteralPath $LogPath) { Get-Content -LiteralPath $LogPath | Write-Host }
    Write-Host "::endgroup::"
}

function Invoke-RefInstaller {
    param([string]$Ref, [string]$Label, [switch]$IncludeDesktop)
    $script = Join-Path $WorkRoot "install-$Label.ps1"
    (Invoke-Git @("-C", $RepoRoot, "show", "$Ref`:scripts/install.ps1")) -join "`n" |
        Set-Content -LiteralPath $script -Encoding UTF8
    $flags = @("-SkipSetup", "-HermesHome", $HermesHome, "-InstallDir", $InstallDir)
    $text = Get-Content -LiteralPath $script -Raw
    if ($text -match '\$NonInteractive') { $flags += "-NonInteractive" }
    if ($IncludeDesktop) {
        # The desktop stage is the point of this leg: a ref without the
        # parameter is a hard failure, not a silent plain install.
        if ($text -notmatch '\$IncludeDesktop') {
            throw "E2E ASSERTION FAILED: ref $Ref does not support -IncludeDesktop; this leg cannot mean what it claims"
        }
        $flags += "-IncludeDesktop"
    }
    New-Item -ItemType Directory -Path (Join-Path $WorkRoot "logs") -Force | Out-Null
    $log = Join-Path $WorkRoot "logs\install-$Label.log"
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script @flags 2>&1 | Add-TsPrefix | Out-File -Encoding UTF8 $log
    $installExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    Write-LogGroup "install.ps1 ($Label) transcript" $log
    Assert-True ($installExit -eq 0) "install.ps1 ($Label) exited 0"
}

function Assert-DesktopArtifact([string]$Label) {
    Assert-True ($null -ne (Get-DesktopExe)) "$Label -- desktop app built by installer under apps\desktop\release"
}

function Invoke-HermesUpdate {
    # The venv updater. --yes reaches the update subcommand only in later
    # releases; ask the installed binary, never parse its source.
    $hermesExe = Join-Path $InstallDir "venv\Scripts\hermes.exe"
    $updateArgs = @("update")
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $helpText = & $hermesExe update --help 2>&1 | Out-String
    if ($helpText -match '--yes') { $updateArgs += "--yes" }
    New-Item -ItemType Directory -Path (Join-Path $WorkRoot "logs") -Force | Out-Null
    $log = Join-Path $WorkRoot "logs\update.log"
    Push-Location $InstallDir
    try {
        & $hermesExe @updateArgs 2>&1 | Add-TsPrefix | Out-File -Encoding UTF8 $log
        $updateExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEap
    }
    Write-LogGroup "hermes update transcript" $log
    Assert-True ($updateExit -eq 0) "hermes update exited 0"
}

function Invoke-HermesDesktopAppUpdate([string]$TargetSha) {
    # The hermes-desktop launch surface: `hermes desktop` runs its whole
    # real pipeline; the driver intercepts the product's final spawn
    # (argv/cwd/env captured by e2e-assets/launch-capture/sitecustomize.py)
    # and re-executes it under Playwright, which clicks Update now.
    $hermesExe = Join-Path $InstallDir "venv\Scripts\hermes.exe"
    $spec = Join-Path $WorkRoot "launch-spec.json"
    New-Item -ItemType Directory -Path (Join-Path $WorkRoot "logs") -Force | Out-Null
    $log = Join-Path $WorkRoot "logs\desktop-launch-capture.log"

    $capDir = Join-Path $AssetsDir "launch-capture"
    $prevPy = $env:PYTHONPATH
    $prevCap = $env:HERMES_E2E_CAPTURE_LAUNCH
    $env:PYTHONPATH = if ($prevPy) { "$capDir;$prevPy" } else { $capDir }
    $env:HERMES_E2E_CAPTURE_LAUNCH = $spec
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $InstallDir
    try {
        & $hermesExe desktop 2>&1 | Add-TsPrefix | Out-File -Encoding UTF8 $log
        $capExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEap
        $env:PYTHONPATH = $prevPy
        $env:HERMES_E2E_CAPTURE_LAUNCH = $prevCap
    }
    Write-LogGroup "hermes desktop (launch capture) transcript" $log
    Assert-True ($capExit -eq 0) "hermes desktop exited 0 during launch capture"
    Assert-True (Test-Path -LiteralPath "$spec.captured") "a launch was actually captured (exit 0 without a launch must not pass)"

    $node = Get-ManagedNode
    $driverDir = Join-Path $WorkRoot "pw-driver"
    New-Item -ItemType Directory -Path $driverDir -Force | Out-Null
    $npmCli = Join-Path (Split-Path -Parent $node) "node_modules\npm\bin\npm-cli.js"
    Assert-True (Test-Path -LiteralPath $npmCli) "managed npm exists beside the managed node"
    Push-Location $driverDir
    try {
        & $node $npmCli install --no-save --no-audit --no-fund "@playwright/test@$PlaywrightVersion" 2>&1 |
            Select-Object -Last 5 | ForEach-Object { Write-Host "  npm| $_" }
        $npmExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Assert-True ($npmExit -eq 0) "npm install @playwright/test@$PlaywrightVersion into the driver dir"

    Copy-Item (Join-Path $AssetsDir "launch-from-spec.mjs") (Join-Path $driverDir "launch-from-spec.mjs") -Force
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $driverDir
    try {
        & $node "launch-from-spec.mjs" --spec $spec `
            --result (Join-Path $HermesHome ".hermes-update-result.json") `
            --expect-sha $TargetSha --repo-dir $InstallDir 2>&1 |
            ForEach-Object { Write-Host "  pw| $_" }
        $driveExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEap
    }
    Assert-True ($driveExit -eq 0) "app driven via captured hermes desktop spec; update completed"
}

function Save-DesktopScreenshot([string]$OutFile) {
    # Single full-desktop screenshot (primary screen).
    try {
        Add-Type -AssemblyName System.Windows.Forms, System.Drawing
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
        $gfx = [System.Drawing.Graphics]::FromImage($bmp)
        $gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bmp.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
        $gfx.Dispose(); $bmp.Dispose()
        Write-Host "  desktop screenshot: $OutFile"
    } catch {
        Write-Host "  WARNING: desktop screenshot failed: $($_.Exception.Message)"
    }
}

function Start-DesktopRecorder([string]$OutDir) {
    # Rolling desktop capture: one PNG every 3s from a detached PowerShell,
    # capped at 800 frames (~40 min). Proof that survives any step failure.
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $script = Join-Path $WorkRoot "recorder.ps1"
    @'
param([string]$OutDir)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
for ($i = 0; $i -lt 800; $i++) {
    if (Test-Path (Join-Path $OutDir "STOP")) { break }
    try {
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bmp = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
        $gfx = [System.Drawing.Graphics]::FromImage($bmp)
        $gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bmp.Save((Join-Path $OutDir ("frame-{0:D4}.png" -f $i)), [System.Drawing.Imaging.ImageFormat]::Png)
        $gfx.Dispose(); $bmp.Dispose()
    } catch {}
    Start-Sleep -Seconds 3
}
'@ | Set-Content -LiteralPath $script -Encoding UTF8
    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "-OutDir", $OutDir `
        -WindowStyle Hidden -PassThru
    Write-Host "  desktop recorder started (pid $($proc.Id)) -> $OutDir"
    return $proc
}

function Stop-DesktopRecorder($proc, [string]$OutDir) {
    try { Set-Content -LiteralPath (Join-Path $OutDir "STOP") -Value "stop" } catch {}
    if ($proc) {
        try { $proc.WaitForExit(8000) | Out-Null } catch {}
        try { if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force } } catch {}
    }
}

function Stop-HermesAppProcesses([string]$Label) {
    # Close the desktop app the blunt way between phases (a user quitting).
    # Only Hermes.exe (Electron) -- never hermes.exe (the venv CLI shim).
    $procs = @(Get-Process -Name "Hermes" -ErrorAction SilentlyContinue)
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
    if ($procs.Count -gt 0) {
        Write-Host "  [$Label] stopped $($procs.Count) Hermes.exe process(es)"
        Start-Sleep -Seconds 3
    }
}

function Get-ManagedNode {
    # `hermes update`/desktop builds use the Hermes-managed Node; use the same
    # one to run the Playwright driver so no system Node is required.
    $candidates = @(
        (Join-Path $HermesHome "node\node.exe"),
        (Join-Path $HermesHome "bin\node\node.exe"),
        (Join-Path $InstallDir "node\node.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    $fromPath = Get-Command node -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    throw "No node.exe found (managed or on PATH)"
}

# ----------------------------------------------------------------------------
# Phase: stage -- serve.git with `main` at OLD (advanced to HEAD by update-gui)
# ----------------------------------------------------------------------------
function Invoke-PhaseStage {
    Write-Step "STAGE: bare serve repo, main -> OLD (install base)"

    if (Test-Path -LiteralPath $WorkRoot) {
        Remove-Item -LiteralPath $WorkRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
    # The purge above deleted the redirect gitconfig; re-arm it so the
    # bare-clone below (and everything after) sees the redirect file.
    Set-GitRedirect

    $current = Invoke-Git @("-C", $RepoRoot, "rev-parse", "HEAD")
    Write-Host "  HEAD (update target): $current"

    # OLD: explicit -InstallRef, or the newest release tag -- the version a
    # user who installed on release day is on.
    $oldRef = $InstallRef
    if (-not $oldRef -or $oldRef -eq "auto") {
        # Parens matter: without them PowerShell binds -split as an
        # argument to Invoke-Git instead of an operator on its result.
        $tagList = Invoke-Git @("-C", $RepoRoot, "tag", "--list", "v*", "--sort=-creatordate")
        $oldRef = ($tagList -split "\r?\n" | Select-Object -First 1)
        if (-not $oldRef) { throw "no v* release tags in the checkout and no -InstallRef given -- cannot pick an OLD version" }
    }
    $old = Invoke-Git @("-C", $RepoRoot, "rev-parse", "$oldRef^{commit}")
    Write-Host "  OLD  ($oldRef): $old"
    Assert-True ($old -ne $current) "OLD differs from HEAD (an update is genuinely available)"

    # Bare-clone the checkout: this is the repo the installer and updater
    # actually talk to. Local-path clone hardlinks objects, so it's fast
    # even for full history. The published installer carries NO commit pin
    # (Pin { commit: None, branch: main }) -- it installs whatever `main`
    # serves, so staging OLD means parking `main` there; the update phase
    # advances it to HEAD.
    Invoke-Git @("clone", "--bare", "--quiet", $RepoRoot, $ServeRepo) | Out-Null
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $old) | Out-Null
    Invoke-Git @("-C", $ServeRepo, "symbolic-ref", "HEAD", "refs/heads/main") | Out-Null

    # Belt-and-braces: SOME installer builds do bake a -Commit pin. A pinned
    # sha is in serve.git's history but not at a ref tip, so the redirected
    # fetch needs any-SHA1 upload-pack permission (GitHub grants the
    # equivalent for fetch of reachable commits).
    Invoke-Git @("-C", $ServeRepo, "config", "uploadpack.allowAnySHA1InWant", "true") | Out-Null
    Write-Host "  serve.git: uploadpack.allowAnySHA1InWant=true (installer commit pin, if any)"

    @{ old = $old; old_ref = $oldRef; current = $current } |
        ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    Write-Host "  state written: $StatePath"
    New-Item -ItemType Directory -Path $ProofRoot -Force | Out-Null
}

# ----------------------------------------------------------------------------
# Phase: install-gui -- website Hermes-Setup.exe, headed, AHK-driven
# ----------------------------------------------------------------------------
function Invoke-PhaseInstallGui {
    param(
        # "install" (first run, must land on OLD) or "update" (re-run over an
        # existing install after serve.git advanced, must land on the target).
        [string]$Mode = "install",
        [string]$ExpectedSha = "",
        [string]$ExpectedLabel = ""
    )
    $state = Read-State
    if ($Mode -eq "install") {
        $ExpectedSha = $state.old
        $ExpectedLabel = "OLD ($($state.old_ref))"
    }
    Write-Step "$($Mode.ToUpper()) (GUI): Hermes-Setup.exe from the website, headed, AHK clicks"
    $proof = Join-Path $ProofRoot $(if ($Mode -eq "install") { "install-gui" } else { "update-gui-installer" })
    New-Item -ItemType Directory -Path $proof -Force | Out-Null

    # The production installer, from the website. This is the binary users
    # double-click, run EXACTLY as shipped: its own pinned install.ps1, its
    # own baked BUILD_PIN_COMMIT. The only environmental difference is the
    # git URL redirect to serve.git.
    $setupExe = Join-Path $WorkRoot "Hermes-Setup.exe"
    if (-not (Test-Path -LiteralPath $setupExe)) {
        Write-Host "  downloading $SetupExeUrl"
        Invoke-WebRequest -Uri $SetupExeUrl -OutFile $setupExe
    }
    Assert-True ((Get-Item $setupExe).Length -gt 1MB) "Hermes-Setup.exe downloaded ($([math]::Round((Get-Item $setupExe).Length / 1MB, 1)) MB)"

    # AutoHotkey v2, portable zip (no installer, no winget flakes).
    $ahkExe = Join-Path $AhkDir "AutoHotkey64.exe"
    if (-not (Test-Path -LiteralPath $ahkExe)) {
        $zip = Join-Path $WorkRoot "ahk.zip"
        Invoke-WebRequest -Uri "https://github.com/AutoHotkey/AutoHotkey/releases/download/v2.0.19/AutoHotkey_2.0.19.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $AhkDir -Force
    }
    Assert-True (Test-Path -LiteralPath $ahkExe) "AutoHotkey64.exe available"

    # AHK script + button templates side by side (ImageSearch resolves
    # relative to the script dir).
    Copy-Item -Path (Join-Path $AssetsDir "install-and-launch.ahk"), (Join-Path $AssetsDir "install-button.png"), (Join-Path $AssetsDir "launch-button.png") -Destination $AhkDir -Force

    $env:HERMES_HOME = $HermesHome
    # As shipped: NO dev-root override, no pin override. Ensure a stray
    # local dev checkout can't hijack resolution.
    Remove-Item Env:HERMES_SETUP_DEV_REPO_ROOT -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null

    $recorder = Start-DesktopRecorder (Join-Path $proof "desktop-frames")
    $ahkLog = Join-Path $proof "ahk.log"
    try {
        Save-DesktopScreenshot (Join-Path $proof "00-before-installer.png")

        # Launch the REAL installer, headed -- exactly a double-click.
        $installer = Start-Process -FilePath $setupExe -PassThru
        Write-Host "  Hermes-Setup.exe launched (pid $($installer.Id))"

        # Drive it: Install click -> wait -> Launch click -> Hermes.exe window.
        # Arg 3 lets the AHK script use the installer's own log as the
        # install-finished fallback signal.
        $ahk = Start-Process -FilePath $ahkExe `
            -ArgumentList (Join-Path $AhkDir "install-and-launch.ahk"), $ahkLog, "Hermes-Setup.exe", (Join-Path $HermesHome "logs\bootstrap-installer.log") `
            -PassThru
        # Install on a cold runner takes a while; the AHK script's own inner
        # timeout (45 min on the Launch wait) is the effective budget.
        if (-not $ahk.WaitForExit(50 * 60 * 1000)) {
            Stop-Process -Id $ahk.Id -Force -ErrorAction SilentlyContinue
            throw "AutoHotkey driver did not finish within 50 minutes"
        }
        if (Test-Path -LiteralPath $ahkLog) {
            Get-Content -LiteralPath $ahkLog | ForEach-Object { Write-Host "  ahk| $_" }
        }
        Assert-True ($ahk.ExitCode -eq 0) "AutoHotkey driver exited 0 (Install clicked, Launch clicked, app window seen)"

        Save-DesktopScreenshot (Join-Path $proof "01-app-launched.png")

        # The Launch hand-off under test: the app the installer spawned must
        # actually be running.
        Assert-True ($null -ne (Get-Process -Name "Hermes" -ErrorAction SilentlyContinue)) "Hermes.exe process is running (installer Launch hand-off worked)"

        # Installer should have exited after Launch.
        if (-not $installer.HasExited) {
            Start-Sleep -Seconds 10
        }
        Assert-True $installer.HasExited "Hermes-Setup.exe exited after Launch"
    }
    finally {
        Stop-DesktopRecorder $recorder (Join-Path $proof "desktop-frames")
        # Surface the installer's own log win or lose, full and folded.
        $bootLog = Join-Path $HermesHome "logs\bootstrap-installer.log"
        if (Test-Path -LiteralPath $bootLog) {
            Write-Host "::group::bootstrap-installer.log"
            Get-Content -LiteralPath $bootLog | Write-Host
            Write-Host "::endgroup::"
            Copy-Item $bootLog $proof -Force -ErrorAction SilentlyContinue
        }
    }

    # Close the freshly launched app (user quits after first look).
    Stop-HermesAppProcesses "post-install"

    # The installer cloned/updated from serve.git's `main`; the phase's
    # expected sha says where that must land (install: OLD; update: HEAD).
    $installedSha = Get-InstalledHead
    Write-Host "  installer landed on: $installedSha (expected $ExpectedLabel = $ExpectedSha)"
    Assert-True ($installedSha -eq $ExpectedSha) "installed checkout is at $ExpectedLabel"
    if ($Mode -eq "install") {
        Assert-True ($installedSha -ne $state.current) "installed checkout differs from HEAD (an update is genuinely available)"
    }
    Test-HermesRuns "post-$Mode-gui"
    Assert-True ($null -ne (Get-DesktopExe)) "packaged Desktop Hermes.exe exists"

    # Seed a provider so the update leg meets the ready app shell, not the
    # onboarding overlay (an updating user has a configured provider).
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path -LiteralPath $envFile) -or -not ((Get-Content $envFile -Raw -ErrorAction SilentlyContinue) -match "OPENROUTER_API_KEY")) {
        Add-Content -LiteralPath $envFile -Value "OPENROUTER_API_KEY=sk-or-...-key"
    }
    Write-Host "  seeded placeholder provider key for the update leg"
}

# ----------------------------------------------------------------------------
# Phase: update-gui -- OLD -> HEAD through the selected route
# ----------------------------------------------------------------------------
function Invoke-GuiUpdateDesktopRoute([string]$TargetSha) {
    Write-Step "UPDATE (GUI, route=desktop): advance served main -> $TargetSha, click Update now"
    $proof = Join-Path $ProofRoot "update-gui"
    New-Item -ItemType Directory -Path $proof -Force | Out-Null

    $env:HERMES_HOME = $HermesHome

    # The update becomes available the way it does for a real user: the
    # remote's main moves forward. (Install ran against main = OLD.)
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $TargetSha) | Out-Null
    Write-Host "  serve.git main advanced to $TargetSha"

    $desktopExe = Get-DesktopExe
    Assert-True ($null -ne $desktopExe) "packaged Hermes.exe present before update"

    $resultPath = Join-Path $HermesHome ".hermes-update-result.json"
    $markerPath = Join-Path $HermesHome ".hermes-update-in-progress"
    Remove-Item -LiteralPath $resultPath -Force -ErrorAction SilentlyContinue

    $node = Get-ManagedNode
    # The Playwright driver gets its OWN pinned @playwright/test in a
    # scratch dir -- NEVER the installed tree's copy. The driver talks to
    # the app over Playwright's inspection pipe, so its Playwright version
    # is independent of the app under test; installing it ourselves makes
    # the leg identical for every OLD ref (older releases predate the
    # dependency entirely, and hoisting moves it around in newer ones).
    $driverDir = Join-Path $WorkRoot "pw-driver"
    New-Item -ItemType Directory -Path $driverDir -Force | Out-Null
    $npmCli = Join-Path (Split-Path -Parent $node) "node_modules\npm\bin\npm-cli.js"
    Assert-True (Test-Path -LiteralPath $npmCli) "managed npm exists beside the managed node"
    Push-Location $driverDir
    try {
        & $node $npmCli install --no-save --no-audit --no-fund "@playwright/test@$PlaywrightVersion" 2>&1 |
            Select-Object -Last 5 | ForEach-Object { Write-Host "  npm| $_" }
        $npmExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Assert-True ($npmExit -eq 0) "npm install @playwright/test@$PlaywrightVersion into the driver dir"

    $recorder = Start-DesktopRecorder (Join-Path $proof "desktop-frames")
    try {
        # Launch the installed app and click through Settings -> About ->
        # Update now. Exit 0 = the app quit for the updater hand-off.
        # Copy the driver INTO $driverDir first: Node resolves
        # require('@playwright/test') from the SCRIPT's own directory upward,
        # so running it from the CI checkout would resolve the wrong (or no)
        # node_modules.
        $driver = Join-Path $driverDir "e2e-drive-update.cjs"
        Copy-Item (Join-Path $AssetsDir "drive-update.cjs") $driver -Force
        Push-Location $driverDir
        try {
            & $node $driver $desktopExe $proof 2>&1 |
                ForEach-Object { Write-Host "  $_" }
            $driveExit = $LASTEXITCODE
        } finally {
            Pop-Location
            Remove-Item -LiteralPath $driver -Force -ErrorAction SilentlyContinue
        }
        Assert-True ($driveExit -eq 0) "GUI driver clicked Update now and the app quit for hand-off"

        # The detached updater (spawned by the app, NOT by us) now runs
        # `hermes update` + desktop rebuild + relaunch. Which updater depends
        # on the installed checkout, and BOTH are production paths:
        #   * checkouts shipping scripts/desktop-update.ps1 -> that script,
        #     which writes .hermes-update-result.json on every exit;
        #   * older checkouts -> the staged hermes-setup.exe --update flow,
        #     which does NOT write the result JSON.
        # So: poll for COMPLETION = (result JSON) OR (checkout reached the
        # target sha AND the marker is gone). The sha/marker/hermes/relaunch
        # asserts below are the hard gate either way; the JSON is asserted
        # only when the script path produced it.
        #
        # The update pulls a large diff AND does a full Electron desktop
        # rebuild (vite + electron-builder) plus a uv sync; a WORKING updater
        # finishes well under 35 minutes on these runners (slowest observed
        # leg anywhere in the matrix: 29m end to end). A wedged updater never
        # finishes at any bound, so a longer wait only delays the report.
        # The desktop-build output goes to logs/update.log (not the streamed
        # handoff log), so we tail update.log here to show progress.
        Write-Host "  waiting for the detached updater to finish (up to 35 min) ..."
        $updateLog = Join-Path $HermesHome "logs\update.log"
        $updateLogPos = 0
        $deadline = (Get-Date).AddMinutes(35)
        while ((Get-Date) -lt $deadline) {
            if (Test-Path -LiteralPath $resultPath) { break }
            $head = ""
            try { $head = Get-InstalledHead } catch {}
            if ($head -eq $TargetSha -and -not (Test-Path -LiteralPath $markerPath)) { break }
            # Tail any new update.log lines so the desktop-rebuild phase is
            # visible in the CI step output.
            if (Test-Path -LiteralPath $updateLog) {
                try {
                    $lines = Get-Content -LiteralPath $updateLog -ErrorAction SilentlyContinue
                    if ($lines.Count -gt $updateLogPos) {
                        $lines[$updateLogPos..($lines.Count - 1)] | ForEach-Object { Write-Host "    update.log| $_" }
                        $updateLogPos = $lines.Count
                    }
                } catch {}
            }
            Start-Sleep -Seconds 20
        }
        if (Test-Path -LiteralPath $resultPath) {
            $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
            Write-Host "  updater result: ok=$($result.ok) code=$($result.exit_code) msg=$($result.message)"
            Assert-True ([bool]$result.ok) "updater result ok=true"
        } else {
            Write-Host "  (no result JSON -- staged-binary updater path; relying on sha/marker/relaunch asserts)"
        }

        # Marker may briefly outlive the result write; allow it a moment.
        $mDeadline = (Get-Date).AddMinutes(2)
        while ((Get-Date) -lt $mDeadline -and (Test-Path -LiteralPath $markerPath)) { Start-Sleep -Seconds 5 }
        Assert-True (-not (Test-Path -LiteralPath $markerPath)) "update marker cleaned up"

        Assert-True ((Get-InstalledHead) -eq $TargetSha) "checkout landed on target commit"
        Test-HermesRuns "post-update"
        Assert-True ($null -ne (Get-DesktopExe)) "Hermes.exe still present after update"

        # The production hand-off relaunches the desktop (RelaunchExe).
        # A relaunched window is the user-visible proof the update loop closed.
        Write-Host "  waiting for the relaunched Hermes.exe ..."
        $rDeadline = (Get-Date).AddMinutes(5)
        $relaunched = $null
        while ((Get-Date) -lt $rDeadline) {
            $relaunched = Get-Process -Name "Hermes" -ErrorAction SilentlyContinue
            if ($relaunched) { break }
            Start-Sleep -Seconds 5
        }
        Assert-True ($null -ne $relaunched) "updater relaunched the desktop app"
        Start-Sleep -Seconds 12   # let the window paint for the screenshot
        # Foreground the relaunched Hermes window so the proof screenshot
        # captures IT, not whatever else is on top (the full-desktop grab is
        # otherwise at the mercy of z-order -- an earlier run caught VS Code).
        try {
            $mainProc = Get-Process -Name "Hermes" -ErrorAction SilentlyContinue |
                Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
            if ($mainProc) {
                Add-Type -Namespace HdE2E -Name Win -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool SetForegroundWindow(System.IntPtr h);
[System.Runtime.InteropServices.DllImport("user32.dll")] public static extern bool ShowWindow(System.IntPtr h, int n);
'@ -ErrorAction SilentlyContinue
                [HdE2E.Win]::ShowWindow($mainProc.MainWindowHandle, 9) | Out-Null   # SW_RESTORE
                [HdE2E.Win]::SetForegroundWindow($mainProc.MainWindowHandle) | Out-Null
                Start-Sleep -Seconds 2
            }
        } catch {}
        Save-DesktopScreenshot (Join-Path $proof "99-relaunched-desktop.png")
    }
    finally {
        Stop-DesktopRecorder $recorder (Join-Path $proof "desktop-frames")
        $handoffLog = Join-Path $HermesHome "logs\desktop-update-handoff.log"
        if (Test-Path -LiteralPath $handoffLog) {
            Write-Host "::group::desktop-update-handoff.log"
            Get-Content -LiteralPath $handoffLog | Write-Host
            Write-Host "::endgroup::"
            Copy-Item $handoffLog (Join-Path $proof "desktop-update-handoff.log") -Force -ErrorAction SilentlyContinue
        }
        # Quit the relaunched app so job teardown is clean.
        Stop-HermesAppProcesses "post-update"
    }
}

function Invoke-PhaseInstall {
    # Dispatch on the install axis. Each arm ends with the same contract:
    # checkout at OLD, hermes runs, and state carries how OLD landed so any
    # update arm can follow any install arm.
    $state = Read-State
    # Isolated install target for every arm; serve.git's file:// origin
    # looks like a fork to the updater, whose "add the official repo as
    # upstream?" prompt would hang a headless run - the marker is the
    # product's own suppression mechanism.
    $env:HERMES_HOME = $HermesHome
    New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null
    switch ($InstallMethod) {
        "desktop-installer@latest" {
            Invoke-PhaseInstallGui
        }
        "installer-script" {
            Write-Step "INSTALL (script): OLD's own install.ps1, headless"
            Invoke-RefInstaller $state.old "old"
            Assert-True ((Get-InstalledHead) -eq $state.old) "installed checkout is at OLD"
            Test-HermesRuns "post-install-script"
        }
        "installer-script+desktop" {
            Write-Step "INSTALL (script+desktop): OLD's own install.ps1 -IncludeDesktop, headless"
            Invoke-RefInstaller $state.old "old" -IncludeDesktop
            Assert-True ((Get-InstalledHead) -eq $state.old) "installed checkout is at OLD"
            Test-HermesRuns "post-install-script-desktop"
            Assert-DesktopArtifact "OLD"
        }
    }
}

function Invoke-PhaseUpdate {
    $state = Read-State
    $env:HERMES_HOME = $HermesHome

    # The update becomes available the way it does for a real user: the
    # remote's main moves forward. The GUI route re-advances harmlessly
    # (same sha); script routes need it here because only the GUI arm's
    # helper used to own this step.
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $state.current) | Out-Null
    Write-Host "  serve.git main advanced to $($state.current)"

    switch ($Route) {
        "open-app-update" {
            # Meaningful only where an OS entry point exists - install.ps1
            # -IncludeDesktop registers shortcuts too, so both desktop-
            # bearing installs qualify; the workflow gate enforces which
            # pairs are dispatched.
            Invoke-GuiUpdateDesktopRoute $state.current
        }
        "hermes-desktop-app-update" {
            Invoke-HermesDesktopAppUpdate $state.current
        }
        "hermes-update" {
            Invoke-HermesUpdate
        }
        "installer-script" {
            # A user re-running the one-liner today gets the CURRENT script.
            Invoke-RefInstaller $state.current "head"
        }
        "installer-script+desktop" {
            Invoke-RefInstaller $state.current "head" -IncludeDesktop
            Assert-DesktopArtifact "HEAD"
        }
        "desktop-installer@latest" {
            # A user re-downloading Hermes-Setup.exe and clicking Install over
            # the existing install (the GUI twin of re-running the one-liner).
            # Windows has no already-installed fast path, so the full installer
            # UI shows and the same AHK drive applies; install.ps1's repository
            # stage fetches into the existing checkout, now aimed at HEAD.
            # Rotate the bootstrap log first: it appends across runs, and the
            # AHK's "bootstrap complete" fallback must not match the install
            # phase's completion line.
            $bootLog = Join-Path $HermesHome "logs\bootstrap-installer.log"
            if (Test-Path -LiteralPath $bootLog) {
                Move-Item -LiteralPath $bootLog -Destination "$bootLog.install-phase" -Force
            }
            Invoke-PhaseInstallGui -Mode "update" -ExpectedSha $state.current -ExpectedLabel "HEAD"
            Assert-DesktopArtifact "HEAD"
        }
    }

    Assert-True ((Get-InstalledHead) -eq $state.current) "checkout landed on HEAD"
    Test-HermesRuns "post-update"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
Write-Host "Windows install/update E2E driver (real user flows)"
Write-Host "  phase:    $Phase"
Write-Host "  install:  $InstallMethod"
Write-Host "  route:    $Route"
Write-Host "  repo:     $RepoRoot"
Write-Host "  workroot: $WorkRoot"

$script:RealGitExe = (Get-Command git.exe -ErrorAction Stop).Source

Set-GitRedirect

switch ($Phase) {
    "stage"   { Invoke-PhaseStage }
    "install" { Invoke-PhaseInstall }
    "update"  { Invoke-PhaseUpdate }
    "all" {
        Invoke-PhaseStage
        Invoke-PhaseInstall
        Invoke-PhaseUpdate
    }
}

Write-Host ""
Write-Host "Phase '$Phase' completed successfully."
