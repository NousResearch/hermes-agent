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
#                 update     run `hermes update` from the installed command
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
#   * The git shim reports the official URL. Detached updaters can resolve a
#     different git.exe, so the test home also records that upstream setup was
#     declined. The file:// transport must not prompt to add a second remote.
#
# USAGE (local Windows box or CI):
#   powershell -File tests\install\windows-e2e.ps1 -Phase all
#   ... -Phase stage / install / update / verify-stamp
#   Phases share state via <workroot>\shas.json, so CI can run them as
#   separate steps for readable logs. -InstallMethod and -Route are
#   orthogonal axes: the install phase dispatches on -InstallMethod, the
#   update phase on -Route, and install writes what update needs (paths,
#   how OLD landed) into the shared state - so any implemented update can
#   follow any implemented install.
# ============================================================================

param(
    [ValidateSet("stage", "install", "update", "verify-stamp", "all")]
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
    # Update target ref (default HEAD). A stable-to-stable leg passes the
    # next release tag here; only label the leg stable-to-stable when BOTH
    # refs are release tags. NEXT mints a synthetic child of -InstallRef
    # (the HEAD -> NEXT leg: install HEAD, update with HEAD's own updater).
    [string]$UpdateRef = "HEAD",

    # Repo checkout whose HEAD is the update target.
    [string]$RepoRoot = "",

    [string]$WorkRoot = $(if ($env:HERMES_E2E_WORKROOT) { $env:HERMES_E2E_WORKROOT } else { Join-Path $env:TEMP "hermes-desktop-gui-e2e" }),

    [string]$SetupExeUrl = "https://hermes-assets.nousresearch.com/Hermes-Setup.exe",

    # Driver dependencies come from the current checkout lockfile.
    [string]$DriverNode = $env:HERMES_E2E_NODE
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# Match an interactive Unicode console when Python output is piped into the
# UTF-8 transcript. Old releases otherwise select cp1252 and crash on banners.
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding

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
if (-not $DriverNode) { $DriverNode = (Get-Command node.exe -ErrorAction Stop).Source }
$env:HERMES_E2E_NODE = $DriverNode
$env:HERMES_DESKTOP_USER_DATA_DIR = Join-Path $WorkRoot 'electron-user-data'
$script:ChatMock = $null
$script:ChatFailure = $false
. (Join-Path $AssetsDir 'desktop-smoke-windows.ps1')

function Start-JourneyChat {
    if (-not $script:ChatMock) {
        $script:ChatFailure = $true
        $script:ChatMock = Start-DesktopJourneyMock $DriverNode $AssetsDir $WorkRoot $HermesHome $ProofRoot
        $script:ChatFailure = $false
    }
}

function Invoke-DesktopCheckpoint([string]$ChatPhase, [string]$Commit, [string]$Method) {
    $script:ChatFailure = $true
    $prevEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $DriverNode (Join-Path $AssetsDir 'source-desktop-smoke.mjs') `
            --root $InstallDir --home $HermesHome --user-data $env:HERMES_DESKTOP_USER_DATA_DIR `
            --out $ProofRoot --phase $ChatPhase --expect-commit $Commit --desktop $script:ExpectedDesktop --method $Method
        $chatExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $prevEap }
    if ($chatExit -ne 0) { throw "Mandatory desktop chat $ChatPhase failed (exit $chatExit)" }
    $script:ChatFailure = $false
}

function Confirm-OldChat([string]$Out) {
    $receipt = Get-Content -LiteralPath (Join-Path $Out 'desktop-chat-old.json') -Raw | ConvertFrom-Json
    if ($receipt.status -ne 'passed') { throw 'Mandatory OLD update-window chat failed' }
    $script:ChatFailure = $false
}

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

    # The dispatch-time capture, not PATH: a fresh-machine leg has already
    # stripped git from PATH by the time stage re-arms the redirect.
    $realGit = $script:RealGitExe
        # Export the real git so later checks can observe the TRANSPORT url. Once
        # the shim below is on PATH, `git` reports the official origin for
        # `remote get-url origin` (so fork detection sees it); any check that must
        # see the file:// redirect instead has to bypass the shim via this path.
        $env:HERMES_E2E_REAL_GIT = $realGit

    if ($script:FreshMachine) {
        # A fresh Windows box has no git. install.ps1's Get-PinnedGit returns
        # ANY git on PATH (the dev shortcut), so the runner's git -- or the
        # shim below -- would skip pinned-git staging entirely. Take every
        # git.exe directory off PATH and install no shim: the product must
        # provision its own. The shim's one job (fork detection seeing the
        # official origin) is covered by .skip_upstream_prompt, same as
        # routes whose detached updater bypasses the shim.
        $kept = @($env:PATH -split ';' | Where-Object { $_ -and -not (Test-Path -LiteralPath (Join-Path $_ 'git.exe')) })
        $env:PATH = $kept -join ';'
        Assert-True (-not (Get-Command git -ErrorAction SilentlyContinue)) "fresh machine: no git resolvable on PATH"
        Write-Host "  fresh machine: git removed from PATH, no remote get-url shim"
        return
    }
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

# Install-side state snapshot, taken BEFORE Test-HermesRuns can throw: on
# app-update legs the updater runs detached and its transcript lands in the
# product logs and hand-off files, not in this driver. Copy those plus the
# venv entry-point dir while the install is still there to inspect, so a
# failed post-update assertion leaves its evidence in the proof tree.
function Save-InstallSideState([string]$Label) {
    $dest = Join-Path $ProofRoot "install-side-$Label"
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    $logsDir = Join-Path $HermesHome "logs"
    if (Test-Path -LiteralPath $logsDir) {
        Copy-Item $logsDir (Join-Path $dest "hermes-logs") -Recurse -Force -ErrorAction SilentlyContinue
    }
    $resultFile = Join-Path $HermesHome ".hermes-update-result.json"
    if (Test-Path -LiteralPath $resultFile) {
        Copy-Item $resultFile $dest -Force -ErrorAction SilentlyContinue
    }
    $venvScripts = Join-Path $InstallDir "venv\Scripts"
    if (Test-Path -LiteralPath $venvScripts) {
        Get-ChildItem -LiteralPath $venvScripts |
            Select-Object Name, Length, LastWriteTime |
            Format-Table -AutoSize | Out-String |
            Set-Content (Join-Path $dest "venv-scripts-ls.txt")
    }
    Get-ChildItem -LiteralPath $HermesHome -ErrorAction SilentlyContinue |
        Select-Object Name, Length, LastWriteTime |
        Format-Table -AutoSize | Out-String |
        Set-Content (Join-Path $dest "hermes-home-ls.txt")
}

function Test-HermesRuns([string]$Label) {
    Save-InstallSideState $Label
    $hermesExe = $null
    try {
        $hermesExe = Get-SourceHermes $InstallDir
    } catch {
        # A pre-handoff release cannot complete inside `hermes update`: its
        # update path reaches no retired-hook seam, so the update ends with the
        # tree at HEAD and no published launcher. The NEXT ordinary startup
        # completes it (hermes_bootstrap -> prepare_launch -> sync PM, publish
        # launchers, re-exec). Drive that startup here, WITHOUT the lazy-install
        # ban, and only when the launcher is missing -- so a healthy update is
        # still judged by the strict checks below, and `--version` probes keep
        # their ban: a probe must never complete an unfinished update.
        Write-Host "  no published launcher yet; running the next ordinary startup (this is what completes a pre-handoff release)"
        $startupHermes = Get-SourceHermesForStartup $InstallDir
        $startupLog = Join-Path $WorkRoot 'logs\post-update-startup.log'
        New-Item -ItemType Directory -Force -Path (Split-Path $startupLog) | Out-Null
        # prepare_launch reports its progress on stderr, and a native command's
        # stderr becomes a terminating NativeCommandError under the wrong
        # preference -- which killed this step before the heal could finish.
        # Same idiom the --version probe below already uses.
        $prevStartupEap = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $startupHermes status 2>&1 | Out-File -Encoding UTF8 $startupLog
            $startupExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $prevStartupEap
        }
        Write-Host "  first startup after the update ran (exit $startupExit); the checks below assert the launcher it must have published"
        $hermesExe = Get-SourceHermes $InstallDir
    }
    & python -B (Join-Path $AssetsDir 'source_driver.py') --root $InstallDir --launcher $hermesExe --desktop $script:ExpectedDesktop
    Assert-True ($LASTEXITCODE -eq 0) "$Label -- read-only install verification (no repair)"
    $prevLazy = $env:HERMES_DISABLE_LAZY_INSTALLS
    $prevBytecode = $env:PYTHONDONTWRITEBYTECODE
    $prevEap = $ErrorActionPreference
    try {
        $env:HERMES_DISABLE_LAZY_INSTALLS = '1'
        $env:PYTHONDONTWRITEBYTECODE = '1'
        $ErrorActionPreference = 'Continue'
        & $hermesExe --version 2>&1 | ForEach-Object { Write-Host "    hermes --version| $_" }
        $versionExit = $LASTEXITCODE
    } finally {
        $env:HERMES_DISABLE_LAZY_INSTALLS = $prevLazy
        $env:PYTHONDONTWRITEBYTECODE = $prevBytecode
        $ErrorActionPreference = $prevEap
    }
    Assert-True ($versionExit -eq 0) "$Label -- hermes --version exits 0"
}

# ----------------------------------------------------------------------------
# Script-install arm: the irm | iex one-liner, headless (the install.ps1
# shipped AT the ref under test, run with flags probed from that ref's own
# script text - older releases reject parameters added later).
# ----------------------------------------------------------------------------
# shellcheck source=../e2e-assets/ts-prefix.ps1
. (Join-Path $PSScriptRoot "e2e-assets\ts-prefix.ps1")
. (Join-Path $PSScriptRoot "e2e-assets\source-driver.ps1")
. (Join-Path $AssetsDir 'source-build-env.ps1')

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
    $flags = @("-HermesHome", $HermesHome, "-InstallDir", $InstallDir)
    $text = Get-Content -LiteralPath $script -Raw
    if ($text -match '\$NonInteractive') { $flags += "-NonInteractive" }
    else { $flags += "-SkipSetup" }
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
    # uv reads the CURRENT DIRECTORY's project metadata. Run from inside this
    # checkout and every interpreter probe in a ref's installer is resolved
    # against HEAD's requires-python (3.14), so it refuses the version that ref
    # pins and the install fails. A user runs the script from their own
    # directory, so give it one that is not a project.
    $runDir = Join-Path $WorkRoot "install-cwd"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $runDir
    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $script @flags 2>&1 | Add-TsPrefix | Out-File -Encoding UTF8 $log
        $installExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $ErrorActionPreference = $prevEap
    Write-LogGroup "install.ps1 ($Label) transcript" $log
    Assert-True ($installExit -eq 0) "install.ps1 ($Label) exited 0"
}

function Assert-DesktopArtifact([string]$Label) {
    Assert-True ($null -ne (Get-DesktopExe)) "$Label -- desktop app built by installer under apps\desktop\release"
}

function Invoke-HermesUpdate {
    # --yes reaches the update subcommand only in later
    # releases; ask the installed binary, never parse its source.
    $hermesExe = Get-SourceHermes $InstallDir
    $updateArgs = @("update")
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $helpText = & $hermesExe update --help 2>&1 | Out-String
    $helpExit = $LASTEXITCODE
    if ($helpExit -ne 0) {
        $ErrorActionPreference = $prevEap
        throw "Installed update --help failed: $helpText"
    }
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
    Assert-True ($updateExit -eq 0) "hermes update exited $updateExit (expected 0)"
}

function Invoke-ManualCardUpdate([string]$ReceiptPath, [string]$TargetSha) {
    Assert-True (Test-Path -LiteralPath $ReceiptPath) "manual update card produced a receipt"
    $manual = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    Assert-True ($manual.command -match '^hermes update(?:\s|$)') "manual update card instructed hermes update"
    Invoke-HermesUpdate
    Assert-True ((Get-InstalledHead) -eq $TargetSha) "manual update landed on target commit"
    Test-HermesRuns "post-manual-update"
    Assert-True ($null -ne (Get-DesktopExe)) "Hermes.exe still present after manual update"
}

function Clear-HistoricalInstallerChurn {
    # The GUI driver refuses a dirty source tree. v2026.7.1's installer leaves
    # its own state there: `npm install` rewrites package-lock.json on Windows
    # (later installers use `npm ci`, #112378) and v2026.6.19-era installers
    # write an unignored .install_method. Undo only that installer-generated
    # state in this disposable clone; any other change still fails, listed.
    # porcelain=v2: Invoke-Git trims output, which would eat v1's leading " M".
    $lines = @((Invoke-Git @("-C", $InstallDir, "status", "--porcelain=v2", "--untracked-files=all")) -split "\r?\n" |
        Where-Object { $_ })
    if ($lines.Count -eq 0) { return }
    Write-Host "  source status before GUI update:"
    $lines | ForEach-Object { Write-Host "    $_" }
    $locks = @(); $other = @()
    foreach ($line in $lines) {
        $fields = $line -split " ", 9
        if ($fields[0] -eq "1" -and $fields[1] -eq ".M" -and $fields.Count -eq 9 -and
            ($fields[8] -eq "package-lock.json" -or $fields[8] -like "*/package-lock.json")) {
            $locks += $fields[8]
        } elseif ($line -eq "? .install_method") {
            Add-Content -LiteralPath (Join-Path $InstallDir ".git\info\exclude") -Value "/.install_method"
        } else {
            $other += $line
        }
    }
    Assert-True ($other.Count -eq 0) "installed source has only installer-generated changes (other: $($other -join '; '))"
    if ($locks.Count) { Invoke-Git (@("-C", $InstallDir, "checkout", "--") + $locks) | Out-Null }
    $left = @((Invoke-Git @("-C", $InstallDir, "status", "--porcelain", "--untracked-files=all")) -split "\r?\n" |
        Where-Object { $_ })
    Assert-True ($left.Count -eq 0) "undid only installer-generated source churn before the GUI update"
}

function Invoke-HermesDesktopAppUpdate([string]$TargetSha) {
    # The hermes-desktop launch surface: `hermes desktop` runs its whole
    # real pipeline; the driver intercepts the product's final spawn
    # (argv/cwd/env captured by e2e-assets/launch-capture/sitecustomize.py)
    # and re-executes it under Playwright, which clicks Update now.
    $hermesExe = Get-SourceHermes $InstallDir
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
    Clear-HistoricalInstallerChurn

    $node = $DriverNode
    $chatOut = Join-Path $ProofRoot 'update-window'
    Remove-Item -LiteralPath (Join-Path $chatOut 'desktop-chat-old.json') -Force -ErrorAction SilentlyContinue
    $script:ChatFailure = $true
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    Push-Location $WorkRoot
    try {
        & $node (Join-Path $AssetsDir "launch-from-spec.mjs") --spec $spec `
            --old-sha (Read-State).old --chat-out $chatOut --mock-url $env:HERMES_E2E_MOCK_URL `
            --result (Join-Path $HermesHome ".hermes-update-result.json") `
            --expect-sha $TargetSha --repo-dir $InstallDir 2>&1 |
            ForEach-Object { Write-Host "  pw| $_" }
        $driveExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $prevEap
    }
    Confirm-OldChat $chatOut
    $manualReceipt = Join-Path $chatOut 'manual-update.json'
    if ($driveExit -eq 42) {
        Invoke-ManualCardUpdate $manualReceipt $TargetSha
        return
    }
    Assert-True ($driveExit -eq 0) "app driven via captured hermes desktop spec; update completed"

    # The production updater relaunches Hermes. Close that verified window
    # normally so the test-owned checkpoint starts and owns its own backend.
    $desktopExe = Get-DesktopExe
    $deadline = (Get-Date).AddMinutes(5)
    $windows = @()
    while ((Get-Date) -lt $deadline) {
        $windows = @(Get-VerifiedDesktopWindows $desktopExe)
        if ($windows.Count -eq 1) { break }
        Start-Sleep -Seconds 2
    }
    Assert-True ($windows.Count -eq 1) "updated desktop relaunched exactly one verified window"
    $script:ChatFailure = $true
    Close-VerifiedDesktop $desktopExe $windows[0].Id
}

# Evidence for a GUI-driver failure, taken while the installer is still alive: which Hermes
# processes exist (was Hermes.exe ever started, and by whom), the installer's thread states,
# and a full memory dump of the installer. The installer's tracing log is buffered and never
# reaches disk when the job kills it; the dump still holds it. A Launch that left the
# installer on LAUNCHING had no other trace (tests/install/e2e-assets/install-and-launch.ahk).
function Save-GuiDriverFailureEvidence([System.Diagnostics.Process]$Installer, [string]$OutDir) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match '^(hermes|msedgewebview2|python|uv|git|node)' -or $_.ParentProcessId -eq $Installer.Id } |
        Sort-Object CreationDate |
        Select-Object ProcessId, ParentProcessId, CreationDate, Name, CommandLine |
        Format-Table -AutoSize -Wrap | Out-String -Width 400 |
        Tee-Object -FilePath (Join-Path $OutDir "processes.txt") | Write-Host
    if ($Installer.HasExited) {
        Write-Host "  Hermes-Setup.exe already exited (code $($Installer.ExitCode) at $($Installer.ExitTime))"
        return
    }
    $Installer.Refresh()
    $Installer.Threads |
        Select-Object Id, ThreadState, WaitReason, StartTime, TotalProcessorTime |
        Format-Table -AutoSize | Out-String -Width 200 |
        Tee-Object -FilePath (Join-Path $OutDir "installer-threads.txt") | Write-Host
    if (-not ('HdE2E.Dump' -as [type])) {
        Add-Type -Namespace HdE2E -Name Dump -MemberDefinition @'
[DllImport("dbghelp.dll", SetLastError = true)]
public static extern bool MiniDumpWriteDump(IntPtr hProcess, uint processId, Microsoft.Win32.SafeHandles.SafeFileHandle hFile, uint dumpType, IntPtr exceptionParam, IntPtr userStreamParam, IntPtr callbackParam);
'@
    }
    $dumpPath = Join-Path $OutDir "Hermes-Setup.dmp"
    $file = [System.IO.File]::Create($dumpPath)
    try {
        # MiniDumpWithFullMemory | MiniDumpWithHandleData | MiniDumpWithThreadInfo
        $ok = [HdE2E.Dump]::MiniDumpWriteDump($Installer.Handle, [uint32]$Installer.Id, $file.SafeFileHandle, 0x1006, [IntPtr]::Zero, [IntPtr]::Zero, [IntPtr]::Zero)
        $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    }
    finally {
        $file.Close()
    }
    if ($ok) { Write-Host "  Hermes-Setup.exe dump: $dumpPath ($([math]::Round((Get-Item $dumpPath).Length / 1MB, 1)) MB)" }
    else { Write-Host "  Hermes-Setup.exe dump failed (Win32 error $err)" }
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

# ----------------------------------------------------------------------------
# Phase: stage -- serve.git with `main` at OLD (advanced to HEAD by update-gui)
# ----------------------------------------------------------------------------
# Mirror of resolve_update_ref's NEXT arm in e2e-assets/installer-common.sh:
# a synthetic child of $Parent whose tree adds one marker file (a real diff,
# not an empty fast-forward), written to the object store only -- no ref, no
# worktree change. A local `clone --bare` copies objects/ wholesale, which is
# how it reaches serve.git. A throwaway index stands in for mktree so no
# NUL-delimited stdin has to cross PowerShell's native pipe.
function New-NextCommit([string]$Repo, [string]$Parent) {
    $marker = Join-Path $WorkRoot "next-marker.txt"
    Set-Content -LiteralPath $marker -Encoding ASCII -Value "synthetic next commit for the HEAD -> NEXT install E2E leg"
    $blob = Invoke-Git @("-C", $Repo, "hash-object", "-w", "--no-filters", $marker)
    $saved = @{}
    $vars = @{
        GIT_INDEX_FILE = (Join-Path $WorkRoot "next.index")
        GIT_AUTHOR_NAME = "Hermes E2E"; GIT_AUTHOR_EMAIL = "e2e@hermes.invalid"
        GIT_COMMITTER_NAME = "Hermes E2E"; GIT_COMMITTER_EMAIL = "e2e@hermes.invalid"
    }
    foreach ($k in $vars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $vars[$k]) }
    try {
        Invoke-Git @("-C", $Repo, "read-tree", $Parent) | Out-Null
        Invoke-Git @("-C", $Repo, "update-index", "--add", "--cacheinfo", "100644,$blob,.hermes-e2e-next") | Out-Null
        $tree = Invoke-Git @("-C", $Repo, "write-tree")
        return Invoke-Git @("-C", $Repo, "commit-tree", $tree, "-p", $Parent, "-m", "e2e: synthetic next commit")
    } finally {
        foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        Remove-Item -LiteralPath $vars.GIT_INDEX_FILE -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-PhaseStage {
    Write-Step "STAGE: bare serve repo, main -> OLD (install base)"

    if (Test-Path -LiteralPath $WorkRoot) {
        Remove-Item -LiteralPath $WorkRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $WorkRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $WorkRoot "logs") -Force | Out-Null
    # The purge above deleted the redirect gitconfig; re-arm it so the
    # bare-clone below (and everything after) sees the redirect file.
    Set-GitRedirect

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
    # NEXT is minted before the clone so it rides along into serve.git.
    $targetLabel = $UpdateRef
    $current = if ($UpdateRef -eq "NEXT") { New-NextCommit $RepoRoot $old } else {
        Invoke-Git @("-C", $RepoRoot, "rev-parse", "${UpdateRef}^{commit}")
    }
    Write-Host "  update target ($targetLabel): $current"
    Assert-True ($old -ne $current) "OLD differs from $targetLabel (an update is genuinely available)"

    # Bare-clone the checkout: this is the repo the installer and updater
    # actually talk to. Local-path clone hardlinks objects, so it's fast
    # even for full history. The published installer carries NO commit pin
    # (Pin { commit: None, branch: main }) -- it installs whatever `main`
    # serves, so staging OLD means parking `main` there; the update phase
    # advances it to HEAD.
    Invoke-Git @("clone", "--bare", "--quiet", $RepoRoot, $ServeRepo) | Out-Null
    Invoke-Git @("-C", $ServeRepo, "cat-file", "-e", "$current^{commit}") | Out-Null
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $old) | Out-Null
    Invoke-Git @("-C", $ServeRepo, "symbolic-ref", "HEAD", "refs/heads/main") | Out-Null

    # Belt-and-braces: SOME installer builds do bake a -Commit pin. A pinned
    # sha is in serve.git's history but not at a ref tip, so the redirected
    # fetch needs any-SHA1 upload-pack permission (GitHub grants the
    # equivalent for fetch of reachable commits).
    Invoke-Git @("-C", $ServeRepo, "config", "uploadpack.allowAnySHA1InWant", "true") | Out-Null
    Write-Host "  serve.git: uploadpack.allowAnySHA1InWant=true (installer commit pin, if any)"

    @{ old = $old; old_ref = $oldRef; current = $current; target_label = $targetLabel } |
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

    # The production installer binary comes from the website. Pair its script
    # input with the source revision it will materialize. A branch-following
    # installer can otherwise run today's install.ps1 against OLD, whose tree
    # legitimately lacks helpers added later (for example
    # apps/desktop/scripts/ensure-rolldown-binding.mjs). The bootstrap's public
    # dev-source seam changes only script resolution. The GUI binary and cloned
    # source remain the real artifacts under test.
    $bootstrapRoot = Join-Path $WorkRoot "bootstrap-source-$Mode"
    $bootstrapScripts = Join-Path $bootstrapRoot "scripts"
    New-Item -ItemType Directory -Path $bootstrapScripts -Force | Out-Null
    $installScript = Join-Path $bootstrapScripts "install.ps1"
    (Invoke-Git @("-C", $RepoRoot, "show", "$ExpectedSha`:scripts/install.ps1")) -join "`n" |
        Set-Content -LiteralPath $installScript -Encoding UTF8
    Copy-Item $installScript (Join-Path $proof "bootstrap-install-script.ps1") -Force
    $scriptBlob = Invoke-Git @("-C", $RepoRoot, "rev-parse", "$ExpectedSha`:scripts/install.ps1")
    @(
        "source_commit=$ExpectedSha"
        "script_blob=$scriptBlob"
    ) | Set-Content -LiteralPath (Join-Path $proof "bootstrap-install-script.txt") -Encoding ASCII
    Write-Host "  bootstrap script is scripts/install.ps1 from $ExpectedLabel ($ExpectedSha)"

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
    New-Item -ItemType Directory -Path $HermesHome -Force | Out-Null

    $recorder = Start-DesktopRecorder (Join-Path $proof "desktop-frames")
    $ahkLog = Join-Path $proof "ahk.log"
    try {
        Save-DesktopScreenshot (Join-Path $proof "00-before-installer.png")

        # Launch the real headed installer. Scope the paired script source to
        # this process only so later product launches cannot inherit it.
        $previousSetupSource = $env:HERMES_SETUP_DEV_REPO_ROOT
        $env:HERMES_SETUP_DEV_REPO_ROOT = $bootstrapRoot
        try {
            $installer = Start-Process -FilePath $setupExe -PassThru
        }
        finally {
            if ($null -eq $previousSetupSource) {
                Remove-Item Env:HERMES_SETUP_DEV_REPO_ROOT -ErrorAction SilentlyContinue
            }
            else {
                $env:HERMES_SETUP_DEV_REPO_ROOT = $previousSetupSource
            }
        }
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
        if ($ahk.ExitCode -ne 0) {
            # Evidence must not replace the driver's own failure below.
            try { Save-GuiDriverFailureEvidence $installer (Join-Path $proof "driver-failure") }
            catch { Write-Host "  evidence capture failed: $_" }
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

    # The installer Launch proof above must pass before a test-owned launch.
    $script:ChatFailure = $true
    Close-VerifiedDesktop (Get-DesktopExe)
    # The installer-launched renderer boots before the journey seeds its local
    # mock provider. Its disposable userData can therefore persist the release's
    # default OpenRouter choice and override the mock on the checkpoint relaunch.
    # Reset only this driver-owned pre-checkpoint state; OLD -> HEAD keeps the
    # state created by the checkpoint itself.
    if (Test-Path -LiteralPath $env:HERMES_DESKTOP_USER_DATA_DIR) {
        Remove-Item -LiteralPath $env:HERMES_DESKTOP_USER_DATA_DIR -Recurse -Force
    }
    New-Item -ItemType Directory -Path $env:HERMES_DESKTOP_USER_DATA_DIR -Force | Out-Null
    $chatPhase = if ($Mode -eq 'install') { 'old' } else { 'new' }
    @{ phase=$chatPhase; launch='post-installer-launch'; handoffProof=$proof } | ConvertTo-Json |
        Set-Content (Join-Path $ProofRoot "desktop-chat-$chatPhase-launch.json")
    Invoke-DesktopCheckpoint $chatPhase $ExpectedSha 'desktop-installer@latest'
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

    $node = $DriverNode

    $recorder = Start-DesktopRecorder (Join-Path $proof "desktop-frames")
    try {
        # Launch the installed app and click through Settings -> About ->
        # Update now. Exit 0 = the app quit for the updater hand-off.
        $driver = Join-Path $AssetsDir 'drive-update.cjs'
        Remove-Item -LiteralPath (Join-Path $proof 'desktop-chat-old.json') -Force -ErrorAction SilentlyContinue
        $script:ChatFailure = $true
        Push-Location $WorkRoot
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $node $driver $desktopExe $proof (Read-State).old 2>&1 |
                ForEach-Object { Write-Host "  $_" }
            $driveExit = $LASTEXITCODE
        } finally {
            Pop-Location
            $ErrorActionPreference = $prevEap
        }
        Confirm-OldChat $proof
        $manualReceipt = Join-Path $proof 'manual-update.json'
        if ($driveExit -eq 42) {
            Invoke-ManualCardUpdate $manualReceipt $TargetSha
            Invoke-DesktopCheckpoint 'new' $TargetSha 'open-app-update-manual'
            return
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
        $mainProc = $null
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
        Assert-True ($null -ne $mainProc) "relaunch has a foregroundable desktop window"
        Save-DesktopScreenshot (Join-Path $proof "99-relaunched-desktop.png")
        # Native relaunch and read-only product verification already passed.
        $script:ChatFailure = $true
        Close-VerifiedDesktop (Get-DesktopExe) $mainProc.Id
        @{ phase='new'; launch='post-update-launch'; automaticRelaunch=$true } | ConvertTo-Json |
            Set-Content (Join-Path $ProofRoot 'desktop-chat-new-launch.json')
        Invoke-DesktopCheckpoint 'new' $TargetSha $Route
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

# --- plugin upgrade-preservation hooks -------------------------------------
# A tagged upgrade must not delete or modify anything under the active
# home's plugins/** or any profile's plugins/** tree: wrapper markers
# (mnemosyne-wrapper.json), symlinked runtimes, and the externally-owned
# sidecar witness outside the home. Fixtures are directory-only (no
# pyproject in the scanned root, nothing downloaded). Snapshot is taken
# after install, verified after update.
function Seed-PreservationFixtures {
    $external = Join-Path $WorkRoot "external-mnemosyne-runtime"
    & python (Join-Path $AssetsDir "verify-plugin-preservation.py") seed --home $HermesHome --external $external
    if ($LASTEXITCODE -ne 0) { throw "could not seed fresh preservation fixtures (exit $LASTEXITCODE)" }
}

function Invoke-PreserveSnapshot {
    $out = Join-Path $WorkRoot "plugin-preservation-snapshot.json"
    if (Test-Path -LiteralPath $out) { throw "refusing to overwrite an existing preservation snapshot" }
    Seed-PreservationFixtures
    & python (Join-Path $AssetsDir "verify-plugin-preservation.py") snapshot --home $HermesHome --out $out
    if ($LASTEXITCODE -ne 0) { throw "plugin preservation snapshot failed (exit $LASTEXITCODE)" }

    Write-Host "  pre-upgrade plugin snapshot: $out"
}

function Invoke-PreserveVerify {
    $snap = Join-Path $WorkRoot "plugin-preservation-snapshot.json"
    if (-not (Test-Path -LiteralPath $snap)) { throw "no pre-upgrade plugin snapshot at $snap; cannot verify preservation" }
    & python (Join-Path $AssetsDir "verify-plugin-preservation.py") verify --home $HermesHome --snapshot $snap `
        --report (Join-Path $WorkRoot "logs\plugin-preservation-report.json")
    if ($LASTEXITCODE -ne 0) { throw "plugin preservation violated by the upgrade (exit $LASTEXITCODE); see the report for deleted/modified entries" }
    Write-Host "  plugins/** and profile plugin trees survived the upgrade intact"
}

# ----------------------------------------------------------------------------
# User-state preservation: the user's OWN durable state, produced through the
# ordinary CLI (never seeded by us), snapshotted before the upgrade and
# verified after it. Complements the plugin-tree contract above, which owns
# plugins/** only.
# ----------------------------------------------------------------------------

function Get-UserStateSessionCount {
    # A real chat turn must actually create a session; if it silently does not,
    # the preservation check below would be testing nothing.
    $probe = Join-Path $WorkRoot 'user-state-session-count.py'
    if (-not (Test-Path -LiteralPath $probe)) {
        @'
import sqlite3, sys
try:
    con = sqlite3.connect("file:" + sys.argv[1].replace("\\", "/") + "?mode=ro", uri=True)
    print(con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
except Exception:
    print(-1)
'@ | Set-Content -LiteralPath $probe -Encoding ASCII
    }
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $value = (& python $probe (Join-Path $HermesHome 'state.db') 2>$null | Out-String).Trim() }
    finally { $ErrorActionPreference = $prevEap }
    if ($value -match '^-?\d+$') { return [int]$value }
    return -1
}

function Invoke-UserStateActions {
    # Everything here is a command a user would run against the real installed
    # CLI with a real (mocked-inference) provider configured.
    $hermes = Get-SourceHermes $InstallDir
    if (-not $script:ChatMock) {
        # Same mock + config writer the desktop chat checkpoints use, so the
        # leg has a genuinely configured provider rather than a dummy key.
        $script:ChatMock = Start-DesktopJourneyMock $DriverNode $AssetsDir $WorkRoot $HermesHome $ProofRoot
    }
    $prevLazy = $env:HERMES_DISABLE_LAZY_INSTALLS
    $prevEap = $ErrorActionPreference
    try {
        $env:HERMES_DISABLE_LAZY_INSTALLS = '1'
        $ErrorActionPreference = 'Continue'

        # Probe, do not assume (the harness rule for old refs).
        $chatHelp = (& $hermes chat --help 2>&1 | Out-String)
        if (-not ($chatHelp -match '(^|\s)-q(\s|,|$)' -or $chatHelp -match '--quiet')) {
            throw 'the installed CLI has no one-shot chat flag; this leg cannot produce a session through the user path'
        }
        $before = Get-UserStateSessionCount
        $log = Join-Path $WorkRoot 'logs\user-state-chat.log'
        # A released tag prints through prompt_toolkit, whose Windows output object needs
        # a console screen buffer: piping the CLI's stdout into the log takes that away
        # and the turn dies with NoConsoleScreenBufferError. Run it under a real
        # pseudoconsole (pty-run.py) and keep the capture.
        & python -B (Join-Path $AssetsDir 'pty-run.py') --out $log --timeout 900 -- $hermes chat -q "Reply with the single word: ok"
        $chatExit = $LASTEXITCODE
        Write-LogGroup 'first real chat turn' $log
        if ($chatExit -ne 0) { throw "the first chat turn failed (exit $chatExit); see $log" }
        $after = Get-UserStateSessionCount
        # A fresh install has no state.db until the first turn: -1 means "no
        # readable db yet", the expected starting point, not a failure.
        if ($before -lt 0) { $before = 0 }
        if ($after -lt 0) { $after = 0 }
        if (-not ($after -gt $before)) {
            throw "the chat turn produced no session row (state.db sessions $before -> $after)"
        }
        Write-Host "  a real turn created a session (state.db sessions $before -> $after)"

        if (-not (Test-Path -LiteralPath (Join-Path $HermesHome 'auth.json'))) {
            # A starting tag may not have this subcommand yet: a harness
            # limitation, not a preservation failure.
            & $hermes auth add --help *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Host '  SKIP hermes auth add does not exist on this ref; auth.json is not covered by this leg'
            }
            else {
            # The provider id and the flags are vintage surfaces, so probe them
            # like the rest of this harness does. v2026.8.31 answers "Unknown
            # provider: openai" -- _is_known_provider accepts a registry provider,
            # 'openrouter', or a custom pool -- and HEAD prompts for an optional
            # label unless --label is given, which EOFs with no tty. Take the
            # first provider the installed CLI accepts.
            $authLog = Join-Path $WorkRoot 'logs\user-state-auth.log'
            $labelFlags = @()
            if ((& $hermes auth add --help 2>&1 | Out-String) -match '--label') {
                $labelFlags = @('--label', 'e2e-preservation')
            }
            $added = $false
            foreach ($provider in @('openrouter', 'anthropic')) {
                Add-Content -LiteralPath $authLog -Value "=== hermes auth add $provider ==="
                & $hermes auth add $provider --type api-key `
                    --api-key 'e2e-preservation-not-a-real-key' @labelFlags 2>&1 |
                    Out-File -Encoding UTF8 -Append $authLog
                if (Test-Path -LiteralPath (Join-Path $HermesHome 'auth.json')) {
                    $added = $true
                    break
                }
            }
            if (-not $added) {
                throw "hermes auth add failed for openrouter and anthropic; see $authLog"
            }
            Write-Host '  a pooled credential exists (auth.json)'
            }
        }

        if (-not (Test-Path -LiteralPath (Join-Path $HermesHome 'profiles\e2e-second'))) {
            # Same vintage surface as auth add above: a starting tag may predate
            # the profile command entirely, and that is a harness limitation,
            # not a preservation failure.
            & $hermes profile create --help *> $null
            if ($LASTEXITCODE -ne 0) {
                Write-Host '  SKIP hermes profile create does not exist on this ref; profiles/e2e-second is not covered by this leg'
            }
            else {
            & $hermes profile create e2e-second 2>&1 |
                Out-File -Encoding UTF8 (Join-Path $WorkRoot 'logs\user-state-profile.log')
            if ($LASTEXITCODE -ne 0) { throw 'hermes profile create failed' }
            if (-not (Test-Path -LiteralPath (Join-Path $HermesHome 'profiles\e2e-second'))) {
                throw 'hermes profile create produced no profile dir'
            }
            # Factory templates migrate intentionally; preserve an authored profile instead.
            Add-Content -LiteralPath (Join-Path $HermesHome 'profiles\e2e-second\SOUL.md') `
                -Encoding UTF8 -Value "`nUser preference: preserve my e2e-second profile identity across upgrades."
            Write-Host '  a second profile exists (profiles/e2e-second)'
            }
        }
    }
    finally {
        $env:HERMES_DISABLE_LAZY_INSTALLS = $prevLazy
        $ErrorActionPreference = $prevEap
    }
}

function Invoke-UserStateSnapshot {
    $snap = Join-Path $WorkRoot 'user-state-snapshot.json'
    if (Test-Path -LiteralPath $snap) { throw 'refusing to overwrite an existing user-state snapshot' }
    & python (Join-Path $AssetsDir 'verify-user-state.py') snapshot --home $HermesHome --out $snap
    if ($LASTEXITCODE -ne 0) { throw "user-state snapshot failed (exit $LASTEXITCODE)" }
    Write-Host "  pre-upgrade user-state snapshot: $snap"
}

function Invoke-UserStateVerify {
    $snap = Join-Path $WorkRoot 'user-state-snapshot.json'
    if (-not (Test-Path -LiteralPath $snap)) {
        throw 'no pre-upgrade user-state snapshot; cannot claim preservation'
    }
    $report = Join-Path $WorkRoot 'logs\user-state-report.json'
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        & python (Join-Path $AssetsDir 'verify-user-state.py') verify --home $HermesHome `
            --snapshot $snap --report $report
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $prevEap }
    if ($code -ne 0) {
        throw "the upgrade changed the user's own state (exit $code); report at $report"
    }
    Write-Host "  the user's own durable state survived the upgrade"
}

function Assert-RedirectIsTransportOnly {
    # The redirect must stay at TRANSPORT level: `hermes update` resolves its
    # channel from the release archive and validates the record against
    # `git config --get remote.origin.url`. If the configured URL ever looked
    # like the rehearsal source, channel resolution would fail and this leg
    # would be testing a fork install rather than the real user path.
    $official = @('https://github.com/NousResearch/hermes-agent.git',
                  'git@github.com:NousResearch/hermes-agent.git')
    $configured = (Invoke-Git @('-C', $InstallDir, 'config', '--get', 'remote.origin.url') | Out-String).Trim()
    Assert-True ($official -contains $configured) "origin stays configured as an official URL (got '$configured')"
    $real = if ($env:HERMES_E2E_REAL_GIT) { $env:HERMES_E2E_REAL_GIT } else { 'git' }
    $observed = (& $real -C $InstallDir remote get-url origin 2>$null | Out-String).Trim()
    Assert-True ($observed -match 'serve\.git|^file://') "git transport is redirected to the staged repo (got '$observed')"
}

function Assert-UserShims {
    # A launcher left pointing at a vanished tree is the "update lost
    # something" shape a checkout-hash assertion cannot see.
    $hermes = Get-SourceHermes $InstallDir
    Assert-True (Test-Path -LiteralPath $hermes) "a usable launcher still exists after the upgrade ($hermes)"
    $userShim = Join-Path $HermesHome 'bin\hermes.exe'
    if (-not (Test-Path -LiteralPath $userShim)) { $userShim = Join-Path $HermesHome 'bin\hermes.cmd' }
    if (Test-Path -LiteralPath $userShim) {
        $prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        try {
            & $userShim --version 2>&1 | Out-Null
            $shimExit = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $prevEap }
        Assert-True ($shimExit -eq 0) "the $HermesHome\bin launcher still runs after the upgrade"
    }
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath) {
        # The fixture home contains ``..`` while Windows can persist the same
        # directory canonically. Compare path identities, not raw substrings.
        $expectedUserBin = [IO.Path]::GetFullPath((Join-Path $HermesHome 'bin')).TrimEnd('\')
        $userPathEntries = @(
            foreach ($entry in ($userPath -split ';')) {
                if (-not $entry) { continue }
                $expanded = [Environment]::ExpandEnvironmentVariables($entry)
                try { [IO.Path]::GetFullPath($expanded).TrimEnd('\') }
                catch { $expanded.TrimEnd('\') }
            }
        )
        Assert-True ($userPathEntries -icontains $expectedUserBin) `
            "the USER PATH still exposes $expectedUserBin (actual: $userPath)"
    }
}

function Invoke-PhaseInstall {
    # Dispatch on the install axis. Each arm ends with the same contract:
    # checkout at OLD, hermes runs, and state carries how OLD landed so any
    # update arm can follow any install arm.
    $state = Read-State
    $script:ExpectedDesktop = if ($InstallMethod -eq 'installer-script') { 'absent' } else { 'present' }
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
    if ($InstallMethod -ne 'desktop-installer@latest') {
        Invoke-DesktopCheckpoint 'old' $state.old $InstallMethod
    }
    Assert-RedirectIsTransportOnly
    Invoke-UserStateActions
}

function Invoke-PhaseUpdate {
    $state = Read-State
    $script:ExpectedDesktop = if ($InstallMethod -ne 'installer-script' -or $Route -in @(
        'installer-script+desktop', 'desktop-installer@latest', 'open-app-update', 'hermes-desktop-app-update'
    )) { 'present' } else { 'absent' }
    $env:HERMES_HOME = $HermesHome
    # Match the POSIX driver's explicit opt-out when a detached updater bypasses
    # the PATH shim and sees our local transport as a fork.
    New-Item -ItemType File -Path (Join-Path $HermesHome ".skip_upstream_prompt") -Force | Out-Null

    # The update becomes available the way it does for a real user: the
    # remote's main moves forward. The GUI route re-advances harmlessly
    # (same sha); script routes need it here because only the GUI arm's
    # helper used to own this step.
    # The mock provider is journey setup, not an upgrade mutation. Configure it
    # before preservation snapshots so its stable endpoint is part of baseline state.
    if ($Route -in @('open-app-update', 'hermes-desktop-app-update', 'desktop-installer@latest')) {
        Start-JourneyChat
    }
    # Snapshot every plugin tree BEFORE the upgrade moves anything.
    Invoke-PreserveSnapshot
    # ... and the user's own durable state, produced by the install phase
    # through the ordinary CLI.
    Invoke-UserStateSnapshot
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
            Invoke-PhaseInstallGui -Mode "update" -ExpectedSha $state.current -ExpectedLabel $state.target_label
            Assert-DesktopArtifact "HEAD"
        }
    }

    Assert-True ((Get-InstalledHead) -eq $state.current) "checkout landed on $($state.target_label)"
    Test-HermesRuns "post-update"
    Assert-UserShims
    Invoke-PreserveVerify
    Invoke-UserStateVerify
    if ($Route -notin @('open-app-update', 'desktop-installer@latest')) {
        Invoke-DesktopCheckpoint 'new' $state.current $Route
    }
}

function Invoke-CheckedPhaseUpdate {
    Remove-Item -LiteralPath (Join-Path $WorkRoot "known-failure.json") -Force -ErrorAction SilentlyContinue
    # Only evidence produced by this update attempt can match an exception.
    foreach ($oldLog in @((Join-Path $WorkRoot "logs\update.log"), (Join-Path $HermesHome "logs\desktop.log"))) {
        if (Test-Path -LiteralPath $oldLog) { Move-Item -LiteralPath $oldLog -Destination "$oldLog.before-update" -Force }
    }
    try {
        Invoke-PhaseUpdate
    } catch {
        $failure = $_
        if ($script:ChatFailure) { throw $failure }
        $node = $DriverNode
        $classification = & $node (Join-Path $AssetsDir "known-failures.cjs") $WorkRoot $InstallMethod $Route $failure.Exception.Message
        $classificationExit = $LASTEXITCODE
        if ($classificationExit -ne 0) { throw $failure }
        $receipt = ($classification | Out-String) | ConvertFrom-Json
        Write-Host "KNOWN FAILURE [$($receipt.id)]: $($receipt.title)"
        Write-Host "  $($receipt.explanation)"
        if ($env:GITHUB_OUTPUT) {
            Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value "known_failure=$($receipt.id)" -Encoding UTF8
        }
        if ($env:GITHUB_STEP_SUMMARY) {
            Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding UTF8 -Value "Known historical failure: $($receipt.title). See the result chart footnote and uploaded known-failure.json."
        }
    }
}

function Invoke-PhaseVerifyStamp {
    # Runs AFTER everything (install, update, the new runtime's launch and its
    # smoke checks): the bootstrap-complete receipt and the checkout's source
    # stamp must both tell the truth about the final HEAD. A separate phase —
    # not an install-phase check — because the bootstrap marker can complete
    # on a LATER run than the install itself.
    $state = Read-State
    $head = Get-InstalledHead
    Assert-True ($head -match '^[0-9a-f]{40}$') "installed HEAD readable: '$head'"
    Write-Host "  install HEAD: $($head.Substring(0, 12))"
    & python -B (Join-Path $RepoRoot 'scripts\verify-bootstrap-version-stamp.py') `
        --stamp (Join-Path $InstallDir '.hermes-bootstrap-complete') `
        --repo $InstallDir --expect-commit $state.current
    if ($LASTEXITCODE -ne 0) { throw "stamp verification failed (exit $LASTEXITCODE)" }
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
# The HEAD start is the fresh-machine leg: HEAD's installer on a box with
# nothing on it, git included (see Set-GitRedirect).
$script:FreshMachine = ($InstallRef -eq "HEAD")

Set-GitRedirect

try {
switch ($Phase) {
    "stage"   { Invoke-PhaseStage }
    "install" { Invoke-SourceBuild { Invoke-PhaseInstall } }
    "update"  { Invoke-SourceBuild { Invoke-CheckedPhaseUpdate } }
    "verify-stamp" { Invoke-PhaseVerifyStamp }
    "all" {
        Invoke-PhaseStage
        Invoke-SourceBuild { Invoke-PhaseInstall }
        Invoke-SourceBuild { Invoke-CheckedPhaseUpdate }
        Invoke-PhaseVerifyStamp
    }
}

} finally {
    if ($script:ChatMock -and -not $script:ChatMock.HasExited) {
        Stop-Process -Id $script:ChatMock.Id -ErrorAction SilentlyContinue
    }
}
Write-Host ""
Write-Host "Phase '$Phase' completed successfully."
