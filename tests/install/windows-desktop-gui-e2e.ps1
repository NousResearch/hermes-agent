# ============================================================================
# Windows Desktop GUI install + update E2E driver (the REAL user flow)
# ============================================================================
# Same before -> current -> next staging as windows-desktop-e2e.ps1, but every
# leg goes through the surfaces a real user touches:
#
#   INSTALL   - downloads the production Hermes-Setup.exe from the website,
#               launches it HEADED, and AutoHotkey clicks Install, waits,
#               then clicks Launch. The real Electron Hermes.exe must appear.
#               The exe runs EXACTLY as shipped: it downloads its own pinned
#               install.ps1 from GitHub raw and installs its baked
#               BUILD_PIN_COMMIT -- so the "before" version in this harness is
#               the website release pin, the literal starting point of every
#               real GUI user. (The strict HEAD~1 -> HEAD guarantee is the
#               contract harness's job; this one covers the production pin.)
#   UPDATE x2 - launches the installed Hermes.exe under Playwright's Electron
#               driver and CLICKS Settings -> About -> "Update now". That
#               spawns the production hand-off (desktop-update.ps1 via
#               cmd start, or the staged hermes-setup.exe on old checkouts),
#               the app quits, the detached updater runs `hermes update`,
#               rebuilds the desktop, and RELAUNCHES Hermes.exe. We assert
#               the whole chain: target sha, marker cleanup, working hermes,
#               and the relaunched app window. Leg 1 lands on CURRENT, leg 2
#               on the synthetic NEXT.
#
# PROOF: screenshots at every renderer step (Playwright), full-desktop
# screenshots around the installer/AHK phases, a rolling desktop capture
# (every 3s) for the whole run, ahk.log, and the hand-off log. All uploaded
# as CI artifacts.
#
# DEVIATIONS FROM PRODUCTION (each one deliberate and small):
#   * git URL redirect (GIT_CONFIG_GLOBAL) routes the canonical repo URLs
#     to the staged serve.git -- the staging requirement itself. The
#     installer's raw.githubusercontent install.ps1 download and the pinned
#     ZIP fallback are NOT redirected (real network, as shipped).
#   * serve.git gets uploadpack.allowAnySHA1InWant=true so the installer's
#     baked -Commit pin can be fetched from the redirected clone the same
#     way GitHub's upload-pack allows it.
#   * A dummy provider key is seeded after install so the update legs see
#     the ready app shell instead of the onboarding overlay (a real
#     updating user has a configured provider).
#
# Usage mirrors windows-desktop-e2e.ps1:
#   powershell -File tests\install\windows-desktop-gui-e2e.ps1 -Phase all
#   ... -Phase stage / install-gui / update-gui-to-current / update-gui-to-next
# ============================================================================

param(
    [ValidateSet("stage", "install-gui", "update-gui-to-current", "update-gui-to-next", "all")]
    [string]$Phase = "all",
    [string]$RepoRoot = "",
    [string]$WorkRoot = $(if ($env:HERMES_E2E_WORKROOT) { $env:HERMES_E2E_WORKROOT } else { Join-Path $env:TEMP "hermes-desktop-gui-e2e" }),
    [string]$SetupExeUrl = "https://hermes-assets.nousresearch.com/Hermes-Setup.exe"
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
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE): $output"
        }
        return ($output | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Set-GitRedirect {
    # Same mechanism (and same install.ps1-clobber rationale) as
    # windows-desktop-e2e.ps1: a driver-owned gitconfig via GIT_CONFIG_GLOBAL.
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

function Save-DesktopScreenshot([string]$OutFile) {
    # Single full-desktop screenshot (all monitors' primary screen).
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
    # Close the desktop app the blunt way between legs (a user quitting).
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
# Phase: stage -- reuse the contract driver's stage (identical staging)
# ----------------------------------------------------------------------------
function Invoke-PhaseStage {
    Write-Step "STAGE (gui): delegating to windows-desktop-e2e.ps1 -Phase stage"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File (Join-Path $PSScriptRoot "windows-desktop-e2e.ps1") `
        -Phase stage -RepoRoot $RepoRoot -WorkRoot $WorkRoot
    if ($LASTEXITCODE -ne 0) { throw "stage phase failed" }
    # The website installer pins a specific release commit (-Commit <sha>).
    # That sha is in serve.git's history but not at a ref tip, so the
    # redirected fetch needs any-SHA1 upload-pack permission (GitHub grants
    # the equivalent for archive/fetch of reachable commits).
    Invoke-Git @("-C", $ServeRepo, "config", "uploadpack.allowAnySHA1InWant", "true") | Out-Null
    Write-Host "  serve.git: uploadpack.allowAnySHA1InWant=true (installer commit pin)"
    New-Item -ItemType Directory -Path $ProofRoot -Force | Out-Null
}

# ----------------------------------------------------------------------------
# Phase: install-gui -- website Hermes-Setup.exe, headed, AHK-driven
# ----------------------------------------------------------------------------
function Invoke-PhaseInstallGui {
    $state = Read-State
    Write-Step "INSTALL (GUI): Hermes-Setup.exe from the website, headed, AHK clicks"
    $proof = Join-Path $ProofRoot "install-gui"
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
        # Surface the installer's own log win or lose.
        $bootLog = Join-Path $HermesHome "logs\bootstrap-installer.log"
        if (Test-Path -LiteralPath $bootLog) {
            Write-Host "  --- bootstrap-installer.log (tail) ---"
            Get-Content -LiteralPath $bootLog -Tail 40 | ForEach-Object { Write-Host "  | $_" }
            Copy-Item $bootLog $proof -Force -ErrorAction SilentlyContinue
        }
    }

    # Close the freshly launched app (user quits after first look).
    Stop-HermesAppProcesses "post-install"

    # The website exe installs its baked release pin -- record it as the
    # "before" version. It must differ from CURRENT (an update is genuinely
    # available). We do NOT require it to be an ancestor of CURRENT: on a
    # real `push: main` run CURRENT is main's tip and the release pin is
    # behind it (ancestor), but on a feature branch CURRENT has diverged
    # from main, so the release pin legitimately isn't in its ancestry. The
    # update leg resets the checkout to serve.git's main ref regardless, and
    # asserts it lands on CURRENT -- that is the real forward-update proof.
    $installedSha = Get-InstalledHead
    Write-Host "  installer landed on: $installedSha (website release pin)"
    Assert-True ($installedSha -ne $state.current) "installed pin differs from CURRENT (an update is genuinely available)"
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & git -C $InstallDir merge-base --is-ancestor $installedSha $state.current 2>&1 | Out-Null
    $isAncestor = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    if ($isAncestor) {
        Write-Host "  [ok] installed pin is an ancestor of CURRENT (linear before -> current)"
    } else {
        Write-Host "  [note] installed pin is NOT an ancestor of CURRENT -- expected on a diverged feature branch; the update leg still resets to CURRENT"
    }
    Test-HermesRuns "post-install-gui"
    Assert-True ($null -ne (Get-DesktopExe)) "packaged Desktop Hermes.exe exists"

    # Seed a provider so the update legs meet the ready app shell, not the
    # onboarding overlay (an updating user has a configured provider).
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path -LiteralPath $envFile) -or -not ((Get-Content $envFile -Raw -ErrorAction SilentlyContinue) -match "OPENROUTER_API_KEY")) {
        Add-Content -LiteralPath $envFile -Value "OPENROUTER_API_KEY=sk-or-e2e-placeholder-not-a-real-key"
    }
    Write-Host "  seeded placeholder provider key for update legs"

    # Suppress the interactive "add upstream remote?" prompt during the GUI
    # update legs. Our serve.git origin (file://) looks like a fork to
    # `hermes update`, so `_sync_with_upstream_if_needed` would call bare
    # input() -- which HANGS FOREVER when the Desktop spawns the hand-off via
    # `cmd start /min` (a real but empty console; input() blocks waiting for a
    # keystroke that never comes). Real GUI users on the official github
    # origin never hit this path (_is_fork is false). The skip marker
    # (.skip_upstream_prompt in HERMES_HOME) is the product's own mechanism
    # for "don't ask about upstream", so setting it keeps the update flow
    # faithful while avoiding the fork-only prompt.
    New-Item -ItemType File -Path (Join-Path $HermesHome ".skip_upstream_prompt") -Force | Out-Null
    Write-Host "  set .skip_upstream_prompt (serve.git origin looks like a fork; avoids the fork-only input() hang)"
}

# ----------------------------------------------------------------------------
# Update legs: real app, real clicks (Playwright Electron driver)
# ----------------------------------------------------------------------------
function Invoke-GuiUpdateLeg([string]$TargetSha, [string]$LegName, [string]$LegSlug) {
    Write-Step "UPDATE GUI ($LegName): advance served main -> $TargetSha, click Update now"
    $proof = Join-Path $ProofRoot $LegSlug
    New-Item -ItemType Directory -Path $proof -Force | Out-Null

    $env:HERMES_HOME = $HermesHome
    Invoke-Git @("-C", $ServeRepo, "update-ref", "refs/heads/main", $TargetSha) | Out-Null

    $desktopExe = Get-DesktopExe
    Assert-True ($null -ne $desktopExe) "$LegName -- packaged Hermes.exe present before update"

    $resultPath = Join-Path $HermesHome ".hermes-update-result.json"
    $markerPath = Join-Path $HermesHome ".hermes-update-in-progress"
    Remove-Item -LiteralPath $resultPath -Force -ErrorAction SilentlyContinue

    $node = Get-ManagedNode
    $appsDesktop = Join-Path $InstallDir "apps\desktop"
    # @playwright/test is a workspace devDependency; the root `npm ci`
    # HOISTS it to the repo-root node_modules, not apps/desktop's. Resolve
    # it the way Node will (walk up from apps/desktop) instead of asserting
    # a hardcoded path that hoisting makes wrong.
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $node -e "require.resolve('@playwright/test', { paths: [process.argv[1]] })" $appsDesktop 2>&1 | Out-Null
    $pwResolved = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prevEap
    Assert-True $pwResolved "$LegName -- @playwright/test resolvable from installed apps/desktop"

    $recorder = Start-DesktopRecorder (Join-Path $proof "desktop-frames")
    try {
        # Launch the installed app and click through Settings -> About ->
        # Update now. Exit 0 = the app quit for the updater hand-off.
        # Copy the driver INTO the installed apps/desktop first: Node resolves
        # require('@playwright/test') from the SCRIPT's own directory upward,
        # so running it from the CI checkout would resolve the wrong (or no)
        # node_modules.
        $driver = Join-Path $appsDesktop "e2e-drive-update.cjs"
        Copy-Item (Join-Path $AssetsDir "drive-update.cjs") $driver -Force
        Push-Location $appsDesktop
        try {
            & $node $driver $desktopExe $proof 2>&1 |
                ForEach-Object { Write-Host "  $_" }
            $driveExit = $LASTEXITCODE
        } finally {
            Pop-Location
            Remove-Item -LiteralPath $driver -Force -ErrorAction SilentlyContinue
        }
        Assert-True ($driveExit -eq 0) "$LegName -- GUI driver clicked Update now and the app quit for hand-off"

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
        # This can take a LONG time: the website release we installed is weeks
        # of main behind CURRENT, so the update pulls a large diff AND does a
        # full Electron desktop rebuild (vite + electron-builder) plus a uv
        # sync. The desktop-build output goes to logs/update.log (not the
        # streamed handoff log), so we tail update.log here to show progress
        # instead of going silent for tens of minutes.
        Write-Host "  waiting for the detached updater to finish (up to 90 min; large release->CURRENT rebuild) ..."
        $updateLog = Join-Path $HermesHome "logs\update.log"
        $updateLogPos = 0
        $deadline = (Get-Date).AddMinutes(90)
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
            Assert-True ([bool]$result.ok) "$LegName -- updater result ok=true"
        } else {
            Write-Host "  (no result JSON -- staged-binary updater path; relying on sha/marker/relaunch asserts)"
        }

        # Marker may briefly outlive the result write; allow it a moment.
        $mDeadline = (Get-Date).AddMinutes(2)
        while ((Get-Date) -lt $mDeadline -and (Test-Path -LiteralPath $markerPath)) { Start-Sleep -Seconds 5 }
        Assert-True (-not (Test-Path -LiteralPath $markerPath)) "$LegName -- update marker cleaned up"

        Assert-True ((Get-InstalledHead) -eq $TargetSha) "$LegName -- checkout landed on target commit"
        Test-HermesRuns $LegName
        Assert-True ($null -ne (Get-DesktopExe)) "$LegName -- Hermes.exe still present after update"

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
        Assert-True ($null -ne $relaunched) "$LegName -- updater relaunched the desktop app"
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
            Write-Host "  --- desktop-update-handoff.log (tail) ---"
            Get-Content -LiteralPath $handoffLog -Tail 60 | ForEach-Object { Write-Host "  | $_" }
            Copy-Item $handoffLog (Join-Path $proof "desktop-update-handoff.log") -Force -ErrorAction SilentlyContinue
        }
        # Quit the relaunched app so the next leg (or job teardown) is clean.
        Stop-HermesAppProcesses $LegName
    }
}

function Invoke-PhaseUpdateGuiToCurrent {
    $state = Read-State
    Invoke-GuiUpdateLeg $state.current "BASE -> CURRENT (GUI)" "update-gui-to-current"
}

function Invoke-PhaseUpdateGuiToNext {
    $state = Read-State
    Invoke-GuiUpdateLeg $state.next "CURRENT -> NEXT (GUI)" "update-gui-to-next"
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
Write-Host "Windows Desktop GUI E2E driver (real user flow)"
Write-Host "  phase:    $Phase"
Write-Host "  repo:     $RepoRoot"
Write-Host "  workroot: $WorkRoot"

Set-GitRedirect

switch ($Phase) {
    "stage"                 { Invoke-PhaseStage }
    "install-gui"           { Invoke-PhaseInstallGui }
    "update-gui-to-current" { Invoke-PhaseUpdateGuiToCurrent }
    "update-gui-to-next"    { Invoke-PhaseUpdateGuiToNext }
    "all" {
        Invoke-PhaseStage
        Invoke-PhaseInstallGui
        Invoke-PhaseUpdateGuiToCurrent
        Invoke-PhaseUpdateGuiToNext
    }
}

Write-Host ""
Write-Host "Phase '$Phase' completed successfully."
