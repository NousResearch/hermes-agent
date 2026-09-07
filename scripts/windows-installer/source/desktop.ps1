# Clear the cached Electron download + any half-written unpacked output so the
# next `npm run pack` re-downloads and re-stages from scratch. A corrupt zip in
# the per-user Electron download cache - most often a partial download resumed
# into the same file, leaving concatenated junk - makes electron-builder's
# `app-builder unpack-electron` extract a tree MISSING the electron binary, so
# the final `electron` -> `Hermes` rename dies with ENOENT and every re-run
# repeats the broken extraction forever.
#
# We deliberately do not validate the zip ourselves: the common
# prepended/concatenated-junk corruption slips past naive checks, so a
# self-rolled gate would skip the real-world case. We unconditionally drop the
# cached electron-*.zip (loose copy and any @electron/get hash-subdir copy) plus
# the stale unpacked dir, then let the caller retry once - @electron/get
# re-downloads with its own SHASUM verification, the real source of truth.
#
# Returns the removed paths. Best-effort: never throws.
function Clear-ElectronBuildCache {
    param([string]$DesktopDir)
    $removed = @()

    # Per-user Electron download cache dirs, honoring the overrides @electron/get
    # respects, then the Windows default (%LOCALAPPDATA%\electron\Cache).
    $cacheDirs = @()
    if ($env:electron_config_cache) { $cacheDirs += $env:electron_config_cache }
    if ($env:ELECTRON_CACHE)        { $cacheDirs += $env:ELECTRON_CACHE }
    if ($env:LOCALAPPDATA)          { $cacheDirs += (Join-Path $env:LOCALAPPDATA 'electron\Cache') }
    $cacheDirs += (Join-Path $HOME 'AppData\Local\electron\Cache')

    foreach ($dir in $cacheDirs) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        # Recurse: the bad copy may be the top-level zip OR a copy inside an
        # @electron/get hash subdir.
        $removed += @(Get-ChildItem -LiteralPath $dir -Recurse -Filter 'electron-*.zip' -File -ErrorAction SilentlyContinue | ForEach-Object {
            try { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop; $_.FullName } catch { }
        })
    }

    # A half-written unpacked dir from an interrupted prior pack poisons the
    # rename even after the zip is fixed (win-unpacked / win-arm64-unpacked).
    $releaseDir = Join-Path $DesktopDir 'release'
    if (Test-Path -LiteralPath $releaseDir) {
        $removed += @(Get-ChildItem -LiteralPath $releaseDir -Directory -Filter '*-unpacked' -ErrorAction SilentlyContinue | ForEach-Object {
            try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop; $_.FullName } catch { }
        })
    }

    return $removed
}

# Last-resort Electron mirror after GitHub download fails (#47266).
$script:DesktopElectronFallbackMirror = "https://npmmirror.com/mirrors/electron/"

# Electron package dir -- workspace-local nest first, then root hoist.
function Get-ElectronDir {
    param([string]$InstallDir)
    $desktopLocal = Join-Path $InstallDir 'apps\desktop\node_modules\electron'
    if (Test-Path -LiteralPath $desktopLocal) { return $desktopLocal }
    return (Join-Path $InstallDir 'node_modules\electron')
}

# True when dist/ holds a usable Electron binary (#38673 / run-electron-builder.mjs).
function Test-ElectronDist {
    param([string]$InstallDir)
    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    $distExe = Join-Path $electronDir 'dist\electron.exe'
    return (Test-Path -LiteralPath $distExe)
}

# Best-effort: run electron/install.js to populate dist/ (optional mirror).
function Restore-ElectronDist {
    param([string]$InstallDir, [string]$Mirror)
    if (Test-ElectronDist -InstallDir $InstallDir) { return $true }

    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    $distExe = Join-Path $electronDir 'dist\electron.exe'
    $installer = Join-Path $electronDir 'install.js'
    if (-not (Test-Path -LiteralPath $installer)) { return $false }
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) { return $false }

    $distDir = Join-Path $electronDir 'dist'
    if (Test-Path -LiteralPath $distDir) {
        Remove-Item -LiteralPath $distDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath (Join-Path $electronDir 'path.txt') -Force -ErrorAction SilentlyContinue

    $prevMirror = $env:ELECTRON_MIRROR
    if ($Mirror) { $env:ELECTRON_MIRROR = $Mirror }
    try {
        # Out-Host so the downloader's progress shows on the console WITHOUT
        # leaking into this function's return value (PowerShell returns every
        # object left on the output stream, so a bare pipe here would make the
        # boolean below ambiguous).
        & $node.Source $installer 2>&1 | ForEach-Object { "$_" } | Out-Host
    } catch {
    } finally {
        $env:ELECTRON_MIRROR = $prevMirror
    }
    return (Test-Path -LiteralPath $distExe)
}

function Test-ElectronPkgStagedMissingDist {
    param([string]$InstallDir)
    $electronDir = Get-ElectronDir -InstallDir $InstallDir
    return (
        (Test-Path -LiteralPath (Join-Path $electronDir 'package.json')) -and
        (Test-Path -LiteralPath (Join-Path $electronDir 'install.js')) -and
        (-not (Test-ElectronDist -InstallDir $InstallDir))
    )
}

function Try-RestoreElectronDist {
    param([string]$InstallDir)
    if (Restore-ElectronDist -InstallDir $InstallDir) { return $true }
    if ($env:ELECTRON_MIRROR) { return $false }
    return Restore-ElectronDist -InstallDir $InstallDir -Mirror $script:DesktopElectronFallbackMirror
}

function Install-DesktopVoiceDeps {
    # Desktop ships with working voice out of the box: eagerly install the
    # wake-word + local-STT stacks ([wake] + [voice] extras) instead of
    # leaving them to lazy first-use install. Policy change (Teknium, July
    # 2026, #70509 testing): the first ear-click used to trigger a
    # multi-minute onnxruntime pip install that froze the UI and blew RPC
    # timeouts. Best-effort -- lazy install remains the fallback for anything
    # this step fails to fetch.
    if (-not $script:UvCmd) { Resolve-UvCmd }
    if (-not $script:UvCmd) {
        Write-Warn "uv unavailable -- voice/wake deps will lazy-install at first use instead"
        return
    }
    $env:VIRTUAL_ENV = "$InstallDir\venv"
    Write-Info "Installing voice + wake-word dependencies (onnxruntime, faster-whisper -- 1-3min)..."
    Push-Location $InstallDir
    try {
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install -e ".[wake,voice]" }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Voice + wake-word dependencies installed"
        } else {
            Write-Warn "Voice/wake dependency install failed (exit $LASTEXITCODE) -- they will lazy-install at first use"
        }
    } finally {
        Pop-Location
    }
}

function Install-Desktop {
    # Build apps/desktop into a launchable Hermes.exe. Only called from
    # Stage-Desktop, which is itself only included in the manifest when
    # -IncludeDesktop was passed to install.ps1.
    #
    # The workspace npm install at repo root (done by Install-NodeDeps for
    # browser tools) does NOT pull apps/desktop's dependencies, because the
    # browser-tools workspace at $InstallDir\package.json is a separate
    # workspace from apps/*. We do a full root-level `npm install` here
    # so the workspace resolves apps/desktop's deps (including Electron
    # itself, ~150MB), then run `npm run pack` in apps/desktop which
    # produces the unpacked binary at apps/desktop/release/<os>-unpacked/.
    #
    # The Tauri bootstrap installer's launch_hermes_desktop command
    # resolves apps/desktop/release/win-unpacked/Hermes.exe directly,
    # so an "unpacked" build (electron-builder --dir) is enough -- we
    # don't need to produce an NSIS/MSI artifact here.

    # Always re-resolve Node here. Stages run in separate PowerShell processes,
    # so $script:HasNode from Stage-Node isn't visible; more importantly Test-Node
    # enforces the supported Node lines and prepends the Hermes-managed Node to
    # PATH, so the build never runs on an unsupported system Node -- the cause
    # of the opaque "Build desktop app ... exit code 1" failure (Vite crashes on
    # old Node).
    Test-Node | Out-Null
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warn "Skipping desktop build (Node.js / npm not on PATH)"
        $script:_StageSkippedReason = "Node.js not available"
        return
    }

    $desktopDir = "$InstallDir\apps\desktop"
    if (-not (Test-Path "$desktopDir\package.json")) {
        Write-Warn "Skipping desktop build (apps/desktop not present in checkout)"
        $script:_StageSkippedReason = "apps/desktop not present"
        return
    }

    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Warn "Skipping desktop build (npm not on PATH)"
        $script:_StageSkippedReason = "npm not found"
        return
    }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $sibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $sibling) { $npmExe = $sibling }
    }

    # 1. Workspace-level install so apps/desktop's deps (Electron, Vite,
    # node-pty prebuilds, etc.) actually land in node_modules. This is
    # the SAME `npm install` Install-NodeDeps does for browser tools,
    # but at the root rather than the browser-tools workspace, so all
    # apps/* workspaces resolve.
    Write-Info "Installing desktop workspace dependencies (this includes Electron ~150MB, takes 1-3min)..."
    Push-Location $InstallDir
    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        # Drop --silent so npm emits its full progress + error trail.
        # When this fails on a non-dev box (e.g. native-module build
        # without VS Build Tools, ETARGET on a transitive, etc.), the
        # actual reason needs to reach the Tauri installer's log; with
        # --silent it was completely suppressed and the user just saw
        # "exit 1" with no actionable detail.
        #
        # The streaming sink in bootstrap.rs's run_install_script
        # captures every stdout/stderr line as it's emitted, so we don't
        # need a side TEMP log file -- the installer's bootstrap log
        # IS the artifact a support engineer reads.
        #
        # Prefer `npm ci`: it wipes node_modules and reinstalls from the
        # lockfile, always producing a complete tree. Bare `npm install`
        # can report "up to date" against a stale
        # node_modules\.package-lock.json marker while node_modules is
        # actually empty (Windows workspace-hoisting flake), leaving
        # tsc/typescript unresolved so `npm run pack`'s `tsc -b` dies with
        # no obvious cause. Fall back to `npm install` only if `npm ci`
        # fails (lockfile out of sync / very old npm without ci).
        #
        # Tee the merged output into $npmOut while still emitting every line
        # live. We don't need a side log file (the bootstrap streaming sink
        # is the artifact), but on failure we scan $npmOut for the TLS-trust
        # signature so corporate-proxy users get the NODE_EXTRA_CA_CERTS hint
        # instead of an opaque "exit 1" (issue #38016).
        & $npmExe ci 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable npmOut
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            Write-Info "  npm ci failed (exit $code) -- retrying with npm install..."
            & $npmExe install 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable npmOut
            $code = $LASTEXITCODE
        }
        $ErrorActionPreference = $prevEAP
        if ($code -ne 0) {
            if (Test-ElectronPkgStagedMissingDist -InstallDir $InstallDir) {
                Write-Warn "Desktop dependency install failed with a missing Electron dist; attempting self-heal..."
                Try-RestoreElectronDist -InstallDir $InstallDir | Out-Null
            } else {
                Show-NpmCertHint ($npmOut -join "`n") | Out-Null
                # Replay npm's own debug log into our stream: the terse
                # summary above rarely contains the postinstall stderr
                # (e.g. Electron's install.js) that explains the failure.
                Write-NpmDebugLogTail -NpmOutput ($npmOut -join "`n")
                throw "desktop workspace npm install failed (exit $code) -- see lines above for cause"
            }
        } else {
            Write-Success "Desktop workspace dependencies installed"
        }
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Pop-Location
        throw
    }
    Pop-Location

    # 2. Build apps/desktop. `npm run pack` runs:
    #      assert-root-install + write-build-stamp + stage-native-deps +
    #      tsc -b + vite build + electron-builder --dir
    # The --dir mode produces an unpacked Hermes.exe in
    # apps/desktop/release/win-unpacked/ without bundling NSIS/MSI;
    # we don't need a distributable installer artifact, just a
    # launchable binary the Tauri installer can spawn.
    #
    # CSC_IDENTITY_AUTO_DISCOVERY=false tells electron-builder we are
    # NOT signing the output. Combined with signAndEditExecutable=false in
    # apps/desktop/package.json's build.win block, electron-builder never
    # invokes signtool and therefore never fetches/extracts winCodeSign
    # (whose macOS symlinks crash 7-Zip on non-admin Windows -- a dead end we
    # are NOT trying to work around). The Hermes icon + product name are
    # stamped onto Hermes.exe by our own rcedit step (Set-DesktopExeIdentity)
    # AFTER this build, completely decoupled from electron-builder signing.
    #
    # WIN_CSC_LINK and WIN_CSC_KEY_PASSWORD explicitly cleared as
    # belt-and-suspenders: if the user's environment has them set
    # for some other tool, electron-builder would still try to sign.
    Write-Info "Building desktop app (this takes 1-3 minutes)..."
    $buildLog = "$env:TEMP\hermes-desktop-build-$(Get-Random).log"
    # Seed GITHUB_SHA for write-build-stamp.mjs. The stamp prefers CI env vars
    # over `git rev-parse`, so this covers: (1) node can't find git.exe on PATH
    # even though this PowerShell session can, (2) ZIP/init trees that still
    # lack a HEAD after a failed post-extract fetch. Without it the desktop
    # pack dies with "could not determine git commit" (#50823).
    if (-not $env:GITHUB_SHA) {
        if ($Commit) {
            $env:GITHUB_SHA = $Commit
        } else {
            Push-Location $InstallDir
            try {
                $global:LASTEXITCODE = 0
                $resolvedSha = & git -c windows.appendAtomically=false rev-parse HEAD 2>$null
                if ($LASTEXITCODE -ne 0 -or -not $resolvedSha) {
                    # ZIP path may have FETCH_HEAD after a fetch even when HEAD is unset.
                    $global:LASTEXITCODE = 0
                    $resolvedSha = & git -c windows.appendAtomically=false rev-parse FETCH_HEAD 2>$null
                }
                if ($LASTEXITCODE -eq 0 -and $resolvedSha) {
                    $env:GITHUB_SHA = ("$resolvedSha").Trim()
                }
            } catch { } finally {
                Pop-Location
            }
        }
    }
    if (-not $env:GITHUB_REF_NAME) {
        $env:GITHUB_REF_NAME = if ($Branch) { $Branch } else { "main" }
    }
    if ($env:GITHUB_SHA) {
        $shaPreview = if ($env:GITHUB_SHA.Length -ge 12) { $env:GITHUB_SHA.Substring(0, 12) } else { $env:GITHUB_SHA }
        Write-Info "Desktop build stamp: $shaPreview ($($env:GITHUB_REF_NAME))"
    } else {
        Write-Warn "Could not resolve a git commit for the desktop stamp -- write-build-stamp will use its non-git fallback"
    }
    Push-Location $desktopDir
    $prevEAP = $ErrorActionPreference
    $prevCSCAuto = $env:CSC_IDENTITY_AUTO_DISCOVERY
    $prevWinCscLink = $env:WIN_CSC_LINK
    $prevWinCscKeyPassword = $env:WIN_CSC_KEY_PASSWORD
    try {
        $ErrorActionPreference = "Continue"
        $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
        $env:WIN_CSC_LINK = ""
        $env:WIN_CSC_KEY_PASSWORD = ""
        & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            $purged = @()
            $restored = $false
            if (-not (Test-ElectronDist -InstallDir $InstallDir)) {
                $purged = @(Clear-ElectronBuildCache -DesktopDir $desktopDir)
                $restored = Restore-ElectronDist -InstallDir $InstallDir
            }
            if ($restored) {
                Write-Warn "Desktop build failed - refreshed the Electron download, retrying once:"
                foreach ($p in $purged) { Write-Info "  - $p" }
                & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
                $code = $LASTEXITCODE
            }
        }
        if ($code -ne 0 -and -not $env:ELECTRON_MIRROR) {
            $mirror = $script:DesktopElectronFallbackMirror
            Write-Warn "Desktop build still failing - the Electron download from GitHub looks blocked."
            Write-Warn "Re-downloading Electron via a public mirror ($mirror), then rebuilding:"
            Write-Info "  (set ELECTRON_MIRROR yourself to use a different/trusted mirror)"
            if (-not (Test-ElectronDist -InstallDir $InstallDir)) {
                Restore-ElectronDist -InstallDir $InstallDir -Mirror $mirror | Out-Null
            }
            $prevMirror = $env:ELECTRON_MIRROR
            $env:ELECTRON_MIRROR = $mirror
            try {
                & $npmExe run pack 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $buildLog
                $code = $LASTEXITCODE
            } finally {
                $env:ELECTRON_MIRROR = $prevMirror
            }
        }
        $ErrorActionPreference = $prevEAP
        if ($code -ne 0) {
            $errText = Get-Content $buildLog -Raw -ErrorAction SilentlyContinue
            if ($errText) {
                $snippet = if ($errText.Length -gt 1800) { $errText.Substring(0, 1800) + "..." } else { $errText }
                Write-Info "  desktop build output:"
                foreach ($line in $snippet -split "`n") { Write-Host "    $line" -ForegroundColor DarkGray }
                Write-Info "  Full log: $buildLog"
            }
            # `npm run pack` failures (lifecycle script exits) also land in
            # npm's debug log; replay it so the bootstrap log carries the
            # full evidence even when $buildLog's tail cuts off the cause.
            Write-NpmDebugLogTail -NpmOutput $errText
            throw "apps/desktop build failed (exit $code)"
        }
        Write-Success "Desktop app built"
        Remove-Item -LiteralPath $buildLog -Force -ErrorAction SilentlyContinue
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Pop-Location
        throw
    } finally {
        # Restore env to whatever the caller had -- don't leak our
        # signing-off override into anything install.ps1 invokes later
        # (Stage-PlatformSdks, etc.).
        $env:CSC_IDENTITY_AUTO_DISCOVERY = $prevCSCAuto
        $env:WIN_CSC_LINK = $prevWinCscLink
        $env:WIN_CSC_KEY_PASSWORD = $prevWinCscKeyPassword
    }
    Pop-Location

    # 3. Sanity-check the produced binary. Probe both arches so this works
    # on x64 and arm64 build machines.
    $exeCandidates = @(
        "$desktopDir\release\win-unpacked\Hermes.exe",
        "$desktopDir\release\win-arm64-unpacked\Hermes.exe"
    )
    $found = $false
    $desktopExe = $null
    foreach ($cand in $exeCandidates) {
        if (Test-Path $cand) {
            Write-Success "Desktop ready: $cand"
            $desktopExe = $cand
            $found = $true
            break
        }
    }
    if (-not $found) {
        throw "Desktop build completed but no Hermes.exe was found under $desktopDir\release\*-unpacked\"
    }

    # 3b. The Hermes icon + identity are stamped onto Hermes.exe by the
    #     electron-builder `afterPack` hook (apps/desktop/scripts/after-pack.mjs)
    #     during `npm run pack` above -- for every build, so the installer's
    #     --update rebuild stays branded too. No separate stamp step needed here.
    #     electron-builder's own rcedit step stays disabled (signAndEditExecutable
    #     =false) because enabling it drags in signtool -> winCodeSign -> the
    #     unfixable symlink crash; the afterPack hook runs rcedit directly.

    # 3c. Grant ALL APPLICATION PACKAGES (S-1-15-2-2) RX on the unpacked app
    #     directory. Chromium's GPU/renderer sandboxes CHECK-fail with
    #     0x80000003 when this ACE is missing alongside orphan AppContainer
    #     SIDs under %LOCALAPPDATA% (electron/electron#51761, hermes-agent#38216).
    #     Best-effort -- never fail an otherwise-good install over ACL repair.
    try {
        $appDir = Split-Path -Parent $desktopExe
        & icacls $appDir /grant "*S-1-15-2-2:(OI)(CI)(RX)" /T /C /Q | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Granted AppContainer read access on $appDir"
        } else {
            Write-Warn "icacls AppContainer grant returned exit $LASTEXITCODE for $appDir"
        }
    } catch {
        Write-Warn "Could not grant AppContainer ACL: $($_.Exception.Message)"
    }

    # 4. Create Start Menu + Desktop shortcuts pointing DIRECTLY at the packed
    #    Hermes.exe. We deliberately do NOT point them at `hermes desktop`: that
    #    command rebuilds (npm install + electron-builder) on every launch,
    #    which would cost minutes each time. The packed exe is the consumer --
    #    launching it directly is instant, and updates flow through the
    #    installer's --update path (which rebuilds once, then relaunches).
    New-DesktopShortcuts -TargetExe $desktopExe
}

function New-DesktopShortcuts {
    param([Parameter(Mandatory = $true)][string]$TargetExe)

    # Best-effort: a shortcut failure must never fail an otherwise-good install.
    try {
        $shell = New-Object -ComObject WScript.Shell
        $workDir = Split-Path -Parent $TargetExe

        # Prefer the standalone icon.ico (shipped beside the exe via
        # electron-builder extraResources -> resources/icon.ico) over the exe's
        # embedded resource. An explicit .ico path is more stable across update
        # cycles: pointing at "$TargetExe,0" makes Windows cache the icon it
        # extracted from the exe at shortcut-creation time, and that cached
        # bitmap can persist (showing the OLD/Electron icon) even after the exe
        # is re-stamped on update. A dedicated .ico sidesteps that extraction.
        $iconIco = Join-Path $workDir 'resources\icon.ico'
        if (Test-Path $iconIco) {
            $iconLocation = "$iconIco,0"
        } else {
            $iconLocation = "$TargetExe,0"
        }

        $targets = @(
            (Join-Path ([Environment]::GetFolderPath('Programs')) 'Hermes.lnk'),
            (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Hermes.lnk')
        )

        foreach ($lnkPath in $targets) {
            try {
                $parent = Split-Path -Parent $lnkPath
                if (-not (Test-Path $parent)) {
                    New-Item -ItemType Directory -Force -Path $parent | Out-Null
                }
                $sc = $shell.CreateShortcut($lnkPath)
                $sc.TargetPath = $TargetExe
                $sc.WorkingDirectory = $workDir
                $sc.IconLocation = $iconLocation
                $sc.Description = 'Hermes Agent'
                $sc.Save()
                Write-Success "Shortcut created: $lnkPath"
            } catch {
                Write-Warn "Could not create shortcut $lnkPath : $($_.Exception.Message)"
            }
        }

        # Bust the Windows shell icon cache so the desktop/Start-Menu shortcut
        # repaints with the (possibly newly-stamped) icon instead of a stale
        # cached bitmap. Critical on the --update path: the exe was re-stamped
        # with the Hermes icon, but without this the shortcut can keep drawing
        # the old Electron icon until the user manually refreshes / reboots.
        # Best-effort and silent -- never fail the install over a cosmetic cache.
        try {
            & ie4uinit.exe -show 2>$null
        } catch {
            # ie4uinit may be absent/renamed on some SKUs -- ignore.
        }
    } catch {
        Write-Warn "Skipping shortcut creation: $($_.Exception.Message)"
    }
}

