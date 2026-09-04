# Hermes Agent bootstrap: git checkout + venv + hermes command on PATH.
# Heavy dependencies (tool binaries, browsers, node) are pm's job after
# this: `hermes pm install`. Stage protocol kept for Hermes-Setup:
#   -Manifest             print the stage list as JSON
#   -Stage NAME [-Json]   run one stage
#   -NonInteractive       skip stages that need input
#   -IncludeDesktop       add the desktop build stage
#   -ProtocolVersion      print the stage protocol version
param(
    [string]$Branch = "main",
    [string]$Commit = "",
    [string]$HermesHome = $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }),
    [string]$InstallDir = $(if ($env:HERMES_HOME) { "$env:HERMES_HOME\hermes-agent" } else { "$env:LOCALAPPDATA\hermes\hermes-agent" }),
    [switch]$Manifest,
    [string]$Stage,
    [switch]$ProtocolVersion,
    [switch]$NonInteractive,
    [switch]$Json,
    [switch]$IncludeDesktop
)

$ErrorActionPreference = "Stop"
$RepoUrl = if ($env:HERMES_REPO_URL) { $env:HERMES_REPO_URL } else { "https://github.com/NousResearch/hermes-agent.git" }

# --- BEGIN GENERATED: bootstrap pins (scripts/gen-bootstrap-pins.py) ---
# Derived from pm/lock.json. DO NOT EDIT BY HAND:
# run scripts/gen-bootstrap-pins.py after a pin bump.
$script:UvPinVersion = "0.12.3"
$script:UvPinFiles = @{
    "win32-x64" = @{
        Url    = "https://github.com/astral-sh/uv/releases/download/0.12.3/uv-x86_64-pc-windows-msvc.zip"
        Sha256 = "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e"
    }
    "win32-arm64" = @{
        Url    = "https://github.com/astral-sh/uv/releases/download/0.12.3/uv-aarch64-pc-windows-msvc.zip"
        Sha256 = "4343217d668727b8a8eb5cad92389a1d2eeead93c89940d1b955ba1bb15462eb"
    }
}

$script:GitPinVersion = "2.53.0+3"
$script:GitPinFiles = @{
    "win32-x64" = @{
        Url    = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.3/Git-2.53.0.3-64-bit.tar.bz2"
        Sha256 = "1661f02e85a7901ad7920e2a358ee3772ed9066b00d8590bf2d9046ef10aa8b2"
    }
    "win32-arm64" = @{
        Url    = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.3/Git-2.53.0.3-arm64.tar.bz2"
        Sha256 = "4015f05a68bd2bcf3cc6c426e8d44b65d670fbb879225bb7b7c347cfc3a2758a"
    }
}
# --- END GENERATED: bootstrap pins ---

# Resolve the pm store root (same resolution as pm's store_root()):
# $env:HERMES_RUNTIME_DIR wins, else <HermesHome>\tools.
function Get-PmStoreRoot {
    if ($env:HERMES_RUNTIME_DIR) { return $env:HERMES_RUNTIME_DIR }
    return (Join-Path $HermesHome "tools")
}

# The MACHINE's architecture (registry PROCESSOR_ARCHITECTURE), not the
# interpreter's — an x64 powershell on Windows-on-ARM must stage arm64.
function Get-WindowsArch {
    $machineArch = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment' -ErrorAction SilentlyContinue).PROCESSOR_ARCHITECTURE
    if ($machineArch -eq 'ARM64') { return 'arm64' }
    return 'x64'
}

# Provision uv for this host from the pinned pm/lock.json artifact. Stages
# the EXACT artifact pm itself uses into the same store slot
# (<store>\uv-<version>-<target>\), sha256-verified, so pm adopts the same
# bytes — no astral-latest, no irm|iex. Returns the uv.exe path.
function Get-Uv {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }  # dev shortcut; fetches nothing
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:UvPinFiles[$target]
    if (-not $pin) {
        Fail "no pinned uv artifact for $target; install uv manually: https://docs.astral.sh/uv/"
    }
    $entry = Join-Path (Get-PmStoreRoot) "uv-$($script:UvPinVersion)-$target"
    $uvExe = Join-Path $entry "uv.exe"
    if (Test-Path $uvExe) { return $uvExe }
    Log "staging pinned uv $($script:UvPinVersion) ($target) into the pm store"
    $tmpDir = Join-Path ([IO.Path]::GetTempPath()) "hermes-uv-bootstrap-$PID"
    try {
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $zipPath = Join-Path $tmpDir "uv.zip"
        Invoke-WebRequest -Uri $pin.Url -OutFile $zipPath -UseBasicParsing
        # Digest check BEFORE extraction — a mismatched archive is deleted,
        # never unpacked.
        $digest = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $pin.Sha256.ToLowerInvariant()) {
            Fail "uv digest mismatch (expected $($pin.Sha256), got $digest)"
        }
        $extractDir = Join-Path $tmpDir "unpacked"
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        # The zip carries uv.exe (+ uvx.exe) at the root or under one
        # versioned wrapper dir — take whichever layout arrived.
        $found = Get-ChildItem -Path $extractDir -Filter "uv.exe" -Recurse | Select-Object -First 1
        if (-not $found) { Fail "uv.exe not found in the downloaded archive" }
        New-Item -ItemType Directory -Force -Path $entry | Out-Null
        Move-Item -Path $found.FullName -Destination $uvExe -Force
        $uvx = Get-ChildItem -Path $extractDir -Filter "uvx.exe" -Recurse | Select-Object -First 1
        if ($uvx) { Move-Item -Path $uvx.FullName -Destination (Join-Path $entry "uvx.exe") -Force }
    } finally {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (& $uvExe --version 2>$null)) { Fail "pinned uv staged but does not run on this host" }
    return $uvExe
}

# Provision git for this host from the pinned pm/lock.json artifact, into
# the same store slot (<store>\git-<version>-<target>\) pm uses. Returns the
# git.exe path, or $null when no pinned artifact exists for this target.
function Get-PinnedGit {
    $existing = Get-Command git -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }  # dev shortcut; fetches nothing
    $target = "win32-$(Get-WindowsArch)"
    $pin = $script:GitPinFiles[$target]
    if (-not $pin) { return $null }
    $entry = Join-Path (Get-PmStoreRoot) "git-$($script:GitPinVersion)-$target"
    $gitExe = Join-Path $entry "cmd\git.exe"
    if (Test-Path $gitExe) { return $gitExe }
    Log "staging pinned git $($script:GitPinVersion) ($target) into the pm store"
    $tmpDir = Join-Path ([IO.Path]::GetTempPath()) "hermes-git-bootstrap-$PID"
    try {
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $tarPath = Join-Path $tmpDir "git.tar.bz2"
        Invoke-WebRequest -Uri $pin.Url -OutFile $tarPath -UseBasicParsing
        # Digest check BEFORE extraction — the archive IS code.
        $digest = (Get-FileHash -Path $tarPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $pin.Sha256.ToLowerInvariant()) {
            Fail "git digest mismatch (expected $($pin.Sha256), got $digest)"
        }
        $extractDir = Join-Path $tmpDir "unpacked"
        New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
        # The pinned artifact is a git-for-windows tar.bz2 (the same one pm
        # itself extracts). Windows 10+ ships bsdtar with bzip2 support.
        & tar.exe -xf $tarPath -C $extractDir
        if ($LASTEXITCODE) { Fail "failed to extract pinned git archive" }
        # Layout: Git-<ver>/cmd\git.exe — flatten the single wrapper dir.
        $inner = @(Get-ChildItem $extractDir)
        $src = $extractDir
        if ($inner.Count -eq 1 -and $inner[0].PSIsContainer) { $src = $inner[0].FullName }
        if (-not (Test-Path (Join-Path $src "cmd\git.exe"))) { Fail "git.exe not found in the downloaded archive" }
        if (Test-Path $entry) { Remove-Item -Recurse -Force $entry }
        Move-Item $src $entry
    } finally {
        Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $gitExe
}

# Ensure a usable git for the rest of the ladder: pinned pm store slot
# first, then PATH. Returns $true on success.
function Ensure-Git {
    $g = Get-PinnedGit
    if (-not $g) { return $false }
    if ($g -ne "git") {
        # Store-staged git: expose cmd + usr\bin on this process's PATH so
        # bare `git` works for the rest of the ladder (the same dirs pm's
        # git package env() composes).
        $gitEntry = Split-Path (Split-Path $g -Parent) -Parent
        $env:Path = "$gitEntry\cmd;$gitEntry\usr\bin;$env:Path"
    }
    return $true
}

function Log([string]$msg) { Write-Host "[hermes] $msg" -ForegroundColor Blue }
function Fail([string]$msg) { Write-Host "[hermes] $msg" -ForegroundColor Red; exit 1 }

function Emit-Frame([bool]$ok, [string]$name, [bool]$skipped, [string]$reason = "") {
    $frame = [ordered]@{ ok = $ok; stage = $name; skipped = $skipped }
    if ($reason) { $frame.reason = $reason }
    $frame | ConvertTo-Json -Compress | Write-Output
}

$Stages = @(
    @{ name = "prerequisites"; title = "System prerequisites"; category = "runtime"; needs_user_input = $false },
    @{ name = "repository"; title = "Download Hermes Agent"; category = "runtime"; needs_user_input = $false },
    @{ name = "venv"; title = "Create Python environment"; category = "runtime"; needs_user_input = $false },
    @{ name = "python-deps"; title = "Install Python dependencies"; category = "runtime"; needs_user_input = $false },
    @{ name = "node-deps"; title = "Install tool dependencies"; category = "runtime"; needs_user_input = $false },
    @{ name = "path"; title = "Install hermes command"; category = "runtime"; needs_user_input = $false },
    @{ name = "config"; title = "Prepare config and skills"; category = "configuration"; needs_user_input = $false },
    @{ name = "setup"; title = "Configure API keys and settings"; category = "configuration"; needs_user_input = $true },
    @{ name = "gateway"; title = "Configure gateway service"; category = "configuration"; needs_user_input = $true }
)
if ($IncludeDesktop) {
    $Stages += @{ name = "desktop"; title = "Build desktop app"; category = "runtime"; needs_user_input = $false }
}
$Stages += @{ name = "complete"; title = "Finish install"; category = "runtime"; needs_user_input = $false }

function Stage-Prerequisites {
    if (-not (Ensure-Git)) {
        Fail "git is required. Install Git for Windows: https://git-scm.com/download/win"
    }
    Log "prerequisites ok (git)"
}

function Stage-Repository {
    if (Test-Path (Join-Path $InstallDir ".git")) {
        Log "updating $InstallDir"
        git -C $InstallDir fetch origin $Branch; if ($LASTEXITCODE) { Fail "git fetch failed" }
        git -C $InstallDir checkout $Branch; if ($LASTEXITCODE) { Fail "git checkout failed" }
        git -C $InstallDir pull --ff-only origin $Branch
        if ($LASTEXITCODE) { Log "not fast-forwardable; keeping local state" }
    } else {
        Log "cloning $RepoUrl ($Branch) into $InstallDir"
        New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) | Out-Null
        git clone --branch $Branch $RepoUrl $InstallDir; if ($LASTEXITCODE) { Fail "git clone failed" }
    }
    if ($Commit) {
        git -C $InstallDir checkout $Commit; if ($LASTEXITCODE) { Fail "could not pin commit $Commit" }
    }
}

function Stage-Venv {
    $uv = Get-Uv
    Log "creating venv"
    Push-Location $InstallDir
    & $uv venv --allow-existing venv; $code = $LASTEXITCODE
    Pop-Location
    if ($code) { Fail "uv venv failed" }
}

# Delegate the whole python+venv+tools install to pm: stage the pinned uv,
# let uv run pm.cli, and pm provisions the interpreter, the venv (default
# extras = [all], so it matches what `hermes update` force-syncs), and the
# tool store — all hash-verified against pm/lock.json + uv.lock. install.ps1
# no longer runs `uv sync` directly; pm is the single install authority
# (the run_locked_uv_sync contract moved into pm/packages.py::uv_env).
function Invoke-BootstrapPm {
    $uv = Get-Uv
    $lock = Get-Content (Join-Path $InstallDir "pm\lock.json") -Raw | ConvertFrom-Json
    $pyPin = $lock.packages.python
    $pyVersion = if ($pyPin) { ($pyPin.version -split '\+')[0] -replace '^(\d+\.\d+).*', '$1' } else { '3.11' }
    Log "delegating python + venv + tools to pm (hash-verified via uv.lock)"
    Push-Location $InstallDir
    try {
        & $uv run --no-project --python $pyVersion python -m pm.cli install
        if ($LASTEXITCODE) { Fail "pm install failed" }
    } finally {
        Pop-Location
    }
}

function Stage-PythonDeps {
    Invoke-BootstrapPm
}

function Stage-NodeDeps {
    Log "tool dependencies are managed by pm (hermes pm install)"
}

function Stage-Path {
    $binDir = Join-Path $HermesHome "bin"
    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    # Mint the boot launchers (hermes / hermes-acp) bound to the pm STORE
    # python with PYTHONPATH=repo;venv-site-packages — never the venv
    # python (no boot through the venv; pyvenv.cfg is inert dead config).
    # The venv python below is install-time machinery (the materializer),
    # not a boot path. On a fresh install the store interpreter does not
    # exist yet, so a runtime-resolving .cmd delegator is staged;
    # hermes_cli/_install_repair.py upgrades it to an exe once
    # `hermes pm install` materializes the store.
    $venvPython = Join-Path $InstallDir "venv\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { Fail "venv python missing at $venvPython" }
    Push-Location $InstallDir
    & $venvPython -c "from hermes_cli._launchers import ensure_install_launchers; import sys; written = ensure_install_launchers(r'$InstallDir', r'$binDir'); print(';'.join(written)); sys.exit(0 if written else 1)"
    $code = $LASTEXITCODE
    Pop-Location
    if ($code) { Fail "launcher staging failed" }
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$binDir;$userPath", "User")
        Log "added $binDir to your user PATH (new shells pick it up)"
    }
    Log "hermes command installed at $binDir"
}

function Stage-Config {
    foreach ($d in @("cron","sessions","logs","pairing","hooks","image_cache","audio_cache","memories","skills")) {
        New-Item -ItemType Directory -Force -Path (Join-Path $HermesHome $d) | Out-Null
    }
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path $envFile)) {
        $example = Join-Path $InstallDir ".env.example"
        if (Test-Path $example) { Copy-Item $example $envFile } else { New-Item -ItemType File -Path $envFile | Out-Null }
    }
    $cfg = Join-Path $HermesHome "config.yaml"
    $cfgExample = Join-Path $InstallDir "cli-config.yaml.example"
    if (-not (Test-Path $cfg) -and (Test-Path $cfgExample)) { Copy-Item $cfgExample $cfg }
    Log "config prepared in $HermesHome"
}

function Stage-Setup {
    if ($NonInteractive) { return }
    & (Join-Path $InstallDir "venv\Scripts\python.exe") (Join-Path $InstallDir "hermes") setup
}

function Stage-Gateway {
    if ($NonInteractive) { return }
    & (Join-Path $InstallDir "venv\Scripts\python.exe") (Join-Path $InstallDir "hermes") gateway install
}

function Stage-Desktop {
    Install-DesktopVoiceDeps
    Install-Desktop
}

function Stage-Complete {
    $commit = $Commit
    if (-not $commit) { $commit = git -C $InstallDir rev-parse HEAD 2>$null }
    if ($commit) {
        $marker = [ordered]@{
            schemaVersion = 1
            pinnedCommit = "$commit"
            pinnedBranch = $Branch
            completedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        }
        $marker | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $InstallDir ".hermes-bootstrap-complete") -Encoding UTF8
        Log "bootstrap complete marker written (pinned $commit)"
    }
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

function Install-PlatformSdks {
    # Ensure messaging-platform SDKs matching tokens the user added to
    # ~/.hermes/.env are importable.  Two problems this solves:
    #
    # 1. The tiered `uv pip install` cascade above can fall through to a
    #    lower tier when the first fails (common when RL git deps choke),
    #    which silently skips some messaging SDKs from [messaging].
    # 2. `uv` creates the venv without pip.  If a messaging SDK ends up
    #    missing, the user can't `pip install python-telegram-bot` to
    #    recover -- pip simply isn't in their venv.
    #
    # Strategy: bootstrap pip via `python -m ensurepip` (idempotent), then
    # for each token set in .env, verify the matching SDK imports.  If not,
    # run one targeted `pip install` as last-chance recovery.  Keeps fresh
    # Windows installs from hitting silent "python-telegram-bot not installed"
    # at runtime.
    if ($NoVenv) {
        Write-Info "Skipping platform-SDK verification (-NoVenv: no venv to bootstrap)"
        return
    }

    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Warn "Skipping platform-SDK verification: $pythonExe not found"
        return
    }

    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) { return }
    $envLines = Get-Content $envPath -ErrorAction SilentlyContinue

    # Map: env var set in .env -> (import name, pip spec matching [messaging] extra).
    # Specs mirror pyproject.toml to avoid version drift.
    $sdkMap = @(
        @{ Var = "TELEGRAM_BOT_TOKEN"; Import = "telegram";  Spec = "python-telegram-bot[webhooks]>=22.6,<23" },
        @{ Var = "DISCORD_BOT_TOKEN";  Import = "discord";   Spec = "discord.py[voice]>=2.7.1,<3" },
        @{ Var = "SLACK_BOT_TOKEN";    Import = "slack_sdk"; Spec = "slack-sdk>=3.27.0,<4" },
        @{ Var = "SLACK_APP_TOKEN";    Import = "slack_bolt";Spec = "slack-bolt>=1.18.0,<2" },
        @{ Var = "WHATSAPP_ENABLED";   Import = "qrcode";    Spec = "qrcode>=7.0,<8" }
    )

    # Which tokens are actually set (not placeholder)?
    $needed = @()
    foreach ($sdk in $sdkMap) {
        $match = $envLines | Where-Object {
            $_ -match ("^" + [regex]::Escape($sdk.Var) + "=.+") `
            -and $_ -notmatch "your-token-here" `
            -and $_ -notmatch "^\s*#"
        }
        if ($match) { $needed += $sdk }
    }
    if ($needed.Count -eq 0) { return }

    Write-Host ""
    Write-Info "Verifying platform SDKs for tokens found in $envPath ..."

    # Verify each SDK's import without triggering side-effect imports.
    # Quirk: PowerShell wraps non-zero-exit native stderr as a
    # NativeCommandError that prints even with `2>$null` / `*> $null`
    # unless we set $ErrorActionPreference to SilentlyContinue for the
    # span.  Save + restore rather than nuking globally.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        $missing = @()
        foreach ($sdk in $needed) {
            & $pythonExe -c "import $($sdk.Import)" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $missing += $sdk
                Write-Warn "  $($sdk.Import) NOT importable (needed for $($sdk.Var))"
            } else {
                Write-Success "  $($sdk.Import) OK"
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($missing.Count -eq 0) { return }

    # Bootstrap pip into the venv if it isn't there.  `uv` creates venvs
    # without pip; ensurepip is the stdlib-blessed way to add it.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        & $pythonExe -m pip --version 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Info "Bootstrapping pip into venv (uv doesn't ship pip)..."
            & $pythonExe -m ensurepip --upgrade 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "ensurepip failed -- can't auto-install missing SDKs."
                Write-Info "Manual recovery: $UvCmd pip install `"$($missing[0].Spec)`""
                return
            }
        }

        foreach ($sdk in $missing) {
            Write-Info "  Installing $($sdk.Spec) ..."
            & $pythonExe -m pip install $sdk.Spec 2>&1 | ForEach-Object { Write-Host "    $_" }
            if ($LASTEXITCODE -eq 0) {
                Write-Success "  Installed $($sdk.Import)"
            } else {
                Write-Warn "  Failed to install $($sdk.Spec). Recover manually: $pythonExe -m pip install `"$($sdk.Spec)`""
            }
        }
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}

function Invoke-SetupWizard {
    if ($SkipSetup) {
        Write-Info "Skipping setup wizard (-SkipSetup)"
        return
    }

    if ($NonInteractive) {
        # The setup wizard prompts for API keys, model choice, persona, etc.
        # Non-interactive callers (GUI installer) own that UX themselves; let
        # them drive it after install.ps1 returns.
        Write-Info "Skipping setup wizard (non-interactive). Configure via the GUI or 'hermes setup'."
        return
    }

    Write-Host ""
    Write-Info "Starting setup wizard..."
    Write-Host ""

    Push-Location $InstallDir

    # Run hermes setup using the venv Python directly (no activation needed)
    if (-not $NoVenv) {
        & ".\venv\Scripts\python.exe" -m hermes_cli.main setup
    } else {
        python -m hermes_cli.main setup
    }

    Pop-Location
}

function Start-GatewayIfConfigured {
    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) { return }

    $hasMessaging = $false
    $content = Get-Content $envPath -ErrorAction SilentlyContinue
    foreach ($var in @("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "WHATSAPP_ENABLED")) {
        $match = $content | Where-Object { $_ -match "^${var}=.+" -and $_ -notmatch "your-token-here" }
        if ($match) { $hasMessaging = $true; break }
    }

    if (-not $hasMessaging) { return }

    $hermesCmd = "$InstallDir\venv\Scripts\hermes.exe"
    if (-not (Test-Path $hermesCmd)) {
        $hermesCmd = "hermes"
    }

    # If WhatsApp is enabled but not yet paired, run foreground for QR scan
    $whatsappEnabled = $content | Where-Object { $_ -match "^WHATSAPP_ENABLED=true" }
    $whatsappSession = "$HermesHome\whatsapp\session\creds.json"
    if ($whatsappEnabled -and -not (Test-Path $whatsappSession)) {
        Write-Host ""
        Write-Info "WhatsApp is enabled but not yet paired."
        Write-Info "Running 'hermes whatsapp' to pair via QR code..."
        Write-Host ""
        # Non-interactive callers (GUI installer, CI) skip the QR-pair prompt;
        # WhatsApp pairing requires a human looking at a phone camera, so the
        # downstream UI is responsible for surfacing this when it makes sense.
        if (-not $NonInteractive) {
            $response = Read-Host "Pair WhatsApp now? [Y/n]"
            if ($response -eq "" -or $response -match "^[Yy]") {
                try {
                    & $hermesCmd whatsapp
                } catch {
                    # Expected after pairing completes
                }
            }
        } else {
            Write-Info "Skipping WhatsApp pairing prompt (non-interactive)."
        }
    }

    Write-Host ""
    Write-Info "Messaging platform token detected!"
    Write-Info "The gateway handles messaging platforms and cron job execution."
    Write-Host ""

    # In non-interactive mode the gateway lifecycle is the caller's problem
    # (the GUI manages its own gateway process, CI doesn't want background
    # services on the build agent, etc.).  Treat it like the user declined.
    if ($NonInteractive) {
        Write-Info "Skipping gateway autostart prompt (non-interactive)."
        Write-Info "Start the gateway later with: hermes gateway"
        return
    }

    $response = Read-Host "Would you like to start the gateway now? [Y/n]"

    if ($response -eq "" -or $response -match "^[Yy]") {
        Write-Info "Starting gateway in background..."
        try {
            $logFile = "$HermesHome\logs\gateway.log"
            Start-Process -FilePath $hermesCmd -ArgumentList "gateway" `
                -RedirectStandardOutput $logFile `
                -RedirectStandardError "$HermesHome\logs\gateway-error.log" `
                -WindowStyle Hidden
            Write-Success "Gateway started! Your bot is now online."
            Write-Info "Logs: $logFile"
            Write-Info "To stop: close the gateway process from Task Manager"
        } catch {
            Write-Warn "Failed to start gateway. Run manually: hermes gateway"
        }
    } else {
        Write-Info "Skipped. Start the gateway later with: hermes gateway"
    }
}

function Write-Completion {
    Write-Host ""
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Green
    Write-Host "|              [OK] Installation Complete!                |" -ForegroundColor Green
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Green
    Write-Host ""
    
    # Show file locations
    Write-Host "* Your files:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Config:    " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\config.yaml"
    Write-Host "   API Keys:  " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\.env"
    Write-Host "   Data:      " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\cron\, sessions\, logs\"
    Write-Host "   Code:      " -NoNewline -ForegroundColor Yellow
    Write-Host "$HermesHome\hermes-agent\"
    Write-Host ""
    
    Write-Host "---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "* Commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   hermes              " -NoNewline -ForegroundColor Green
    Write-Host "Start chatting"
    Write-Host "   hermes setup        " -NoNewline -ForegroundColor Green
    Write-Host "Configure API keys & settings"
    Write-Host "   hermes config       " -NoNewline -ForegroundColor Green
    Write-Host "View/edit configuration"
    Write-Host "   hermes config edit  " -NoNewline -ForegroundColor Green
    Write-Host "Open config in editor"
    Write-Host "   hermes gateway      " -NoNewline -ForegroundColor Green
    Write-Host "Start messaging gateway (Telegram, Discord, etc.)"
    Write-Host "   hermes update       " -NoNewline -ForegroundColor Green
    Write-Host "Update to latest version"
    Write-Host ""
    
    Write-Host "---------------------------------------------------------" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[*] Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $HasNode) {
        Write-Host "Note: Node.js could not be installed automatically." -ForegroundColor Yellow
        Write-Host "Browser tools need Node.js. Install manually:" -ForegroundColor Yellow
        Write-Host "  https://nodejs.org/en/download/" -ForegroundColor Yellow
        Write-Host ""
    }
    
    if (-not $HasRipgrep) {
        Write-Host "Note: ripgrep (rg) was not installed. For faster file search:" -ForegroundColor Yellow
        Write-Host "  winget install BurntSushi.ripgrep.MSVC" -ForegroundColor Yellow
        Write-Host ""
    }
}


function Invoke-StageByName([string]$name) {
    switch ($name) {
        "prerequisites" { Stage-Prerequisites }
        "repository" { Stage-Repository }
        "venv" { Stage-Venv }
        "python-deps" { Stage-PythonDeps }
        "node-deps" { Stage-NodeDeps }
        "path" { Stage-Path }
        "config" { Stage-Config }
        "setup" { Stage-Setup }
        "gateway" { Stage-Gateway }
        "desktop" { Stage-Desktop }
        "complete" { Stage-Complete }
        default { Write-Error "unknown stage: $name"; exit 2 }
    }
}

if ($ProtocolVersion) { Write-Output 1; exit 0 }

if ($Manifest) {
    @{ protocol_version = 1; stages = $Stages } | ConvertTo-Json -Depth 4 -Compress | Write-Output
    exit 0
}

if ($Stage) {
    $known = @($Stages | ForEach-Object { $_.name })
    if ($known -notcontains $Stage -and $Stage -ne "desktop") {
        if ($Json) { Emit-Frame $false $Stage $false "unknown stage: $Stage" }
        else { [Console]::Error.WriteLine("unknown stage: $Stage") }
        exit 2
    }
    $needsInput = ($Stage -eq "setup") -or ($Stage -eq "gateway")
    if ($NonInteractive -and $needsInput) {
        if ($Json) { Emit-Frame $true $Stage $true "needs user input" }
        exit 0
    }
    try {
        Invoke-StageByName $Stage
        if ($Json) { Emit-Frame $true $Stage $false }
        exit 0
    } catch {
        if ($Json) { Emit-Frame $false $Stage $false "$_" }
        exit 1
    }
}

# No -Stage: run the whole ladder.
foreach ($s in @("prerequisites","repository","venv","python-deps","node-deps","path","config","setup","gateway","complete")) {
    Invoke-StageByName $s
}
