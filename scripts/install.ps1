# ============================================================================
# Hermes Agent Installer for Windows
# ============================================================================
# Installation script for Windows (PowerShell).
# Uses uv for fast Python provisioning and package management.
#
# Usage:
#   iex (irm https://hermes-agent.nousresearch.com/install.ps1)
#
# Or download and run with options:
#   .\install.ps1 -NoVenv -SkipSetup
#
# ============================================================================

param(
    [switch]$NoVenv,
    [switch]$SkipSetup,
    [string]$Branch = "main",
    # -Commit and -Tag are higher-precedence variants of -Branch for users
    # who need reproducible installs (desktop installer pinning, CI, release
    # bundles).  When set, the repository stage clones $Branch (faster than
    # cloning the full default-branch history) and then `git checkout`s the
    # exact ref.  Precedence: Commit > Tag > Branch.
    [string]$Commit = "",
    [string]$Tag = "",
    [string]$HermesHome = $(if ($env:HERMES_HOME) { $env:HERMES_HOME } else { "$env:LOCALAPPDATA\hermes" }),
    [string]$InstallDir = $(if ($env:HERMES_HOME) { "$env:HERMES_HOME\hermes-agent" } else { "$env:LOCALAPPDATA\hermes\hermes-agent" }),

    # --- Stage protocol (additive; default invocation behaves as before) ----
    # See the "Stage protocol" section near the bottom of the file for the
    # full contract.  Intended for programmatic drivers (the desktop GUI's
    # onboarding wizard, CI, future install.sh parity, etc.).  CLI users
    # running the canonical `irm | iex` one-liner never touch these flags.
    [switch]$Manifest,
    [string]$Stage,
    [switch]$ProtocolVersion,
    [switch]$NonInteractive,
    [switch]$Json,

    # --- Ensure mode (dep_ensure.py entry point) ---
    [string]$Ensure = "",
    [switch]$PostInstall,

    # --- Desktop GUI build (opt-in) ---
    # When set, install.ps1 includes Stage-Desktop in the manifest and
    # builds apps/desktop into a launchable Hermes.exe.
    #
    # Why opt-in:
    #   * Hermes-Setup.exe (the signed Tauri bootstrap installer) passes
    #     -IncludeDesktop so a user who installed via the GUI ends up
    #     with a launchable desktop binary.
    #   * The Electron desktop's own bootstrap-runner.cjs runs install.ps1
    #     from inside an already-launched Hermes.exe; if THAT recursively
    #     built apps/desktop it would try to overwrite the live Hermes.exe
    #     on disk and fail. The recursive path omits the flag.
    #   * The canonical CLI one-liner (irm | iex) omits the flag too;
    #     terminal users don't need a desktop binary built for them, and
    #     `hermes desktop` already builds on demand.
    [switch]$IncludeDesktop
)

$ErrorActionPreference = "Stop"

# Suppress Invoke-WebRequest's per-chunk progress bar.  Windows PowerShell
# 5.1's progress UI repaints synchronously on every received byte, which
# pegs CPU on a single core and throttles downloads by 10-100x (a 57MB
# PortableGit grab can take 5 minutes with progress on vs 20 seconds
# with progress off, on the same network).  Every IWR call in this
# script is fire-and-forget so we never need to see the bar.  Restored
# automatically when the script exits.
$ProgressPreference = "SilentlyContinue"

# Force the console to UTF-8 so non-ASCII output from native commands
# (e.g. playwright's box-drawing progress bars and download banners,
# git's bullet glyphs, npm's check marks) renders correctly instead of
# as IBM437/Windows-1252 mojibake (sequences like 0xE2 0x95 0x94 box-
# drawing chars decoded under the legacy DOS codepage).  This is a
# DISPLAY-only fix; the underlying bytes are already correct.  We do
# NOT change the file's own encoding (it remains pure ASCII for PS 5.1
# parser compatibility; see comments at the top of the entry-point
# dispatch).  This affects only what the user sees in their terminal
# during this install run, and reverts automatically when the script
# exits and the host's console encoding is restored.
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
} catch {
    # Some constrained PowerShell hosts disallow encoding mutation.
    # Mojibake on output is then cosmetic-only, install still works.
}

# ============================================================================
# 8.3 short-path normalization
# ============================================================================
# When the Windows user-profile folder name contains a space (e.g.
# "First Last"), Windows generates an 8.3 short alias for it (e.g. FIRST~1.LAS)
# and may expose %TEMP%/%TMP% in that short form:
#   C:\Users\FIRST~1.LAS\AppData\Local\Temp
# PowerShell's FileSystem provider mishandles the "~1.ext" component when such a
# path is handed to a provider cmdlet like `Tee-Object -FilePath` /
# `Out-File -FilePath`, throwing:
#   "An object at the specified path C:\Users\FIRST~1.LAS does not exist."
# Every Node/Electron build+install stage streams its log to %TEMP% via
# Tee-Object, so they all abort with that error, while the Python/uv stages --
# which never write a side log to %TEMP% through a provider cmdlet -- complete
# fine. Expanding %TEMP%/%TMP% back to their long form once, up front, lets
# every downstream cmdlet (and child process) see a path the provider can
# resolve. (GH: Windows desktop installer fails at Node/Electron stages.)

function ConvertTo-LongPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
    # Only 8.3 short names carry a tilde+digit ("~1"); skip the COM round-trip
    # for ordinary long paths.
    if ($Path -notmatch '~\d') { return $Path }
    try {
        $fso = New-Object -ComObject Scripting.FileSystemObject
        if ($fso.FolderExists($Path)) { return $fso.GetFolder($Path).Path }
        if ($fso.FileExists($Path))   { return $fso.GetFile($Path).Path }
    } catch {
        # COM unavailable / locked-down host: fall back to the original path.
    }
    return $Path
}

foreach ($tmpVar in @('TEMP', 'TMP')) {
    $current = [Environment]::GetEnvironmentVariable($tmpVar)
    if ($current) {
        $expanded = ConvertTo-LongPath $current
        if ($expanded -and $expanded -ne $current) {
            Set-Item -Path "Env:$tmpVar" -Value $expanded
        }
    }
}

# ============================================================================
# Configuration
# ============================================================================

$RepoUrlSsh = "git@github.com:NousResearch/hermes-agent.git"
$RepoUrlHttps = "https://github.com/NousResearch/hermes-agent.git"
$PythonVersion = "3.11"
# Minor versions the installer accepts when the requested $PythonVersion isn't
# available, in preference order.  uv discovers both uv-managed and system
# interpreters, so this list also matches a pre-existing system Python.  Single
# source of truth shared by Test-Python's fallback and Resolve-AvailablePythonVersion.
$PythonFallbackVersions = @("3.12", "3.13", "3.10")
$NodeVersion = "22"

# Stage-protocol version.  Bumped only for genuinely breaking changes to the
# manifest schema, stage-name set semantics, or stdout JSON shape.  Adding a
# new stage does NOT bump this -- drivers iterate the manifest dynamically.
$InstallStageProtocolVersion = 1

# ============================================================================
# Helper functions

# Return the real OS processor architecture as a lowercase string suitable for
# Node.js / electron download URL slugs: "arm64", "x64", or "x86".
#
# Why not just trust [Environment]::Is64BitOperatingSystem or
# [RuntimeInformation]::OSArchitecture?  On Windows on ARM, when this script
# is invoked from Windows PowerShell 5.1 (the default `powershell.exe`) or
# any x64 PowerShell host, the process runs under Prism x64 emulation and
# BOTH of those APIs report `X64` -- they describe the emulated view, not
# the real OS.  We've seen this concretely on Snapdragon X1 hardware: an
# ARM64-based Surface Laptop returns OSArchitecture=X64 from an emulated
# PowerShell session.
#
# Win32_Processor.Architecture is invariant to emulation.  Values:
#   0=x86, 5=ARM, 9=AMD64/x64, 12=ARM64.  We fall back to
#   PROCESSOR_ARCHITEW6432 (set on WoW64 with the real OS arch) and then
#   PROCESSOR_ARCHITECTURE so we still produce a sensible answer if CIM
#   isn't available (locked-down WMI, container, etc.).
function Get-WindowsArch {
    try {
        $proc = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop |
            Select-Object -First 1
        switch ([int]$proc.Architecture) {
            12 { return "arm64" }
            9  { return "x64" }
            0  { return "x86" }
            5  { return "arm" }
        }
    } catch {
        # CIM unavailable -- fall through to env-var path
    }

    $envArch = if ($env:PROCESSOR_ARCHITEW6432) {
        $env:PROCESSOR_ARCHITEW6432
    } else {
        $env:PROCESSOR_ARCHITECTURE
    }
    switch ($envArch) {
        "ARM64" { return "arm64" }
        "AMD64" { return "x64" }
        "x86"   { return "x86" }
        default {
            # Last-resort: respect 64-bitness so we don't ship a 32-bit
            # toolchain to anyone.
            if ([Environment]::Is64BitOperatingSystem) { return "x64" } else { return "x86" }
        }
    }
}

# ============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|             * Hermes Agent Installer                    |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host "|  An open source AI agent by Nous Research.              |" -ForegroundColor Magenta
    Write-Host "+---------------------------------------------------------+" -ForegroundColor Magenta
    Write-Host ""
}

function Write-Info {
    param([string]$Message)
    Write-Host "-> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Invoke-NativeWithRelaxedErrorAction {
    param([scriptblock]$Script)

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Script
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}
function Discard-LockfileChurn {
    param([string]$Repo = $InstallDir)

    if (-not $Repo -or -not (Test-Path (Join-Path $Repo ".git"))) { return }

    try {
        $diff = & git -c windows.appendAtomically=false -C $Repo diff --name-only 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $diff) { return }

        $dirtyPackageDirs = [System.Collections.Generic.HashSet[string]]::new(
            [System.StringComparer]::OrdinalIgnoreCase
        )
        foreach ($path in $diff) {
            if ($path -like "*package.json") {
                $null = $dirtyPackageDirs.Add((Split-Path $path -Parent))
            }
        }

        $dirtyLocks = [System.Collections.Generic.List[string]]::new()
        foreach ($path in $diff) {
            if ($path -notlike "*package-lock.json") { continue }
            $lockDir = Split-Path $path -Parent
            if ($dirtyPackageDirs.Contains($lockDir)) { continue }
            $dirtyLocks.Add($path)
        }

        if ($dirtyLocks.Count -eq 0) { return }
        & git -c windows.appendAtomically=false -C $Repo checkout -- @($dirtyLocks) 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Discarded npm lockfile churn ($($dirtyLocks.Count) file(s))"
        }
    } catch {
        # Best-effort only; never let cleanup block the installer update path.
    }
}
# Inspect npm output for a TLS-trust failure and, if found, print actionable
# remediation. npm/Node surface corporate MITM proxies and missing root CAs as
# "unable to get local issuer certificate" / "self-signed certificate in
# certificate chain" / UNABLE_TO_GET_ISSUER_CERT_LOCALLY -- most commonly while
# Electron's install.js postinstall downloads the Electron binary. The reporter
# usually misreads this as an admin-rights or generic install failure (see
# issue #38016), so detect it once here and route every npm stage through this
# hint. Returns $true when a cert error was detected (caller may adjust its own
# messaging), $false otherwise.
function Show-NpmCertHint {
    param([string]$NpmOutput)
    if (-not $NpmOutput) { return $false }
    $isCertError = $NpmOutput -match "unable to get local issuer certificate" `
        -or $NpmOutput -match "self.signed certificate" `
        -or $NpmOutput -match "UNABLE_TO_GET_ISSUER_CERT_LOCALLY" `
        -or $NpmOutput -match "SELF_SIGNED_CERT_IN_CHAIN" `
        -or $NpmOutput -match "CERT_HAS_EXPIRED"
    if (-not $isCertError) { return $false }
    Write-Warn "This looks like a TLS certificate-trust failure, not a permissions problem."
    Write-Info "  A corporate proxy or antivirus is likely intercepting HTTPS and presenting a"
    Write-Info "  certificate Node.js doesn't trust. To fix, point Node at your org's root CA:"
    Write-Info "    1. Get the corporate root CA as a .pem/.crt from your IT team."
    Write-Info "    2. setx NODE_EXTRA_CA_CERTS `"C:\path\to\corp-ca.pem`""
    Write-Info "    3. Open a NEW terminal (so the env var takes effect) and re-run the installer."
    Write-Info "  Quick (less secure) alternative -- disable TLS verification just for the install:"
    Write-Info "    npm config set strict-ssl false   (re-enable afterwards: npm config set strict-ssl true)"
    return $true
}

# --- Ensure-mode helpers ---

function Resolve-NpmCmd {
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) { return $null }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $npmCmdSibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $npmCmdSibling) { return $npmCmdSibling }
    }
    return $npmExe
}

function Find-SystemBrowser {
    # Honor ONLY an explicit, user-set AGENT_BROWSER_EXECUTABLE_PATH override.
    #
    # We no longer scan well-known install locations for a system browser.
    # Auto-detection silently bound the install to an arbitrary binary instead
    # of the bundled Playwright Chromium, which made the browser tool behave
    # differently across hosts (and, on Linux, picked up a sandboxed Snap
    # Chromium that hangs every browser_navigate). Every install now uses the
    # bundled Chromium unless the user explicitly points elsewhere.
    $override = $env:AGENT_BROWSER_EXECUTABLE_PATH
    if ([string]::IsNullOrWhiteSpace($override)) { return $null }
    if (Test-Path $override) { return $override }
    return $null
}

function Write-BrowserEnv {
    param([string]$BrowserPath)
    if (-not (Test-Path $HermesHome)) {
        New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null
    }
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path $envFile)) {
        Set-Content -Path $envFile -Value "AGENT_BROWSER_EXECUTABLE_PATH=$BrowserPath" -Encoding UTF8
        return
    }
    $content = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
    if ($content -and $content -match "AGENT_BROWSER_EXECUTABLE_PATH=") { return }
    Add-Content -Path $envFile -Value "AGENT_BROWSER_EXECUTABLE_PATH=$BrowserPath" -Encoding UTF8
}

function Install-AgentBrowser {
    param([switch]$SkipChromium)
    $npm = Resolve-NpmCmd
    if (-not $npm) {
        Write-Err "npm not found -- install Node.js first"
        throw "npm not found"
    }

    Write-Info "Installing agent-browser via npm -g --prefix..."
    $prefixDir = Join-Path $HermesHome "node"
    if (-not (Test-Path $prefixDir)) {
        New-Item -ItemType Directory -Path $prefixDir -Force | Out-Null
    }
    $npmLog = [System.IO.Path]::GetTempFileName()
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $npm install -g --prefix $prefixDir --silent --ignore-scripts "agent-browser@^0.26.0" "@askjo/camofox-browser@^1.5.2" 2>&1 | Tee-Object -FilePath $npmLog | Out-Null
    $npmExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($npmExit -ne 0) {
        $npmDetail = Get-Content $npmLog -Raw -ErrorAction SilentlyContinue
        Remove-Item $npmLog -Force -ErrorAction SilentlyContinue
        Write-Err "npm install -g failed (exit $npmExit): $npmDetail"
        Show-NpmCertHint $npmDetail | Out-Null
        throw "npm install failed"
    }
    Remove-Item $npmLog -Force -ErrorAction SilentlyContinue

    if (-not $SkipChromium) {
        $sysBrowser = Find-SystemBrowser
        if ($sysBrowser) {
            Write-BrowserEnv -BrowserPath $sysBrowser
            Write-Info "Explicit browser override set -- skipping bundled Chromium download"
        } else {
            $abExe = Join-Path $prefixDir "agent-browser.cmd"
            if (Test-Path $abExe) {
                Write-Info "Installing Chromium via agent-browser install..."
                $abLog = [System.IO.Path]::GetTempFileName()
                $prevEAP = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                & $abExe install 2>&1 | Tee-Object -FilePath $abLog | Out-Null
                $abExit = $LASTEXITCODE
                $ErrorActionPreference = $prevEAP
                if ($abExit -ne 0) {
                    $abDetail = Get-Content $abLog -Raw -ErrorAction SilentlyContinue
                    Write-Warn "Chromium install failed (exit $abExit): $abDetail"
                }
                Remove-Item $abLog -Force -ErrorAction SilentlyContinue
            } else {
                Write-Warn "agent-browser.cmd not found at $abExe"
            }
        }
    }
    Write-Success "Agent-browser ready"
}

# ============================================================================
# Dependency checks
# ============================================================================

# Resolve the PowerShell host executable used to spawn child PowerShell
# processes (the astral uv installer below).  We must NOT hardcode the bare
# name `powershell`: it names *Windows PowerShell* and only resolves when its
# System32 directory is on PATH.  When install.ps1 is run under PowerShell 7+
# (`pwsh`) -- or any session where `powershell` isn't on PATH -- a bare
# `powershell` spawn dies with "The term 'powershell' is not recognized",
# aborting uv installation (field report: Windows install stuck, uv install
# failed with exactly that message).  Prefer the absolute path of the host we
# are already running in (PATH-independent), then fall back to whichever of
# powershell/pwsh is resolvable, and only then to the bare name.
function Get-PowerShellHostExe {
    try {
        $hostExe = (Get-Process -Id $PID).Path
        if ($hostExe -and (Test-Path $hostExe)) {
            $leaf = Split-Path $hostExe -Leaf
            # Only trust the current host when it is a real PowerShell CLI
            # (not e.g. powershell_ise.exe or an embedded host that can't take
            # `-ExecutionPolicy`/`-Command`).
            if ($leaf -match '^(?i:powershell|pwsh)\.exe$') { return $hostExe }
        }
    } catch { }
    foreach ($candidate in @("powershell", "pwsh")) {
        $cmd = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($cmd -and $cmd.Source) { return $cmd.Source }
    }
    # Last-ditch: hand back the bare name so the spawn surfaces its own error.
    return "powershell"
}

function Install-Uv {
    # Hermes owns its own uv at $HermesHome\bin\uv.exe.  Always install there —
    # no PATH probing, no conda guards, no multi-location resolution chains.
    # The runtime update path (hermes_cli/managed_uv.py) looks in the same
    # place, so install.ps1 and `hermes update` stay in sync.
    $managedUv = Join-Path $HermesHome "bin\uv.exe"

    if (Test-Path $managedUv) {
        $script:UvCmd = $managedUv
        $version = & $managedUv --version
        Write-Success "Managed uv found ($version)"
        return $true
    }

    Write-Info "Installing managed uv into $HermesHome\bin ..."
    New-Item -ItemType Directory -Path (Join-Path $HermesHome "bin") -Force | Out-Null

    # UV_INSTALL_DIR tells the astral installer to place the binary
    # directly into $HermesHome\bin instead of ~/.local/bin.
    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $env:UV_INSTALL_DIR = Join-Path $HermesHome "bin"
        # Spawn via the resolved host exe (see Get-PowerShellHostExe) rather
        # than a bare `powershell`, which isn't guaranteed to be on PATH under
        # PowerShell 7 / pwsh-only setups.
        $psHostExe = Get-PowerShellHostExe
        & $psHostExe -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null
        $ErrorActionPreference = $prevEAP

        if (Test-Path $managedUv) {
            $script:UvCmd = $managedUv
            $version = & $managedUv --version
            Write-Success "Managed uv installed ($version)"
            return $true
        }

        Write-Err "uv installed but not found at $managedUv"
        Write-Info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Write-Err "Failed to install uv: $_"
        Write-Info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    }
}

# Refresh $env:Path from the User + Machine registry hives.  Stage drivers
# invoke each stage in a fresh powershell process, but those processes
# inherit env from the parent driver shell, NOT from the registry.  When
# an earlier stage (Stage-Git, Stage-Node, ...) installs a binary and
# pushes its directory into User PATH, the next child process's $env:Path
# is stale and the binary appears missing.  This helper re-reads PATH
# from the registry so every Invoke-Stage starts from a fresh, up-to-date
# PATH view.  Cheap (registry reads, no I/O elsewhere) and idempotent.
function Sync-EnvPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

# Re-discover uv without re-installing it.  Cross-process stage drivers
# (the desktop GUI's onboarding wizard, CI step-runners) invoke each stage
# in a fresh powershell process, so $script:UvCmd set by Install-Uv in a
# prior process is not visible here.  Later stages (Test-Python,
# Install-Venv, Install-Dependencies, Install-PlatformSdks) call this
# at the top to populate $script:UvCmd from the managed location.
# Throws if uv is not findable — the caller's stage then surfaces a
# clean error via the stage-driver's try/catch.
function Resolve-UvCmd {
    # Already resolved (default invocation path: Install-Uv ran earlier
    # in the same process and set $script:UvCmd).
    if ($script:UvCmd) {
        if ($script:UvCmd -eq "uv") {
            # "uv" on PATH -- verify it's still resolvable (PATH could have
            # changed mid-session; cheap to recheck).
            if (Get-Command uv -ErrorAction SilentlyContinue) { return }
        } elseif (Test-Path $script:UvCmd) {
            return
        }
        # Stale; fall through to re-discover.
    }

    # Check the managed location first — this is where Install-Uv puts it.
    $managedUv = Join-Path $HermesHome "bin\uv.exe"
    if (Test-Path $managedUv) {
        $script:UvCmd = $managedUv
        return
    }

    # Fall back to PATH (covers edge cases where the installer ran in a
    # sibling process and HERMES_HOME wasn't propagated).
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $script:UvCmd = "uv"
        return
    }

    # Refresh PATH from registry in case the current process started before
    # Install-Uv updated User PATH.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $script:UvCmd = "uv"
        return
    }

    throw "uv is not installed. Run install.ps1 -Stage uv first."
}

function Resolve-AvailablePythonVersion {
    # Return the first Python minor version uv can actually find, preferring the
    # requested $PythonVersion and then $PythonFallbackVersions.  Returns $null
    # when none are available.
    #
    # This is the cross-process-safe counterpart to Test-Python's in-memory
    # ``$script:PythonVersion = $fallbackVer`` mutation.  Under Hermes-Setup.exe
    # each ``-Stage NAME`` runs in a *fresh* powershell.exe, so the fallback the
    # ``python`` stage settled on (e.g. 3.12 when 3.11 is absent) does NOT
    # survive into the ``venv`` stage's process -- there $PythonVersion is back
    # at its "3.11" default.  Consumers re-resolve here instead of trusting that
    # default, which is exactly the propagation gap behind issue #50769.
    $candidates = @($PythonVersion) + $PythonFallbackVersions
    $seen = @{}
    foreach ($ver in $candidates) {
        if (-not $ver -or $seen.ContainsKey($ver)) { continue }
        $seen[$ver] = $true
        try {
            $found = & $UvCmd python find $ver 2>$null
            if ($found) { return $ver }
        } catch { }
    }
    return $null
}

function Test-Python {
    Write-Info "Checking Python $PythonVersion..."
    
    # Let uv find or install Python
    try {
        $pythonPath = & $UvCmd python find $PythonVersion 2>$null
        if ($pythonPath) {
            $ver = & $pythonPath --version 2>$null
            Write-Success "Python found: $ver"
            return $true
        }
    } catch { }
    
    # Python not found -- use uv to install it (no admin needed!)
    Write-Info "Python $PythonVersion not found, installing via uv..."
    # Capture EAP outside the try block so the catch's restore call always
    # has a meaningful value (see Install-Uv for the full rationale).
    $prevEAP = $ErrorActionPreference
    try {
        # Temporarily relax ErrorActionPreference: uv writes download progress
        # ("Downloading cpython-3.11.15-windows-x86_64-none (24.5MiB)") to
        # stderr.  With $ErrorActionPreference = "Stop" (set at the top of this
        # script) PowerShell wraps stderr lines from native commands as
        # ErrorRecord objects when captured via 2>&1, then throws a terminating
        # exception on the first one -- even though uv exits 0 and Python was
        # installed successfully.  Verify success via `uv python find`
        # afterwards, which is the reliable signal regardless of exit-code
        # semantics or stderr noise.  This fix was previously landed as
        # commit ec1714e71 and then lost in a release squash; reapplied here.
        $ErrorActionPreference = "Continue"
        $uvOutput = & $UvCmd python install $PythonVersion 2>&1
        $uvExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        # Check if Python is now available (more reliable than exit code
        # since uv may return non-zero due to "already installed" etc.)
        $pythonPath = & $UvCmd python find $PythonVersion 2>$null
        if ($pythonPath) {
            $ver = & $pythonPath --version 2>$null
            Write-Success "Python installed: $ver"
            return $true
        }

        # uv ran but Python still not findable -- show what happened
        if ($uvExitCode -ne 0) {
            Write-Warn "uv python install output:"
            Write-Host $uvOutput -ForegroundColor DarkGray
        }
    } catch {
        # Restore EAP in case the try block threw before the assignment
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Write-Warn "uv python install error: $_"
    }

    # Fallback: check if ANY Python 3.10+ is already available on the system
    Write-Info "Trying to find any existing Python 3.10+..."
    foreach ($fallbackVer in $PythonFallbackVersions) {
        try {
            $pythonPath = & $UvCmd python find $fallbackVer 2>$null
            if ($pythonPath) {
                $ver = & $pythonPath --version 2>$null
                Write-Success "Found fallback: $ver"
                $script:PythonVersion = $fallbackVer
                return $true
            }
        } catch { }
    }

    # Fallback: try system python -- but skip the Microsoft Store stub.
    # On Windows, %LOCALAPPDATA%\Microsoft\WindowsApps\python.exe is a 0-byte
    # reparse-point stub that prints "Python was not found; run without
    # arguments to install from the Microsoft Store..." to stdout and exits
    # non-zero.  Get-Command finds it; invoking it produces a confusing error
    # that the user sees as our installer crashing.
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $isStoreStub = $false
        try {
            $pythonSource = $pythonCmd.Source
            if ($pythonSource -and $pythonSource -like "*\WindowsApps\*") {
                $isStoreStub = $true
            } else {
                # Even outside WindowsApps, a 0-byte file is the stub
                $item = Get-Item $pythonSource -ErrorAction SilentlyContinue
                if ($item -and $item.Length -eq 0) { $isStoreStub = $true }
            }
        } catch { }

        if (-not $isStoreStub) {
            try {
                $prevEAP2 = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $sysVer = & python --version 2>&1
                $ErrorActionPreference = $prevEAP2
                if ($sysVer -match "Python 3\.(1[0-9]|[1-9][0-9])") {
                    Write-Success "Using system Python: $sysVer"
                    return $true
                }
            } catch {
                if ($prevEAP2) { $ErrorActionPreference = $prevEAP2 }
            }
        }
    }

    Write-Err "Failed to install Python $PythonVersion"
    Write-Info "Install Python 3.11 manually, then re-run this script:"
    Write-Info "  https://www.python.org/downloads/"
    Write-Info "  Or: winget install Python.Python.3.11"
    return $false
}

function Install-Git {
    <#
    .SYNOPSIS
    Ensure Git (and Git Bash) are installed.  Git for Windows bundles bash.exe
    which Hermes uses to run shell commands.

    Priority order (deliberately simple -- no winget, no registry, no system
    package manager):
      1. Existing ``git`` on PATH -- use it as-is (the common fast path).
      2. Download **PortableGit** from the official git-for-windows GitHub
         release (self-extracting 7z.exe) and unpack it to
         ``%LOCALAPPDATA%\hermes\git`` -- never touches system Git, never
         requires admin, works even on locked-down machines and machines
         with a broken system Git install.

    **Why PortableGit, not MinGit:**  MinGit is the minimal-automation
    distribution and ships ONLY ``git.exe`` -- no bash, no POSIX utilities.
    Hermes needs ``bash.exe`` to run shell commands.  PortableGit is the
    full Git for Windows distribution without the installer UI; it ships
    ``git.exe`` + ``bash.exe`` + ``sh``, ``awk``, ``sed``, ``grep``, ``curl``,
    ``ssh``, etc. in ``usr\bin\``.

    We deliberately skip winget because it fails badly when the system Git
    install is in a half-installed state (partially registered, or uninstall-
    blocked).  Owning the Hermes copy of Git ourselves is predictable and
    recoverable: if it ever breaks, ``Remove-Item %LOCALAPPDATA%\hermes\git``
    and re-running this installer fully recovers.

    After install we locate ``bash.exe`` and persist the path in
    ``HERMES_GIT_BASH_PATH`` (User scope) so Hermes can find it in a fresh
    shell without a second PATH refresh.
    #>
    Write-Info "Checking Git..."

    if (Get-Command git -ErrorAction SilentlyContinue) {
        $version = git --version
        Write-Success "Git found ($version)"
        Set-GitBashEnvVar
        return $true
    }

    # Download PortableGit into $HermesHome\git.  Always works as long as
    # we can reach github.com -- no admin, no winget, no reliance on the
    # user's possibly-broken system Git install.
    Write-Info "Git not found -- downloading PortableGit to $HermesHome\git\ ..."
    Write-Info "(no admin rights required; isolated from any system Git install)"

    try {
        $arch = Get-WindowsArch
        if ($arch -eq 'arm64') {
            $assetTag = 'arm64'
            $downloadIsZip = $false
        } elseif ($arch -eq 'x64') {
            $assetTag = '64-bit'
            $downloadIsZip = $false
        } else {
            # PortableGit does not ship 32-bit / arm builds -- fall back to MinGit
            # 32-bit with a warning that bash-based features will be unavailable.
            $assetTag = '32-bit-mingit'
            $downloadIsZip = $true
        }

        # Pinned git-for-windows release. We deliberately do NOT hit
        # api.github.com/repos/.../releases/latest here: that endpoint
        # is rate-limited to 60 requests/hour/IP for unauthenticated
        # callers, and users behind CGNAT / corporate NAT / dorm WiFi
        # routinely hit the limit, breaking the installer.
        # Static github.com/.../releases/download/<tag>/<asset> URLs
        # are not subject to the API rate limit.
        $gitTag    = "v2.54.0.windows.1"
        $gitVer    = "2.54.0"
        $gitVerTag = "$gitVer.windows.1"

        if ($arch -eq "32-bit-mingit") {
            Write-Warn "32-bit Windows detected -- PortableGit is 64-bit only.  Installing MinGit 32-bit as a last resort; bash-dependent Hermes features (terminal tool, agent-browser) will not work on this machine."
            $assetName    = "MinGit-$gitVer-32-bit.zip"
            $downloadIsZip = $true
        } elseif ($arch -eq "arm64") {
            $assetName    = "PortableGit-$gitVer-arm64.7z.exe"
            $downloadIsZip = $false
        } else {
            $assetName    = "PortableGit-$gitVer-64-bit.7z.exe"
            $downloadIsZip = $false
        }

        $downloadUrl = "https://github.com/git-for-windows/git/releases/download/$gitTag/$assetName"
        $downloadExt = if ($downloadIsZip) { "zip" } else { "7z.exe" }
        $tmpFile = "$env:TEMP\$assetName"
        $gitDir = "$HermesHome\git"

        Write-Info "Downloading $assetName (Git for Windows $gitVerTag)..."
        Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpFile -UseBasicParsing

        if (Test-Path $gitDir) {
            Write-Info "Removing previous Git install at $gitDir ..."
            Remove-Item -Recurse -Force $gitDir
        }
        New-Item -ItemType Directory -Path $gitDir -Force | Out-Null

        if ($downloadIsZip) {
            Expand-Archive -Path $tmpFile -DestinationPath $gitDir -Force
        } else {
            # PortableGit is a self-extracting 7z archive.  Invoke it with
            # `-o<target> -y` (silent) to extract to $gitDir.  No 7z install
            # required; it's fully self-contained.
            Write-Info "Extracting PortableGit to $gitDir ..."
            $extractProc = Start-Process -FilePath $tmpFile `
                -ArgumentList "-o`"$gitDir`"", "-y" `
                -NoNewWindow -Wait -PassThru
            if ($extractProc.ExitCode -ne 0) {
                throw "PortableGit extraction failed (exit code $($extractProc.ExitCode))"
            }
        }
        Remove-Item -Force $tmpFile -ErrorAction SilentlyContinue

        # PortableGit layout: cmd\git.exe + bin\bash.exe + usr\bin\ (coreutils)
        # MinGit layout:      cmd\git.exe + usr\bin\bash.exe (if present)
        $gitExe = "$gitDir\cmd\git.exe"
        if (-not (Test-Path $gitExe)) {
            throw "Git extraction did not produce git.exe at $gitExe"
        }

        # Add to session PATH so the rest of this install run can use git.
        $env:Path = "$gitDir\cmd;$env:Path"

        # Persist to User PATH so fresh shells see it.  PortableGit needs
        # cmd\ (for git.exe), bin\ (for bash.exe + core tools), and
        # usr\bin\ (for perl, ssh, curl, and other POSIX coreutils).
        $newPathEntries = @(
            "$gitDir\cmd",
            "$gitDir\bin",
            "$gitDir\usr\bin"
        )
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $userPathItems = if ($userPath) { $userPath -split ";" } else { @() }
        $changed = $false
        foreach ($entry in $newPathEntries) {
            if ($userPathItems -notcontains $entry) {
                $userPathItems += $entry
                $changed = $true
            }
        }
        if ($changed) {
            [Environment]::SetEnvironmentVariable("Path", ($userPathItems -join ";"), "User")
        }

        $version = & $gitExe --version
        Write-Success "Git $version installed to $gitDir (portable, user-scoped)"
        Set-GitBashEnvVar
        return $true
    } catch {
        Write-Err "Could not install portable Git: $_"
        Write-Info ""
        Write-Info "Fallback: install Git manually from https://git-scm.com/download/win"
        Write-Info "then re-run this installer.  Hermes needs Git Bash on Windows to run"
        Write-Info "shell commands (same as Claude Code and other coding agents)."
        return $false
    }
}

function Set-GitBashEnvVar {
    <#
    .SYNOPSIS
    Locate ``bash.exe`` from an already-installed Git and persist the path in
    ``HERMES_GIT_BASH_PATH`` (User env scope) so Hermes can find it even before
    PATH propagation completes in a newly-spawned shell.
    #>
    $candidates = @()

    # Our own portable Git install is ALWAYS checked first, so a broken
    # system Git doesn't hijack us.  If the user had a working system Git
    # we'd have returned early from Install-Git's fast path and never called
    # this with a system-Git-only installation anyway.
    #
    # Layouts:
    #   PortableGit (our default): $HermesHome\git\bin\bash.exe
    #   MinGit (32-bit fallback):  $HermesHome\git\usr\bin\bash.exe
    $candidates += "$HermesHome\git\bin\bash.exe"       # PortableGit layout (primary)
    $candidates += "$HermesHome\git\usr\bin\bash.exe"   # MinGit / PortableGit usr\bin fallback

    # git.exe on PATH can tell us where the install root is
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd) {
        $gitExe = $gitCmd.Source
        # Git for Windows (full installer): <root>\cmd\git.exe + <root>\bin\bash.exe
        # MinGit:                           <root>\cmd\git.exe + <root>\usr\bin\bash.exe
        $gitRoot = Split-Path (Split-Path $gitExe -Parent) -Parent
        $candidates += "$gitRoot\bin\bash.exe"
        $candidates += "$gitRoot\usr\bin\bash.exe"
    }

    # Standard system install locations as a final fallback.  Note:
    # ProgramFiles(x86) can't be referenced via ${env:...} string interpolation
    # because of the parens -- use [Environment]::GetEnvironmentVariable().
    $candidates += "${env:ProgramFiles}\Git\bin\bash.exe"
    $pf86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($pf86) { $candidates += "$pf86\Git\bin\bash.exe" }
    $candidates += "${env:LocalAppData}\Programs\Git\bin\bash.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            [Environment]::SetEnvironmentVariable("HERMES_GIT_BASH_PATH", $candidate, "User")
            $env:HERMES_GIT_BASH_PATH = $candidate
            Write-Info "Set HERMES_GIT_BASH_PATH=$candidate"
            return
        }
    }

    Write-Warn "Could not locate bash.exe -- Hermes may not find Git Bash."
    Write-Info "If needed, set HERMES_GIT_BASH_PATH manually to your bash.exe path."
}

# The desktop build runs Vite ^8, which refuses to start on Node outside
# `^20.19 || >=22.12` -- older Node lacks node:util.styleText, so `vite build`
# crashes with a SyntaxError that surfaces only as the opaque "Build desktop
# app ... exit code 1" install failure. Returns $true when a `node --version`
# string clears that floor.
function Test-NodeVersionOk {
    param([string]$Version)
    try {
        $v = [version]($Version -replace '^v', '' -replace '-.*$', '')
    } catch {
        return $false
    }
    if ($v.Major -eq 20 -and $v.Minor -ge 19) { return $true }
    if ($v.Major -ge 22 -and ($v.Major -gt 22 -or $v.Minor -ge 12)) { return $true }
    return $false
}

function Test-Node {
    Write-Info "Checking Node.js (for browser tools)..."

    if (Get-Command node -ErrorAction SilentlyContinue) {
        $version = node --version
        if (Test-NodeVersionOk $version) {
            Write-Success "Node.js $version found"
            $script:HasNode = $true
            return $true
        }
        Write-Warn "Node.js $version is too old for the desktop build (need ^20.19 or >=22.12)"
    }

    # Prefer a Hermes-managed Node from a previous run over a too-old system one.
    $managedNode = "$HermesHome\node\node.exe"
    if ((Test-Path $managedNode) -and (Test-NodeVersionOk (& $managedNode --version))) {
        $version = & $managedNode --version
        $env:Path = "$HermesHome\node;$env:Path"
        Write-Success "Node.js $version found (Hermes-managed)"
        $script:HasNode = $true
        return $true
    }

    Write-Info "Installing Hermes-managed Node.js $NodeVersion LTS..."

    # Try the portable-zip path FIRST -- no UAC, no admin, no winget MSI.
    # winget install OpenJS.NodeJS.LTS triggers a system-wide MSI install
    # which prompts UAC (the dialog often appears minimized in the taskbar
    # and the install silently waits for consent, looking like a hang).
    # The portable zip path drops node.exe + npm into $HermesHome\node\
    # which is user-scoped and identical to how Install-Git handles
    # PortableGit.  Same UX guarantee: works on locked-down enterprise
    # machines with no admin rights.
    Write-Info "Downloading portable Node.js $NodeVersion to $HermesHome\node\ ..."
    Write-Info "(no admin rights required; isolated from any system Node install)"
    try {
        $arch = Get-WindowsArch
        $indexUrl = "https://nodejs.org/dist/latest-v${NodeVersion}.x/"
        $indexPage = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing
        $zipName = ($indexPage.Content | Select-String -Pattern "node-v${NodeVersion}\.\d+\.\d+-win-${arch}\.zip" -AllMatches).Matches[0].Value

        if ($zipName) {
            $downloadUrl = "${indexUrl}${zipName}"
            $tmpZip = "$env:TEMP\$zipName"
            $tmpDir = "$env:TEMP\hermes-node-extract"

            Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpZip -UseBasicParsing
            if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
            Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force

            $extractedDir = Get-ChildItem $tmpDir -Directory | Select-Object -First 1
            if ($extractedDir) {
                if (Test-Path "$HermesHome\node") { Remove-Item -Recurse -Force "$HermesHome\node" }
                Move-Item $extractedDir.FullName "$HermesHome\node"

                # Session PATH so the rest of this run sees node/npm.
                $env:Path = "$HermesHome\node;$env:Path"

                # Persist to User PATH so fresh shells (and future stages
                # in cross-process driver mode) see it.  Matches the
                # pattern Install-Git uses for PortableGit.
                $nodeDir = "$HermesHome\node"
                $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
                $userPathItems = if ($userPath) { $userPath -split ";" } else { @() }
                if ($userPathItems -notcontains $nodeDir) {
                    $userPathItems += $nodeDir
                    [Environment]::SetEnvironmentVariable("Path", ($userPathItems -join ";"), "User")
                }

                $version = & "$HermesHome\node\node.exe" --version
                Write-Success "Node.js $version installed to $HermesHome\node\ (portable, user-scoped)"
                $script:HasNode = $true

                Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                return $true
            }
        }
    } catch {
        Write-Warn "Portable Node.js download failed: $_"
    }

    # Fallback: try winget (used to be primary, demoted because the MSI
    # install triggers a UAC prompt that frequently appears minimized in
    # the taskbar -- looks like a hang to users on stock Windows).
    # Kept for environments where the portable download fails (proxy,
    # locked firewall, etc.) but the user is willing to consent to UAC.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Falling back to winget (may prompt UAC -- check your taskbar for a flashing icon)..."
        # Capture EAP outside the try block so the catch's restore call always
        # has a meaningful value (see Install-Uv for the full rationale).
        $prevEAP = $ErrorActionPreference
        try {
            # Relax EAP=Stop so stderr lines from winget don't get wrapped
            # as ErrorRecords and short-circuit the 2>&1 pipe before we can
            # check the post-condition.  See the long comment in Install-Uv
            # for the same pattern.
            $ErrorActionPreference = "Continue"
            # On ARM64, force winget to fetch the ARM64 installer.  Without
            # the explicit override, winget on WoW64 sometimes still resolves
            # to x64 manifests, leaving us with an emulated Node toolchain
            # even after a "successful" install.  The OpenJS manifest does
            # publish an arm64 installer, so this is safe.
            $wingetArgs = @(
                'install','OpenJS.NodeJS.LTS','--silent',
                '--accept-package-agreements','--accept-source-agreements'
            )
            if ((Get-WindowsArch) -eq 'arm64') {
                $wingetArgs += @('--architecture','arm64')
            }
            winget @wingetArgs 2>&1 | Out-Null
            $ErrorActionPreference = $prevEAP
            # Refresh PATH
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
            if (Get-Command node -ErrorAction SilentlyContinue) {
                $version = node --version
                Write-Success "Node.js $version installed via winget"
                $script:HasNode = $true
                return $true
            }
        } catch {
            if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        }
    }


    Write-Info "Install manually: https://nodejs.org/en/download/"
    $script:HasNode = $false
    return $true
}

function Update-ProcessPathForPackages {
    # Make freshly-installed shims (rg.exe, ffmpeg.exe) visible to Get-Command in
    # THIS process without spawning a new shell, by folding the persisted
    # User+Machine hives plus winget's alias-shim directory into $env:Path.
    # Called after every package-manager attempt (winget/choco/scoop): previously
    # PATH was only refreshed inside the winget branch, so a successful
    # choco/scoop fallback -- or any install on a box without winget -- could be
    # misreported as "not installed".
    #
    # MERGE rather than overwrite: start from the existing process PATH so any
    # process-only entries added earlier in this installer run survive, then
    # APPEND hive/winget-Links entries not already present (case-insensitive,
    # order-preserving dedupe). A wholesale replace would silently drop those
    # process-only entries.
    $candidates = @()
    $candidates += $env:Path
    $candidates += [Environment]::GetEnvironmentVariable("Path", "User")
    $candidates += [Environment]::GetEnvironmentVariable("Path", "Machine")
    $wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
    if (Test-Path $wingetLinks) {
        $candidates += $wingetLinks
    }
    $seen = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
    $ordered = New-Object System.Collections.Generic.List[string]
    foreach ($chunk in $candidates) {
        if ([string]::IsNullOrEmpty($chunk)) { continue }
        foreach ($entry in $chunk.Split(';')) {
            $trimmed = $entry.Trim()
            if ($trimmed -and $seen.Add($trimmed)) {
                $ordered.Add($trimmed)
            }
        }
    }
    $env:Path = [string]::Join(';', $ordered)
}

function Install-SystemPackages {
    $script:HasRipgrep = $false
    $script:HasFfmpeg = $false
    $needRipgrep = $false
    $needFfmpeg = $false

    Write-Info "Checking ripgrep (fast file search)..."
    if (Get-Command rg -ErrorAction SilentlyContinue) {
        $version = rg --version | Select-Object -First 1
        Write-Success "$version found"
        $script:HasRipgrep = $true
    } else {
        $needRipgrep = $true
    }

    Write-Info "Checking ffmpeg (TTS voice messages)..."
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Success "ffmpeg found"
        $script:HasFfmpeg = $true
    } else {
        $needFfmpeg = $true
    }

    if (-not $needRipgrep -and -not $needFfmpeg) { return }

    # Build description and package lists for each package manager
    $descParts = @()
    $wingetPkgs = @()
    $chocoPkgs = @()
    $scoopPkgs = @()

    if ($needRipgrep) {
        $descParts += "ripgrep for faster file search"
        $wingetPkgs += "BurntSushi.ripgrep.MSVC"
        $chocoPkgs += "ripgrep"
        $scoopPkgs += "ripgrep"
    }
    if ($needFfmpeg) {
        $descParts += "ffmpeg for TTS voice messages"
        $wingetPkgs += "Gyan.FFmpeg"
        $chocoPkgs += "ffmpeg"
        $scoopPkgs += "ffmpeg"
    }

    $description = $descParts -join " and "
    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    $hasChoco = Get-Command choco -ErrorAction SilentlyContinue
    $hasScoop = Get-Command scoop -ErrorAction SilentlyContinue

    # Try winget first (most common on modern Windows)
    if ($hasWinget) {
        Write-Info "Installing $description via winget..."
        # Per-package log paths -- key the lookup by package id so we can
        # decide AFTER the post-install Get-Command check whether to keep
        # the log (still missing -> keep as breadcrumb) or delete it (now
        # present -> happy path, no clutter).
        $pkgLogs = @{}
        foreach ($pkg in $wingetPkgs) {
            $log = "$env:TEMP\hermes-winget-$($pkg -replace '[^A-Za-z0-9]','_')-$(Get-Random).log"
            $pkgLogs[$pkg] = $log
            # --source winget pins us to the github-backed source.  Without this,
            # a broken msstore source (cert validation failures like 0x8a15005e
            # are common on Windows-on-ARM and some corporate networks) makes
            # winget bail with "please specify --source" *before* attempting any
            # install -- and it exits 0, so the surrounding try/catch never fires.
            # We don't ship anything from msstore, so pinning is safe.
            try {
                $output = winget install --exact --id $pkg --source winget --silent `
                    --accept-package-agreements --accept-source-agreements 2>&1
                $code = $LASTEXITCODE
                $output | Out-File -FilePath $log -Encoding utf8
                "winget exit: $code" | Out-File -FilePath $log -Encoding utf8 -Append
                # 0x8A15002B (-1978335189) = APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE.
                # winget treats `install` on a package it already has registered as
                # an *upgrade*, finds no newer version, and bails with this code --
                # even when the binary is gone from disk/PATH (stale registration,
                # files removed outside winget, or a missing alias shim). We KNOW the
                # command was missing (that's why we're here), so a plain install
                # dead-ends forever. Force a reinstall to repair the registration so
                # the shim reappears.
                if ($code -eq -1978335189) {
                    "-> already-installed/no-upgrade; retrying with --force" | Out-File -FilePath $log -Encoding utf8 -Append
                    $output = winget install --exact --id $pkg --source winget --silent --force `
                        --accept-package-agreements --accept-source-agreements 2>&1
                    $output | Out-File -FilePath $log -Encoding utf8 -Append
                    "winget exit (force): $LASTEXITCODE" | Out-File -FilePath $log -Encoding utf8 -Append
                }
            } catch {
                $_ | Out-File -FilePath $log -Encoding utf8 -Append
                "winget exit: <exception>" | Out-File -FilePath $log -Encoding utf8 -Append
            }
        }
        # Refresh PATH so packages winget exposed via "command line aliases" in
        # %LOCALAPPDATA%\Microsoft\WinGet\Links (added to PATH only in
        # newly-spawned shells, not this process) are visible to Get-Command below.
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed"
            $script:HasRipgrep = $true
            $needRipgrep = $false
            Remove-Item -Path $pkgLogs["BurntSushi.ripgrep.MSVC"] -ErrorAction SilentlyContinue
        } elseif ($pkgLogs.ContainsKey("BurntSushi.ripgrep.MSVC")) {
            Write-Warn "winget could not install ripgrep; details: $($pkgLogs['BurntSushi.ripgrep.MSVC'])"
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
            Remove-Item -Path $pkgLogs["Gyan.FFmpeg"] -ErrorAction SilentlyContinue
        } elseif ($pkgLogs.ContainsKey("Gyan.FFmpeg")) {
            Write-Warn "winget could not install ffmpeg; details: $($pkgLogs['Gyan.FFmpeg'])"
        }
        if (-not $needRipgrep -and -not $needFfmpeg) { return }
    }

    # Fallback: choco
    if ($hasChoco -and ($needRipgrep -or $needFfmpeg)) {
        Write-Info "Trying Chocolatey..."
        foreach ($pkg in $chocoPkgs) {
            try { choco install $pkg -y 2>&1 | Out-Null } catch { }
        }
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed via chocolatey"
            $script:HasRipgrep = $true
            $needRipgrep = $false
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed via chocolatey"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
        }
    }

    # Fallback: scoop
    if ($hasScoop -and ($needRipgrep -or $needFfmpeg)) {
        Write-Info "Trying Scoop..."
        foreach ($pkg in $scoopPkgs) {
            try { scoop install $pkg 2>&1 | Out-Null } catch { }
        }
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed via scoop"
            $script:HasRipgrep = $true
            $needRipgrep = $false
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed via scoop"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
        }
    }

    # Show manual instructions for anything still missing
    if ($needRipgrep) {
        Write-Warn "ripgrep not installed (file search will use findstr fallback)"
        Write-Info "  winget install BurntSushi.ripgrep.MSVC"
    }
    if ($needFfmpeg) {
        Write-Warn "ffmpeg not installed (TTS voice messages will be limited)"
        Write-Info "  winget install Gyan.FFmpeg"
    }
}

# ============================================================================
# Installation
# ============================================================================

function Install-Repository {
    Write-Info "Installing to $InstallDir..."

    $didUpdate = $false

    if (Test-Path $InstallDir) {
        # Test-Path "$InstallDir\.git" returns True when .git is a file OR a
        # directory OR a symlink OR a submodule-style gitfile -- and also when
        # it's a broken stub left over from a failed previous install (e.g.
        # a partial Remove-Item that couldn't delete a locked index.lock).
        # Validate the repo properly by asking git itself.  Three checks
        # belt-and-braces: rev-parse (work tree), git status, and a resolvable
        # HEAD (an initial commit).  If any fails the repo is broken and we
        # fall through to a fresh clone.
        $repoValid = $false
        if (Test-Path "$InstallDir\.git") {
            Push-Location $InstallDir
            try {
                # Reset $LASTEXITCODE before the probe so we don't pick up
                # a stale 0 from an earlier git call in this session.
                $global:LASTEXITCODE = 0
                $revParseOut = & git -c windows.appendAtomically=false rev-parse --is-inside-work-tree 2>&1
                $revParseOk = ($LASTEXITCODE -eq 0) -and ($revParseOut -match "true")

                $global:LASTEXITCODE = 0
                $null = & git -c windows.appendAtomically=false status --short 2>&1
                $statusOk = ($LASTEXITCODE -eq 0)

                # An interrupted previous clone leaves a repo with NO initial
                # commit. rev-parse/status still succeed there, but the update
                # path's `git stash` (and later `git checkout`) abort with
                # "You do not have the initial commit yet" and fail the install
                # (#40998). Require a resolvable HEAD so such partial checkouts
                # are treated as broken and re-cloned fresh below.
                $global:LASTEXITCODE = 0
                $null = & git -c windows.appendAtomically=false rev-parse --verify HEAD 2>&1
                $hasCommit = ($LASTEXITCODE -eq 0)

                if ($revParseOk -and $statusOk -and $hasCommit) {
                    $repoValid = $true
                }
            } catch {}
            Pop-Location
        }

        if ($repoValid) {
            Write-Info "Existing installation found, updating..."
            Push-Location $InstallDir
            # Wrap the entire fetch+checkout block in EAP=Continue so git's
            # routine stderr output (e.g. 'From <url>' info lines emitted by
            # `git fetch`) doesn't terminate the script under the global
            # EAP=Stop.  We rely on $LASTEXITCODE for actual failures.
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $autostashRef = ""
            try {
                # This is a MANAGED checkout, not a repo the user edits. Git for
                # Windows defaults to core.autocrlf=true, which renormalizes the
                # repo's LF-only text files to CRLF in the working tree -- so
                # tracked files (.envrc, AGENTS.md, agent/*.py, workflows, ...)
                # show as locally modified even though nobody touched them. A
                # bare `git checkout` then aborts with "Your local changes would
                # be overwritten by checkout", which is exactly the failure GUI
                # users hit on update. Pin autocrlf=false so the dirt is never
                # created in the first place.
                git -c windows.appendAtomically=false config core.autocrlf false 2>$null
                Discard-LockfileChurn $InstallDir
                # Preserve any real local changes before the checkout instead of
                # discarding them with `reset --hard HEAD`. The old hard reset
                # silently destroyed agent-edited source on managed clones (the
                # #38542 data-loss class). Stash + restore mirrors install.sh:
                # nothing is lost, and a failed restore leaves the work in a
                # git stash for manual recovery. Untracked files are included so
                # agent-created dirs (e.g. tinker-atropos/) survive too.
                $statusOut = git -c windows.appendAtomically=false status --porcelain 2>$null
                if (-not [string]::IsNullOrWhiteSpace(($statusOut -join "`n"))) {
                    # A previously interrupted update can leave the index with
                    # unmerged entries. In that state `git stash` aborts with
                    # "could not write index" and the following `git checkout`
                    # aborts with "you need to resolve your current index first"
                    # -- the GUI "git checkout main failed (exit 1)" install
                    # failure. Clear the conflict markers with `git reset` first:
                    # working-tree changes are kept (and stashed just below); only
                    # the index conflict state is dropped. Mirrors the `hermes
                    # update` path (#4735).
                    $unmergedOut = git -c windows.appendAtomically=false ls-files --unmerged 2>$null
                    if (-not [string]::IsNullOrWhiteSpace(($unmergedOut -join "`n"))) {
                        Write-Info "Clearing unmerged index entries from a previous conflict..."
                        git -c windows.appendAtomically=false reset -q 2>$null
                    }
                    $stashName = "hermes-install-autostash-" + (Get-Date -Format "yyyyMMdd-HHmmss")
                    Write-Info "Local changes detected, stashing before update..."
                    git -c windows.appendAtomically=false stash push --include-untracked -m "$stashName"
                    if ($LASTEXITCODE -eq 0) { $autostashRef = "stash@{0}" }
                }
                git -c windows.appendAtomically=false fetch origin $Branch
                if ($LASTEXITCODE -ne 0) { throw "git fetch failed (exit $LASTEXITCODE)" }
                # Precedence: Commit > Tag > Branch.  Commit and Tag check
                # out as detached HEAD intentionally -- they're meant to be
                # reproducible pins, not branches the user pulls into.
                if ($Commit) {
                    # Make sure we have the commit locally (a tag-less commit
                    # SHA isn't always reachable from any one branch fetch).
                    git -c windows.appendAtomically=false fetch origin $Commit
                    git -c windows.appendAtomically=false checkout --detach $Commit
                    if ($LASTEXITCODE -ne 0) { throw "git checkout $Commit failed (exit $LASTEXITCODE)" }
                } elseif ($Tag) {
                    git -c windows.appendAtomically=false fetch origin "refs/tags/${Tag}:refs/tags/${Tag}"
                    git -c windows.appendAtomically=false checkout --detach "refs/tags/$Tag"
                    if ($LASTEXITCODE -ne 0) { throw "git checkout tag $Tag failed (exit $LASTEXITCODE)" }
                } else {
                    git -c windows.appendAtomically=false checkout $Branch
                    if ($LASTEXITCODE -ne 0) { throw "git checkout $Branch failed (exit $LASTEXITCODE)" }
                    # Managed installs should follow origin/$Branch exactly. If
                    # the checkout has diverged (or has local-only commits),
                    # ff-only pull cannot succeed — mirror ``hermes update`` and
                    # reset to the fetched remote so bootstrap/install can recover.
                    git -c windows.appendAtomically=false pull --ff-only origin $Branch
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warn "Fast-forward not possible; resetting managed install to origin/$Branch..."
                        git -c windows.appendAtomically=false reset --hard "origin/$Branch"
                        if ($LASTEXITCODE -ne 0) { throw "git reset --hard origin/$Branch failed (exit $LASTEXITCODE)" }
                    }
                }

                if ($autostashRef) {
                    # Default to restoring so work is never silently dropped.
                    # Only prompt when we're certain a human can answer: an
                    # interactive session AND a real, non-redirected console on
                    # both stdin and stdout. The desktop "Update" button and
                    # bootstrap run the installer without a usable console -- in
                    # those cases Read-Host would hang or return empty, so we
                    # skip the prompt and just restore (the safe default).
                    $restoreNow = $true
                    $hasConsole = $false
                    try {
                        $hasConsole = (
                            [Environment]::UserInteractive `
                            -and (-not [Console]::IsInputRedirected) `
                            -and (-not [Console]::IsOutputRedirected) `
                            -and ($Host.Name -eq "ConsoleHost")
                        )
                    } catch { $hasConsole = $false }
                    if ($hasConsole) {
                        Write-Warn "Local changes were stashed before updating."
                        Write-Warn "Restoring them may reapply local customizations onto the updated codebase."
                        $restoreAnswer = Read-Host "Restore local changes now? [Y/n]"
                        if ($restoreAnswer -match '^(n|no)$') { $restoreNow = $false }
                    }

                    if ($restoreNow) {
                        Write-Info "Restoring local changes..."
                        git -c windows.appendAtomically=false stash apply $autostashRef
                        if ($LASTEXITCODE -eq 0) {
                            git -c windows.appendAtomically=false stash drop $autostashRef 2>$null
                            Write-Warn "Local changes were restored on top of the updated codebase."
                            Write-Warn "Review git diff / git status if Hermes behaves unexpectedly."
                        } else {
                            Write-Err "Update succeeded, but restoring local changes failed. Your changes are still preserved in git stash."
                            Write-Info "Resolve manually with: git stash apply $autostashRef"
                            throw "git stash apply failed after update"
                        }
                    } else {
                        Write-Info "Skipped restoring local changes."
                        Write-Info "Your changes are still preserved in git stash."
                        Write-Info "Restore manually with: git stash apply $autostashRef"
                    }
                    $autostashRef = ""
                }
            } finally {
                if ($autostashRef) {
                    # We stashed but never reached the restore block (a fetch/
                    # checkout/pull failure threw). Leave the stash in place and
                    # tell the user how to recover it -- never silently drop it.
                    Write-Warn "Update did not complete. Your local changes are preserved in git stash."
                    Write-Info "Restore manually with: git stash apply $autostashRef"
                }
                $ErrorActionPreference = $prevEAP
                Pop-Location
            }
            $didUpdate = $true
        } else {
            # Directory exists but isn't a usable git repo -- e.g. an
            # interrupted clone with no initial commit (#40998), or a leftover
            # ``.git`` stub from a partial uninstall that used to lock the
            # installer into the "update" branch forever. Move it aside rather
            # than deleting it -- never destroy a directory the user might still
            # want -- and fall through to a fresh clone.
            $backupDir = "$InstallDir.broken-" + (Get-Date -Format "yyyyMMdd-HHmmss")
            Write-Warn "Existing directory at $InstallDir is not a valid git repo."
            Write-Warn "Moving it aside to $backupDir before re-cloning."
            try {
                Move-Item -LiteralPath $InstallDir -Destination $backupDir -ErrorAction Stop
            } catch {
                Write-Err "Could not move $InstallDir aside : $_"
                Write-Info "Close any programs that might be using files in $InstallDir (editors,"
                Write-Info "terminals, running hermes processes) and try again."
                throw
            }
        }
    }

    if (-not $didUpdate) {
        $cloneSuccess = $false

        # Fix Windows git "copy-fd: write returned: Invalid argument" error.
        # Git for Windows can fail on atomic file operations (hook templates,
        # config lock files) due to antivirus, OneDrive, or NTFS filter drivers.
        # The -c flag injects config before any file I/O occurs.
        Write-Info "Configuring git for Windows compatibility..."
        $env:GIT_CONFIG_COUNT = "1"
        $env:GIT_CONFIG_KEY_0 = "windows.appendAtomically"
        $env:GIT_CONFIG_VALUE_0 = "false"
        git config --global windows.appendAtomically false 2>$null

        # Try SSH first, then HTTPS, with -c flag for atomic write fix
        Write-Info "Trying SSH clone..."
        $env:GIT_SSH_COMMAND = "ssh -o BatchMode=yes -o ConnectTimeout=5"
        try {
            Invoke-NativeWithRelaxedErrorAction { git -c windows.appendAtomically=false clone --depth 1 --branch $Branch $RepoUrlSsh $InstallDir }
            if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
        } catch { }
        $env:GIT_SSH_COMMAND = $null

        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Info "SSH failed, trying HTTPS..."
            try {
                Invoke-NativeWithRelaxedErrorAction { git -c windows.appendAtomically=false clone --depth 1 --branch $Branch $RepoUrlHttps $InstallDir }
                if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
            } catch { }
        }

        # Fallback: download ZIP archive (bypasses git file I/O issues entirely)
        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Warn "Git clone failed -- downloading ZIP archive instead..."
            try {
                # Pick the ZIP URL for the most-specific ref the caller asked
                # for.  GitHub supports archive URLs for commits, tags, and
                # branches; we honour Commit > Tag > Branch.
                if ($Commit) {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/$Commit.zip"
                    $zipLabel = $Commit
                } elseif ($Tag) {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/refs/tags/$Tag.zip"
                    $zipLabel = $Tag
                } else {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/refs/heads/$Branch.zip"
                    $zipLabel = $Branch
                }
                $zipPath = "$env:TEMP\hermes-agent-$zipLabel.zip"
                $extractPath = "$env:TEMP\hermes-agent-extract"

                Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
                if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
                Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

                # GitHub ZIPs extract to repo-branch/ subdirectory
                $extractedDir = Get-ChildItem $extractPath -Directory | Select-Object -First 1
                if ($extractedDir) {
                    New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) -ErrorAction SilentlyContinue | Out-Null
                    Move-Item $extractedDir.FullName $InstallDir -Force
                    Write-Success "Downloaded and extracted"

                    # Initialize git repo so updates work later
                    Push-Location $InstallDir
                    git -c windows.appendAtomically=false init 2>$null
                    git -c windows.appendAtomically=false config windows.appendAtomically false 2>$null
                    git remote add origin $RepoUrlHttps 2>$null
                    Pop-Location
                    Write-Success "Git repo initialized for future updates"

                    $cloneSuccess = $true
                }

                # Cleanup temp files
                Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
                Remove-Item -Recurse -Force $extractPath -ErrorAction SilentlyContinue
            } catch {
                Write-Err "ZIP download also failed: $_"
            }
        }

        if (-not $cloneSuccess) {
            throw "Failed to download repository (tried git clone SSH, HTTPS, and ZIP)"
        }
    }

    # Set per-repo config (harmless if it fails)
    Push-Location $InstallDir
    git -c windows.appendAtomically=false config windows.appendAtomically false 2>$null
    # Pin autocrlf=false on the managed clone so git never renormalizes the
    # repo's LF text files to CRLF in the working tree. Without this, the very
    # next `hermes update` checkout aborts on a "dirty" tree the user never
    # touched (see the update path above).
    git -c windows.appendAtomically=false config core.autocrlf false 2>$null

    # Post-clone pin: when a clone (or ZIP-fallback init) just landed us on
    # $Branch's tip, honour the higher-precedence $Commit / $Tag by checking
    # the exact ref out as a detached HEAD.  Skipped for the in-place update
    # path (above) since that already routed via the same precedence.
    if (-not $didUpdate) {
        # Same EAP=Continue wrap as the update path -- git fetch's 'From <url>'
        # info line goes to stderr and would terminate the script under the
        # global EAP=Stop otherwise.  We check $LASTEXITCODE for real errors.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($Commit) {
                Write-Info "Pinning to commit $Commit..."
                git -c windows.appendAtomically=false fetch origin $Commit
                git -c windows.appendAtomically=false checkout --detach $Commit
                if ($LASTEXITCODE -ne 0) {
                    throw "git checkout $Commit failed (exit $LASTEXITCODE)"
                }
            } elseif ($Tag) {
                Write-Info "Pinning to tag $Tag..."
                git -c windows.appendAtomically=false fetch origin "refs/tags/${Tag}:refs/tags/${Tag}"
                git -c windows.appendAtomically=false checkout --detach "refs/tags/$Tag"
                if ($LASTEXITCODE -ne 0) {
                    throw "git checkout tag $Tag failed (exit $LASTEXITCODE)"
                }
            }
        } finally {
            $ErrorActionPreference = $prevEAP
        }
    }

    Write-Success "Repository ready"
}

function Install-Venv {
    if ($NoVenv) {
        Write-Info "Skipping virtual environment (-NoVenv)"
        return
    }

    # Re-resolve the interpreter before creating the venv.  Under Hermes-Setup.exe
    # each stage runs in its own powershell.exe, so the fallback the `python`
    # stage picked (e.g. 3.12 when 3.11 is absent) did NOT propagate into this
    # fresh process -- $PythonVersion is back at its "3.11" default.  Trusting it
    # here made `uv venv venv --python 3.11` fail with exit 2 on machines without
    # 3.11 even though the `python` stage reported success (issue #50769).
    $resolved = Resolve-AvailablePythonVersion
    if ($resolved -and $resolved -ne $PythonVersion) {
        Write-Info "Python $PythonVersion not available; using detected Python $resolved"
        $script:PythonVersion = $resolved
    }

    Write-Info "Creating virtual environment with Python $PythonVersion..."
    
    Push-Location $InstallDir
    
    if (Test-Path "venv") {
        Write-Info "Virtual environment already exists, recreating..."
        # On Windows, native Python extensions (e.g. _bcrypt.pyd, tornado's
        # speedups.pyd) are loaded as DLLs by any running hermes process.
        # Windows denies deletion of loaded DLLs, so every process running out
        # of this venv must be stopped before removing it -- otherwise
        # Remove-Item fails with "Access to the path '...' is denied" and the
        # whole install/update aborts at this stage.
        if ($env:OS -eq "Windows_NT") {
            $myPid = $PID
            Write-Info "Stopping any running hermes processes before recreating venv..."
            # The launcher CLI (hermes.exe) plus its child tree.
            & taskkill /F /T /IM hermes.exe /FI "PID ne $myPid" 2>$null | Out-Null
            # taskkill /IM hermes.exe is NOT enough: the gateway/agent that a
            # scheduled task or watchdog autostarts runs as
            # `pythonw.exe -m hermes_cli.main gateway run` straight out of
            # venv\Scripts\, so its image name is python/pythonw, not hermes.exe.
            # That process holds the venv's .pyd files open and re-triggers the
            # access-denied failure. Stop anything whose executable lives under
            # this venv, matched by path prefix so the image name does not matter
            # and a global/system python outside the venv is never touched.
            #
            # The gateway autostart task registers with /RL LIMITED as the current
            # user (see hermes_cli/gateway_windows.py), so the installer always
            # runs at equal-or-higher integrity and can read its executable path.
            # Get-CimInstance is used over Get-Process because it returns a null
            # ExecutablePath for a process it cannot inspect (a different session)
            # instead of throwing, so an unreadable process is skipped rather than
            # aborting the whole sweep.
            $venvPrefix = [System.IO.Path]::GetFullPath((Join-Path $InstallDir "venv")).TrimEnd('\') + '\'
            try {
                Get-CimInstance Win32_Process -ErrorAction Stop |
                    Where-Object { $_.ProcessId -ne $myPid -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase) } |
                    ForEach-Object {
                        Write-Info "  stopping PID $($_.ProcessId) ($($_.Name)) running from venv"
                        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                    }
            } catch {
                Write-Warn "Could not enumerate venv processes: $($_.Exception.Message)"
            }
            Start-Sleep -Milliseconds 800
        }
        Remove-Item -Recurse -Force "venv" -ErrorAction SilentlyContinue
        # A killed process can take a moment to release its file handles, so a
        # first Remove-Item may still hit a locked .pyd. Retry once after a short
        # pause before giving up and letting the stage fail loudly.
        if (Test-Path "venv") {
            Start-Sleep -Seconds 2
            Remove-Item -Recurse -Force "venv"
        }
    }
    
    # uv creates the venv and pins the Python version in one step.  uv emits
    # normal progress such as "Using CPython ..." on stderr; under Windows
    # PowerShell 5.1 with EAP=Stop that stderr is a NativeCommandError unless
    # we temporarily relax EAP and trust $LASTEXITCODE for real failures.
    Invoke-NativeWithRelaxedErrorAction { & $UvCmd venv venv --python $PythonVersion }
    # Relaxing EAP above means a *genuine* uv-venv failure (exit != 0) no longer
    # aborts on its own. Capture $LASTEXITCODE immediately and fail fast, so the
    # `venv` stage can't falsely report success (and Invoke-Stage can't emit
    # ok=true) when the venv was never created.
    $venvExitCode = $LASTEXITCODE
    if ($venvExitCode -ne 0) {
        Pop-Location
        throw "Failed to create virtual environment (uv venv exited with $venvExitCode)"
    }

    # Neutralize any inherited UV_PYTHON (e.g. $env:UV_PYTHON = "3.14" left in
    # the user's shell). uv honours UV_PYTHON over an existing venv for the
    # later `uv sync` / `uv pip install` tiers, so without this it would
    # silently delete this 3.11 venv and recreate it at the inherited version
    # -- building Rust transitives that have no wheel for that version from
    # source via maturin, which fails. Pinning UV_PYTHON to the interpreter we
    # just created forces every subsequent uv command onto it.
    $venvPythonExe = Join-Path $InstallDir "venv\Scripts\python.exe"
    if (Test-Path $venvPythonExe) {
        $env:UV_PYTHON = $venvPythonExe
    }

    Pop-Location
    
    Write-Success "Virtual environment ready (Python $PythonVersion)"
}

function Install-Dependencies {
    Write-Info "Installing dependencies..."
    
    Push-Location $InstallDir
    
    if (-not $NoVenv) {
        # Tell uv to install into our venv (no activation needed)
        $env:VIRTUAL_ENV = "$InstallDir\venv"
    }

    # Re-pin UV_PYTHON to the venv interpreter. Install-Venv already does this,
    # but the bootstrap runs install stages (venv, python-deps) as separate
    # processes, so the env var set in Install-Venv does NOT survive into a
    # separate python-deps invocation. Re-deriving it here covers that path.
    # Without it, an inherited $env:UV_PYTHON = "3.14" makes the uv sync/pip
    # tiers below recreate the venv at 3.14 and fail the maturin source build
    # (no cp314 wheels yet).
    if (-not $NoVenv) {
        $venvPythonExe = Join-Path $InstallDir "venv\Scripts\python.exe"
        if (Test-Path $venvPythonExe) {
            $env:UV_PYTHON = $venvPythonExe
        }
    }

    # Hash-verified install (Tier 0) -- when uv.lock is present, prefer
    # `uv sync --locked`. The lockfile records SHA256 hashes for every
    # transitive dependency, so a compromised transitive (different hash
    # than what we shipped) is REJECTED by the resolver. This is the
    # *only* path that protects against the "direct dep is fine, but the
    # dep's dep got worm-poisoned overnight" failure mode. The
    # `uv pip install` tiers below re-resolve transitives fresh from PyPI
    # without any hash verification -- they exist to keep installs working
    # when the lockfile is stale, missing, or out-of-sync with the
    # current extras spec, NOT because they're equivalent in posture.
    if (Test-Path "uv.lock") {
        Write-Info "Trying tier: hash-verified (uv.lock) ..."
        # Critical flag choice: `--extra all`, NOT `--all-extras`.
        #   --all-extras = every [project.optional-dependencies] key,
        #                  bypassing the curated [all] extra. On Windows
        #                  that means [matrix] -> python-olm (no wheel,
        #                  needs `make` to build from sdist) and the
        #                  install fails.
        #   --extra all  = just the [all] extra's contents (curated).
        #
        # UV_PROJECT_ENVIRONMENT pins the sync target to our venv\.
        # Without it, modern uv (>=0.5) ignores VIRTUAL_ENV for `sync`
        # and creates a sibling .venv\ inside the repo -- leaving venv\
        # empty and producing the broken state where `hermes.exe` exists
        # in the wrong directory and imports fail with ModuleNotFoundError.
        # (Mirrors the same flag in scripts/install.sh::install_deps.)
        $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd sync --extra all --locked }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Main package installed (hash-verified via uv.lock)"
            $script:InstalledTier = "hash-verified (uv.lock)"
            # Skip the rest of the tiered cascade -- we already have a
            # complete, hash-verified install.
            $skipPipFallback = $true
        } else {
            Write-Warn "uv.lock sync failed (lockfile may be stale), falling back to PyPI resolve..."
            $skipPipFallback = $false
        }
    } else {
        Write-Info "uv.lock not found -- falling back to PyPI resolve (no hash verification)"
        $skipPipFallback = $false
    }

    # Install main package.  Tiered fallback so a single flaky transitive
    # doesn't silently drop everything.  Each tier's stdout/stderr is
    # preserved -- no Out-Null swallowing -- so the user can see what failed.
    #
    # Tier 1: [all] -- the curated extra in pyproject.toml.
    # Tier 2: [all] minus the currently-broken extras list ($brokenExtras).
    #         Edit $brokenExtras below when something on PyPI breaks; this
    #         lets users keep the rest of [all] when one transitive is
    #         unavailable. The list of [all]'s contents is parsed from
    #         pyproject.toml at runtime -- there is NO hand-mirrored copy
    #         to drift out of sync.
    # Tier 3: bare `.` -- last-resort so at least the core CLI launches.

    # Currently-broken extras. Edit this list when an upstream package
    # gets quarantined / yanked / breaks resolution. Empty means everything
    # in [all] should be installable; populate with the names of extras
    # whose deps are temporarily unavailable.
    $brokenExtras = @()

    # Parse [project.optional-dependencies].all from pyproject.toml.
    # tomllib is stdlib on Python 3.11+ which the bootstrap guarantees.
    $pythonExeForParse = if (-not $NoVenv) { "$InstallDir\venv\Scripts\python.exe" } else { (& $UvCmd python find $PythonVersion) }
    $allExtras = @()
    if (Test-Path $pythonExeForParse) {
        $parsed = & $pythonExeForParse -c @"
import re, sys, tomllib
try:
    with open('pyproject.toml', 'rb') as fh:
        data = tomllib.load(fh)
    specs = data['project']['optional-dependencies']['all']
    out = []
    for s in specs:
        m = re.search(r'hermes-agent\[([\w-]+)\]', s)
        if m: out.append(m.group(1))
    print(','.join(out))
except Exception:
    sys.exit(1)
"@ 2>$null
        if ($LASTEXITCODE -eq 0 -and $parsed) {
            $allExtras = $parsed.Trim().Split(',')
        }
    }
    if (-not $allExtras -or $allExtras.Count -eq 0) {
        Write-Warn "Could not parse [all] from pyproject.toml; Tier 2 will be a no-op."
        $safeAll = "all"
    } else {
        $safeAll = ($allExtras | Where-Object { $brokenExtras -notcontains $_ }) -join ","
    }
    $brokenLabel = if ($brokenExtras) { ($brokenExtras -join ", ") } else { "none" }

    $installTiers = @(
        @{ Name = "all"; Spec = ".[all]" },
        @{ Name = "all minus known-broken ($brokenLabel)"; Spec = ".[$safeAll]" },
        @{ Name = "core only (no extras)"; Spec = "." }
    )
    $installed = $skipPipFallback
    if (-not $skipPipFallback) {
        foreach ($tier in $installTiers) {
        Write-Info "Trying tier: $($tier.Name) ..."
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install -e $tier.Spec }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Main package installed ($($tier.Name))"
            $script:InstalledTier = $tier.Name
            $installed = $true
            break
        }
        Write-Warn "Tier '$($tier.Name)' failed (exit $LASTEXITCODE). Trying next tier..."
        }
    }
    if (-not $installed) {
        throw "Failed to install hermes-agent package even with no extras. Inspect the uv pip install output above."
    }

    # Baseline-import gate. Even if a tier reported success above, the
    # actual deps may have landed somewhere other than $InstallDir\venv\
    # (e.g. uv 0.5+ syncing into a sibling .venv\ when UV_PROJECT_ENVIRONMENT
    # isn't set, leaving venv\ empty and hermes.exe broken with
    # `ModuleNotFoundError: No module named 'dotenv'` on first run).
    # We probe via the venv's own python so a misdirected sync is caught
    # here, not 30 seconds later when the user runs `hermes`.
    if (-not $NoVenv) {
        $venvPython = "$InstallDir\venv\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            throw "Install reported success but $venvPython does not exist. The dependency sync likely landed in a sibling .venv\ directory. Re-run the installer; if it persists, manually: cd '$InstallDir'; Remove-Item -Recurse -Force venv,.venv; uv venv venv --python $PythonVersion; `$env:UV_PROJECT_ENVIRONMENT='$InstallDir\venv'; uv sync --extra all --locked"
        }
        # Relax EAP=Stop while running the import probe.  Python writes
        # deprecation warnings and import-system info to stderr; under
        # EAP=Stop the 2>&1 merge wraps those as ErrorRecord objects and
        # throws even when the imports succeed.  $LASTEXITCODE is the
        # reliable signal (it's 0 iff the python invocation exited 0,
        # regardless of what was written to stderr).
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $venvPython -c "import dotenv, openai, rich, prompt_toolkit" 2>&1 | Out-Null
        $importExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP
        if ($importExitCode -ne 0) {
            $sibling = "$InstallDir\.venv"
            $hint = if (Test-Path $sibling) {
                "Detected sibling .venv\ at $sibling -- uv synced there instead of venv\. Recover with: cd '$InstallDir'; Remove-Item -Recurse -Force venv; Move-Item .venv venv"
            } else {
                "Recover with: cd '$InstallDir'; `$env:UV_PROJECT_ENVIRONMENT='$InstallDir\venv'; uv sync --extra all --locked"
            }
            throw "Baseline imports failed in $InstallDir\venv (dotenv/openai/rich/prompt_toolkit). The install completed but dependencies are not in the venv. $hint"
        }
        Write-Success "Baseline imports verified in venv"
    }

    if (-not $NoVenv) {
        # uv on Windows can register hermes.exe in dist-info/RECORD but fail to
        # materialise the .exe (file lock during self-update, distlib edge case).
        # Catch it here so a fresh install/update does not finish with a broken
        # `hermes` command while hermes-agent.exe / hermes-acp.exe exist
        $scriptsDir = Join-Path $InstallDir "venv\Scripts"
        $pythonExe = Join-Path $scriptsDir "python.exe"
        if ((Test-Path $scriptsDir) -and (Test-Path $pythonExe)) {
            $scriptNames = & $pythonExe -c @"
import tomllib
with open('pyproject.toml', 'rb') as fh:
    scripts = tomllib.load(fh).get('project', {}).get('scripts', {}) or {}
print(','.join(scripts))
"@ 2>$null
            if ($LASTEXITCODE -eq 0 -and $scriptNames) {
                $expected = @($scriptNames.Trim().Split(',') | Where-Object { $_ })
                $missing = @()
                foreach ($name in $expected) {
                    $exe = Join-Path $scriptsDir "$name.exe"
                    if (-not (Test-Path $exe)) { $missing += "$name.exe" }
                }
                if ($missing.Count -gt 0) {
                    Write-Warn "Console entry point(s) missing: $($missing -join ', ')"
                    Write-Info "Reinstalling entry points..."
                    $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
                    Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install --reinstall -e . }
                    $stillMissing = @()
                    foreach ($name in $expected) {
                        $exe = Join-Path $scriptsDir "$name.exe"
                        if (-not (Test-Path $exe)) { $stillMissing += "$name.exe" }
                    }
                    if ($stillMissing.Count -gt 0) {
                        Write-Warn "Entry points still missing after repair: $($stillMissing -join ', ')"
                        Write-Info "Workaround: `"$pythonExe`" -m hermes_cli.main <command>"
                    } else {
                        Write-Success "Console entry points restored"
                    }
                }
            }
        }
    }

    # Verify the dashboard deps specifically -- they're the most common thing
    # users hit and lazy-import errors from `hermes dashboard` are confusing.
    # If tier 1 failed (the common case), [web] was still picked up by tiers
    # 2-3; only tier 4 leaves you without it.
    $pythonExe = if (-not $NoVenv) { "$InstallDir\venv\Scripts\python.exe" } else { (& $UvCmd python find $PythonVersion) }
    if (Test-Path $pythonExe) {
        $webOk = $false
        # Relax EAP=Stop while running the import probe; see the matching
        # comment on the baseline-imports check above.  Python writes
        # deprecation warnings to stderr and we don't want those wrapped
        # as ErrorRecords that silently force the "not importable" path
        # even when fastapi/uvicorn are actually installed.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $pythonExe -c "import fastapi, uvicorn" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $webOk = $true }
        } catch { }
        $ErrorActionPreference = $prevEAP
        if (-not $webOk) {
            Write-Warn "fastapi/uvicorn not importable -- `hermes dashboard` will not work."
            Write-Info "Attempting targeted install of [web] extra as last resort..."
            & $UvCmd pip install -e ".[web]"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "[web] extra installed; `hermes dashboard` should now work."
            } else {
                Write-Warn "Could not install [web] extra. Run manually: uv pip install --python `"$pythonExe`" `"fastapi>=0.104,<1`" `"uvicorn[standard]>=0.24,<1`""
            }
        }
    }
    
    Pop-Location
    
    Write-Success "All dependencies installed"
}

function Set-PathVariable {
    Write-Info "Setting up hermes command..."
    
    if ($NoVenv) {
        $hermesBin = "$InstallDir"
    } else {
        $hermesBin = "$InstallDir\venv\Scripts"
    }
    
    # Add the venv Scripts dir to user PATH so hermes is globally available
    # On Windows, the hermes.exe in venv\Scripts\ has the venv Python baked in
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($currentPath -notlike "*$hermesBin*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$hermesBin;$currentPath",
            "User"
        )
        Write-Success "Added to user PATH: $hermesBin"
    } else {
        Write-Info "PATH already configured"
    }
    
    # Set HERMES_HOME so the Python code finds config/data in the right place.
    # Only needed on Windows where we install to %LOCALAPPDATA%\hermes instead
    # of the Unix default ~/.hermes
    $currentHermesHome = [Environment]::GetEnvironmentVariable("HERMES_HOME", "User")
    if (-not $currentHermesHome -or $currentHermesHome -ne $HermesHome) {
        [Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
        Write-Success "Set HERMES_HOME=$HermesHome"
    }
    $env:HERMES_HOME = $HermesHome
    
    # Update current session
    $env:Path = "$hermesBin;$env:Path"
    
    Write-Success "hermes command ready"
}

function Write-BootstrapMarker {
    # Writes $InstallDir\.hermes-bootstrap-complete which tells the Hermes
    # desktop app (apps/desktop/electron/main.cjs) "install.ps1 ran
    # successfully — DON'T trigger the legacy first-launch bootstrap
    # runner."
    #
    # Schema mirrors what main.cjs's writeBootstrapMarker() / isBootstrap
    # Complete() expect. Keep this in lockstep when either side changes:
    #   apps/desktop/electron/main.cjs lines 1199-1222
    #   BOOTSTRAP_MARKER_SCHEMA_VERSION = 1 (line 187)
    #
    # Pinned commit/branch come from -Commit + -Branch flags (passed by
    # Hermes-Setup.exe) or fall back to whatever git resolves in the
    # checkout. The desktop validates schemaVersion + pinnedCommit
    # length but doesn't enforce that HEAD matches the pin (users
    # update via `hermes update` which moves HEAD legitimately).
    if (-not (Test-Path $InstallDir)) {
        Write-Warn "Skipping bootstrap marker: $InstallDir doesn't exist"
        return
    }

    # Resolve the pinned commit: explicit -Commit wins, otherwise read
    # the checkout's HEAD via git. If git can't run, leave commit empty
    # and the marker will fail desktop validation (pinnedCommit.length
    # >= 7) — better to be invalid than wrong.
    $pinnedCommit = $Commit
    if (-not $pinnedCommit) {
        # PS 5.1 doesn't support the ?. null-conditional operator, so
        # check Get-Command's result explicitly before reading .Source.
        $gitCmd = Get-Command git -ErrorAction SilentlyContinue
        $gitExe = if ($gitCmd) { $gitCmd.Source } else { $null }
        if ($gitExe) {
            Push-Location $InstallDir
            try {
                $resolved = & $gitExe rev-parse HEAD 2>$null
                if ($LASTEXITCODE -eq 0 -and $resolved) {
                    $pinnedCommit = $resolved.Trim()
                }
            } catch {
                # Ignore — pinnedCommit stays empty, marker stays invalid,
                # desktop falls through to its legacy bootstrap path.
            } finally {
                Pop-Location
            }
        }
    }

    $pinnedBranch = $Branch
    if (-not $pinnedBranch) {
        $pinnedBranch = "main"  # install.ps1's own default for -Branch
    }

    $markerPath = Join-Path $InstallDir ".hermes-bootstrap-complete"
    $marker = [ordered]@{
        schemaVersion = 1
        pinnedCommit  = $pinnedCommit
        pinnedBranch  = $pinnedBranch
        completedAt   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        # desktopVersion field intentionally omitted — only the desktop
        # app knows its own version, and the marker validator doesn't
        # require it. The desktop fills it in if/when it writes its
        # own marker (e.g. after a future in-app upgrade).
    }
    $json = $marker | ConvertTo-Json -Compress:$false

    # Write WITHOUT a UTF-8 BOM. PowerShell 5.1's `Set-Content -Encoding UTF8`
    # always emits a BOM, and Node's plain JSON.parse rejects the BOM as an
    # unexpected character — so a BOM'd marker would silently fail the
    # desktop's readJson(), make isBootstrapComplete() return null, and the
    # desktop would re-run the legacy bootstrap runner anyway. Defeats the
    # whole point. Use the .NET API directly for BOM-less UTF-8.
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($markerPath, $json, $utf8NoBom)

    Write-Success "Bootstrap marker written: $markerPath"
}

function Copy-ConfigTemplates {
    Write-Info "Setting up configuration files..."
    
    # Create the HERMES_HOME directory structure ($HermesHome, default %LOCALAPPDATA%\hermes)
    New-Item -ItemType Directory -Force -Path "$HermesHome\cron" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\sessions" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\pairing" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\hooks" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\image_cache" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\audio_cache" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\memories" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\skills" | Out-Null

    
    # Create .env
    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) {
        $examplePath = "$InstallDir\.env.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $envPath
            Write-Success "Created $envPath from template"
        } else {
            New-Item -ItemType File -Force -Path $envPath | Out-Null
            Write-Success "Created $envPath"
        }
    } else {
        Write-Info "$envPath already exists, keeping it"
    }
    
    # Create config.yaml
    $configPath = "$HermesHome\config.yaml"
    if (-not (Test-Path $configPath)) {
        $examplePath = "$InstallDir\cli-config.yaml.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $configPath
            Write-Success "Created $configPath from template"
        }
    } else {
        Write-Info "$configPath already exists, keeping it"
    }
    
    # Create SOUL.md if it doesn't exist (global persona file).
    # IMPORTANT: write without a BOM.  Windows PowerShell 5.1's
    # ``Set-Content -Encoding UTF8`` writes UTF-8 WITH a byte-order-mark
    # (the default PS5 behaviour), and Hermes's prompt-injection scanner
    # flags the BOM as an invisible unicode character and refuses to
    # load the file.  PS7's ``-Encoding utf8NoBOM`` fixes that but we
    # don't control which PowerShell version the user has.  Go direct
    # to .NET with an explicit UTF8Encoding($false) -- BOM-free on every
    # PowerShell version.
    $soulPath = "$HermesHome\SOUL.md"
    if (-not (Test-Path $soulPath)) {
        # MUST match DEFAULT_SOUL_MD in hermes_cli/default_soul.py. The runtime
        # upgrades the old comment-only scaffold to this text on next run, so
        # drift is self-healing, but keep them in sync to avoid first-run churn.
        $soulContent = @"
You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.
"@
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($soulPath, $soulContent, $utf8NoBom)
        Write-Success "Created $soulPath (edit to customize personality)"
    }
    
    Write-Success "Configuration directory ready: $HermesHome"
    
    # Seed bundled skills into $HermesHome\skills (manifest-based, one-time per skill)
    Write-Info "Syncing bundled skills to $HermesHome\skills ..."
    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (Test-Path $pythonExe) {
        try {
            & $pythonExe "$InstallDir\tools\skills_sync.py" 2>$null
            Write-Success "Skills synced to $HermesHome\skills"
        } catch {
            # Fallback: simple directory copy
            $bundledSkills = "$InstallDir\skills"
            $userSkills = "$HermesHome\skills"
            if ((Test-Path $bundledSkills) -and -not (Get-ChildItem $userSkills -Exclude '.bundled_manifest' -ErrorAction SilentlyContinue)) {
                Copy-Item -Path "$bundledSkills\*" -Destination $userSkills -Recurse -Force -ErrorAction SilentlyContinue
                Write-Success "Skills copied to $HermesHome\skills"
            }
        }
    }
}

function Install-NodeDeps {
    if (-not $HasNode) {
        # Cross-process driver mode (Hermes-Setup.exe runs each -Stage NAME
        # in a fresh powershell.exe) means $script:HasNode set by Stage-Node
        # in the previous process isn't visible here. Re-probe rather than
        # trust the stale global — Stage-Node already ran successfully or
        # the bootstrap would've aborted, so npm is reachable.
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
            Write-Info "Skipping Node.js dependencies (Node not installed)"
            return
        }
    }

    # Resolve npm explicitly to npm.cmd, NOT npm.ps1.  Node.js on Windows
    # ships BOTH npm.cmd (a batch shim) and npm.ps1 (a PowerShell shim).
    # Get-Command's default ordering picks whichever comes first in PATHEXT,
    # and on many systems that's .ps1 -- but .ps1 requires scripts to be
    # enabled in PowerShell's execution policy, which most Windows users
    # don't have (the Restricted / RemoteSigned default blocks unsigned
    # .ps1 files).  .cmd has no such restriction and works on every box.
    #
    # Strategy: look next to the npm shim we found and prefer npm.cmd if
    # it exists in the same directory.  Fall back to whatever Get-Command
    # returned if we can't find a .cmd sibling.
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Warn "npm not found on PATH -- skipping Node.js dependencies."
        Write-Info "Open a new PowerShell window and re-run 'hermes setup tools' later."
        return
    }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $npmCmdSibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $npmCmdSibling) {
            Write-Info "Using npm.cmd (PowerShell execution policy blocks npm.ps1)"
            $npmExe = $npmCmdSibling
        } else {
            Write-Warn "Only npm.ps1 available -- install may fail if script execution is disabled."
            Write-Info "  If it fails, either enable PS script execution or install Node via winget."
        }
    }

    # Helper: run "npm install" in a given directory and surface the real
    # error when it fails.  Returns $true on success.
    #
    # Implementation note: ``Start-Process -FilePath npm.cmd`` fails with
    # ``%1 is not a valid Win32 application`` on some PowerShell versions
    # because Start-Process bypasses cmd.exe / PATHEXT and expects a real
    # PE file.  The invocation-operator ``& $npmExe`` routes through the
    # PowerShell command pipeline which DOES honour .cmd batch shims, so
    # it works uniformly for npm.cmd, npx.cmd, and bare .exe files.
    function _Run-NpmInstall([string]$label, [string]$installDir, [string]$logPath, [string]$npmPath) {
        Push-Location $installDir
        # Capture EAP outside the try block so the catch's restore call always
        # has a meaningful value (see Install-Uv for the full rationale).
        $prevEAP = $ErrorActionPreference
        try {
            # Stream npm's output to BOTH the console and the log file via
            # Tee-Object.  Previously this called ``& npm install --silent
            # *> $logPath`` which redirected every stream to disk and left
            # the user staring at a frozen "Installing..." line for the
            # duration of the install.  On a fresh VM that's 1-3 minutes
            # of total silence, indistinguishable from a hang.
            #
            # Tee writes the live output to stdout AND $logPath; we still
            # capture the exit code afterwards and surface diagnostics
            # on failure.  Note: 2>&1 merges npm's stderr into the success
            # stream first because Tee-Object only sees the success
            # stream of the pipeline.  ForEach-Object { "$_" } coerces
            # each item to a string so PowerShell's NativeCommandError
            # formatter doesn't wrap stderr lines as alarming red blocks
            # (cosmetic polish; the underlying text is unchanged).
            #
            # Relax EAP around the npm invocation: with EAP=Stop (set at
            # the top of this script), PowerShell wraps stderr lines from
            # native commands captured via 2>&1 as ErrorRecord objects and
            # throws on the first one -- even though npm exited 0.  This
            # is the same issue Test-Python and Install-Uv work around
            # for uv's stderr-emitting installer.  Check success via
            # $LASTEXITCODE, which is reliable regardless of stderr noise.
            $ErrorActionPreference = "Continue"
            & $npmPath install --silent 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $logPath
            $code = $LASTEXITCODE
            $ErrorActionPreference = $prevEAP
            if ($code -eq 0) {
                Write-Success "$label dependencies installed"
                Remove-Item -Force $logPath -ErrorAction SilentlyContinue
                return $true
            }
            Write-Warn "$label npm install failed -- exit code $code"
            if (Test-Path $logPath) {
                $errText = (Get-Content $logPath -Raw -ErrorAction SilentlyContinue)
                if ($errText) {
                    $snippet = if ($errText.Length -gt 1200) { $errText.Substring(0, 1200) + "..." } else { $errText }
                    Write-Info "  npm output:"
                    foreach ($line in $snippet -split "`n") {
                        Write-Host "    $line" -ForegroundColor DarkGray
                    }
                    Write-Info "  Full log: $logPath"
                    Show-NpmCertHint $errText | Out-Null
                }
            }
            Write-Info "Run manually later: cd `"$installDir`"; npm install"
            return $false
        } catch {
            if ($prevEAP) { $ErrorActionPreference = $prevEAP }
            Write-Warn "$label npm install could not be launched: $_"
            return $false
        } finally {
            Pop-Location
        }
    }

    # Browser tools
    if (Test-Path "$InstallDir\package.json") {
        Write-Info "Installing Node.js dependencies (browser tools)..."
        $browserLog = "$env:TEMP\hermes-npm-browser-$(Get-Random).log"
        $browserNpmOk = _Run-NpmInstall "Browser tools" $InstallDir $browserLog $npmExe

        # Install Playwright Chromium (mirrors scripts/install.sh behaviour for
        # Linux).  Without this, tools/browser_tool.py::check_browser_requirements
        # returns False (no Chromium under %LOCALAPPDATA%\ms-playwright), and the
        # browser_* tools are silently filtered out of the agent's tool schema.
        # System Chrome at "C:\Program Files\Google\Chrome\..." is NOT used by
        # agent-browser -- it expects a Playwright-managed Chromium.
        if ($browserNpmOk) {
            Write-Info "Installing browser engine (Playwright Chromium)..."
            # npx lives next to npm in the same bin dir.  Prefer .cmd to dodge
            # the same execution-policy gotcha that affects npm.ps1 (see above).
            $npmDir = Split-Path $npmExe -Parent
            $npxExe = $null
            foreach ($cand in @("npx.cmd", "npx.exe", "npx")) {
                $try = Join-Path $npmDir $cand
                if (Test-Path $try) { $npxExe = $try; break }
            }
            if (-not $npxExe) {
                $npxCmd = Get-Command npx -ErrorAction SilentlyContinue
                if ($npxCmd) { $npxExe = $npxCmd.Source }
            }
            if (-not $npxExe) {
                Write-Warn "npx not found -- cannot install Playwright Chromium."
                Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
            } else {
                $pwLog = "$env:TEMP\hermes-playwright-install-$(Get-Random).log"
                Push-Location $InstallDir
                # Capture EAP outside the try block so the catch's restore call
                # always has a meaningful value (see Install-Uv for the full
                # rationale).
                $prevEAP = $ErrorActionPreference
                try {
                    # Playwright Chromium is ~170MB compressed and the
                    # download regularly takes 3-10 minutes on a fresh
                    # VM.  Tee the output to console + log so the user
                    # sees download progress in real time instead of
                    # staring at a silent prompt that looks hung.  See
                    # _Run-NpmInstall above for the same pattern and
                    # the rationale behind 2>&1 before the pipe.
                    Write-Info "(this can take several minutes -- streaming progress below)"
                    # --yes auto-accepts npx's "Need to install playwright@X.Y.Z"
                    # confirmation prompt.  Without it, npx 7+ blocks on stdin
                    # waiting for a y/N answer that never comes when this is
                    # invoked through a pipeline (Tee-Object disconnects stdin
                    # from the user's TTY), and the install hangs indefinitely
                    # after printing "Need to install the following packages:
                    # playwright@X.Y.Z".
                    #
                    # Relax EAP around the playwright invocation: playwright
                    # emits a "Chromium downloaded to ..." success banner to
                    # stderr after a successful install.  Under EAP=Stop, the
                    # 2>&1 merge wraps those stderr lines as ErrorRecord
                    # objects and throws -- causing this catch block to fire
                    # with a mangled banner as the error message even though
                    # the install actually succeeded.  Check $LASTEXITCODE
                    # instead, which is the reliable signal.
                    #
                    # The ForEach-Object { "$_" } coercion BEFORE Tee-Object
                    # is a cosmetic polish: with bare 2>&1, PowerShell still
                    # renders stderr lines through its NativeCommandError
                    # formatter (the red "npx.cmd : ..." block).  Coercing
                    # each pipeline item to a string strips that wrapper so
                    # the user sees clean playwright output instead of the
                    # alarming-looking error formatting.
                    $ErrorActionPreference = "Continue"
                    & $npxExe --yes playwright install chromium 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $pwLog
                    $pwCode = $LASTEXITCODE
                    $ErrorActionPreference = $prevEAP
                    if ($pwCode -eq 0) {
                        Write-Success "Playwright Chromium installed (browser tools ready)"
                        Remove-Item -Force $pwLog -ErrorAction SilentlyContinue
                    } else {
                        Write-Warn "Playwright Chromium install failed -- exit code $pwCode"
                        Write-Warn "Browser tools will not work until Chromium is installed."
                        if (Test-Path $pwLog) {
                            $pwErr = Get-Content $pwLog -Raw -ErrorAction SilentlyContinue
                            if ($pwErr) {
                                $snippet = if ($pwErr.Length -gt 1200) { $pwErr.Substring(0, 1200) + "..." } else { $pwErr }
                                Write-Info "  playwright output:"
                                foreach ($line in $snippet -split "`n") {
                                    Write-Host "    $line" -ForegroundColor DarkGray
                                }
                                Write-Info "  Full log: $pwLog"
                            }
                        }
                        Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
                    }
                } catch {
                    if ($prevEAP) { $ErrorActionPreference = $prevEAP }
                    Write-Warn "Playwright Chromium install could not be launched: $_"
                    Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
                } finally {
                    Pop-Location
                }
            }
        }
    }

    # TUI
    $tuiDir = "$InstallDir\ui-tui"
    if (Test-Path "$tuiDir\package.json") {
        Write-Info "Installing TUI dependencies..."
        $tuiLog = "$env:TEMP\hermes-npm-tui-$(Get-Random).log"
        [void](_Run-NpmInstall "TUI" $tuiDir $tuiLog $npmExe)
    }
}

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

# Electron package dir — workspace-local nest first, then root hoist.
function Get-ElectronDir {
    param([string]$InstallDir)
    $desktopLocal = Join-Path $InstallDir 'apps\desktop\node_modules\electron'
    if (Test-Path -LiteralPath $desktopLocal) { return $desktopLocal }
    return (Join-Path $InstallDir 'node_modules\electron')
}

# True when dist/ holds a usable Electron binary (#38673 / run-electron-builder.cjs).
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
    # so an "unpacked" build (electron-builder --dir) is enough — we
    # don't need to produce an NSIS/MSI artifact here.

    # Always re-resolve Node here. Stages run in separate PowerShell processes,
    # so $script:HasNode from Stage-Node isn't visible; more importantly Test-Node
    # enforces the build floor (^20.19 || >=22.12) and prepends the Hermes-managed
    # Node to PATH, so the build never runs on a too-old system Node -- the cause
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
        # need a side TEMP log file — the installer's bootstrap log
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
    # (whose macOS symlinks crash 7-Zip on non-admin Windows — a dead end we
    # are NOT trying to work around). The Hermes icon + product name are
    # stamped onto Hermes.exe by our own rcedit step (Set-DesktopExeIdentity)
    # AFTER this build, completely decoupled from electron-builder signing.
    #
    # WIN_CSC_LINK and WIN_CSC_KEY_PASSWORD explicitly cleared as
    # belt-and-suspenders: if the user's environment has them set
    # for some other tool, electron-builder would still try to sign.
    Write-Info "Building desktop app (this takes 1-3 minutes)..."
    $buildLog = "$env:TEMP\hermes-desktop-build-$(Get-Random).log"
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
            throw "apps/desktop build failed (exit $code)"
        }
        Write-Success "Desktop app built"
        Remove-Item -Force $buildLog -ErrorAction SilentlyContinue
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Pop-Location
        throw
    } finally {
        # Restore env to whatever the caller had — don't leak our
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
    #     electron-builder `afterPack` hook (apps/desktop/scripts/after-pack.cjs)
    #     during `npm run pack` above — for every build, so the installer's
    #     --update rebuild stays branded too. No separate stamp step needed here.
    #     electron-builder's own rcedit step stays disabled (signAndEditExecutable
    #     =false) because enabling it drags in signtool -> winCodeSign -> the
    #     unfixable symlink crash; the afterPack hook runs rcedit directly.

    # 4. Create Start Menu + Desktop shortcuts pointing DIRECTLY at the packed
    #    Hermes.exe. We deliberately do NOT point them at `hermes desktop`: that
    #    command rebuilds (npm install + electron-builder) on every launch,
    #    which would cost minutes each time. The packed exe is the consumer —
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
        # Best-effort and silent — never fail the install over a cosmetic cache.
        try {
            & ie4uinit.exe -show 2>$null
        } catch {
            # ie4uinit may be absent/renamed on some SKUs — ignore.
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
        @{ Var = "DISCORD_BOT_TOKEN";  Import = "discord";   Spec = "discord.py==2.7.1" },
        @{ Var = "DISCORD_BOT_TOKEN";  Import = "nacl";      Spec = "PyNaCl==1.6.2" },
        @{ Var = "DISCORD_BOT_TOKEN";  Import = "davey";     Spec = "davey==0.1.4" },
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
        & ".\venv\Scripts\python.exe" -m hermes_cli setup
    } else {
        python -m hermes_cli setup
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

# ============================================================================
# Stage protocol
# ============================================================================
#
# install.ps1 supports a small, stable "stage protocol" that lets programmatic
# callers (the desktop GUI's onboarding wizard, CI, future install.sh, etc.)
# drive the install one step at a time and surface progress/errors with their
# own UI.  CLI users running the canonical `irm | iex` one-liner never
# encounter this -- default invocation behaves exactly as before.
#
# Entry points:
#
#   install.ps1                       Interactive install (today's behavior).
#   install.ps1 -ProtocolVersion      Emit the protocol version integer.
#   install.ps1 -Manifest             Emit the stage manifest as JSON.
#   install.ps1 -Stage <name>         Run one stage and emit its result.
#   install.ps1 -NonInteractive       Disable all Read-Host prompts (also
#                                     skips the setup wizard and the gateway
#                                     autostart prompt).  Can be combined
#                                     with default invocation to do a full
#                                     non-interactive install.
#   install.ps1 -Json                 Emit machine-readable JSON instead of
#                                     the human-readable success banner at
#                                     the end of a full install.
#
# Manifest schema (the JSON returned by -Manifest):
#
#   {
#     "protocol_version": 1,
#     "stages": [
#       {
#         "name": "uv",
#         "title": "Installing uv package manager",
#         "category": "prereqs",
#         "needs_user_input": false
#       },
#       ...
#     ]
#   }
#
# Stage result (the JSON written by -Stage <name>):
#
#   {
#     "stage": "uv",
#     "ok": true,
#     "skipped": false,
#     "reason": null,
#     "duration_ms": 1234
#   }
#
# Exit codes:
#
#   0 -- success (stage ran, or stage was deliberately skipped).
#   1 -- generic failure; the stage threw.
#   2 -- unknown stage name passed to -Stage.
#
# Adding a stage:
#
#   1. Append an entry to $InstallStages below.
#   2. Make sure the worker function it points at is idempotent and respects
#      $NonInteractive when it has prompts.  Add it before "configure"
#      (the wizard) or "gateway" (autostart) if it should run unconditionally;
#      after those if it's optional post-install glue.
#   3. Do NOT bump $InstallStageProtocolVersion -- adding stages is additive.
#      Drivers iterate the manifest dynamically.
#
# ============================================================================

# Stage definitions -- the single source of truth.  Each entry maps a stable
# stage name (the API contract drivers depend on) to the worker function that
# implements it.  ``Title`` is what UIs show; ``Category`` lets UIs group
# stages; ``NeedsUserInput`` tells UIs "this stage prompts -- either skip it
# or arrange to provide answers another way."
$InstallStages = @(
    @{ Name = "uv";               Title = "Installing uv package manager";        Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Uv" }
    @{ Name = "python";           Title = "Verifying Python $PythonVersion";      Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Python" }
    @{ Name = "git";              Title = "Installing Git";                       Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Git" }
    @{ Name = "node";             Title = "Detecting Node.js";                    Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-Node" }
    @{ Name = "system-packages";  Title = "Installing ripgrep and ffmpeg";        Category = "prereqs";      NeedsUserInput = $false; Worker = "Stage-SystemPackages" }
    @{ Name = "repository";       Title = "Cloning Hermes repository";            Category = "install";      NeedsUserInput = $false; Worker = "Stage-Repository" }
    @{ Name = "venv";             Title = "Creating Python virtual environment";  Category = "install";      NeedsUserInput = $false; Worker = "Stage-Venv" }
    @{ Name = "dependencies";     Title = "Installing Python dependencies";       Category = "install";      NeedsUserInput = $false; Worker = "Stage-Dependencies" }
    @{ Name = "node-deps";        Title = "Installing Node.js dependencies";      Category = "install";      NeedsUserInput = $false; Worker = "Stage-NodeDeps" }
)
if ($IncludeDesktop) {
    # Insert AFTER node-deps so workspace npm is already installed when
    # the desktop build runs. Inserted only when explicitly requested
    # (Hermes-Setup.exe), never via the irm|iex CLI one-liner.
    $InstallStages += @{ Name = "desktop"; Title = "Building desktop app"; Category = "install"; NeedsUserInput = $false; Worker = "Stage-Desktop" }
}
$InstallStages += @(
    @{ Name = "path";             Title = "Adding Hermes to PATH";                Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-Path" }
    @{ Name = "config-templates"; Title = "Writing configuration templates";      Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-ConfigTemplates" }
    @{ Name = "platform-sdks";    Title = "Installing messaging platform SDKs";   Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-PlatformSdks" }
    @{ Name = "bootstrap-marker"; Title = "Marking install complete";              Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-BootstrapMarker" }
    # Interactive stages.  In non-interactive mode these become no-ops; the
    # caller (GUI / CI) handles the equivalent UX themselves.
    @{ Name = "configure";        Title = "Configuring API keys and models";      Category = "post-install"; NeedsUserInput = $true;  Worker = "Stage-Configure" }
    @{ Name = "gateway";          Title = "Starting messaging gateway";           Category = "post-install"; NeedsUserInput = $true;  Worker = "Stage-Gateway" }
)

# Stage workers -- thin wrappers that delegate to the existing Install-* /
# Test-* / Invoke-* functions while preserving their error semantics.  Kept
# as a separate layer so the existing functions remain callable directly
# (helpful for one-off recovery: ``. install.ps1; Install-Venv``).
#
# Stages that depend on uv (anything after Stage-Uv) call Resolve-UvCmd
# first so they work in cross-process driver mode where $script:UvCmd
# set by Stage-Uv in a sibling powershell process is not visible here.
# Resolve-UvCmd is a fast no-op when $script:UvCmd is already populated
# (the default-invocation case where Main runs everything in one
# process), and throws cleanly if uv truly isn't installed yet.
function Stage-Uv               { if (-not (Install-Uv))     { throw "uv installation failed" } }
function Stage-Python           { Resolve-UvCmd; if (-not (Test-Python))    { throw "Python $PythonVersion not available" } }
function Stage-Git              { if (-not (Install-Git))    { throw "Git not available and auto-install failed -- install from https://git-scm.com/download/win then re-run" } }
# Node is optional (browser tools degrade gracefully without it).  Surface
# failure to the JSON contract as skipped=true / reason rather than ok=true,
# so a GUI driver consuming the manifest can distinguish "node ready" from
# "node missing".  Install flow continues either way -- matches the
# existing Write-Completion behavior that prints a "Note: Node.js could
# not be installed" hint instead of aborting.
function Stage-Node             {
    if (-not (Test-Node)) {
        $script:_StageSkippedReason = "Node.js not available; browser tools will be unavailable until node is installed manually from https://nodejs.org/en/download/"
    }
}
function Stage-SystemPackages   { Install-SystemPackages }
function Stage-Repository       { Install-Repository }
function Stage-Venv             { Resolve-UvCmd; Install-Venv }
function Stage-Dependencies     { Resolve-UvCmd; Install-Dependencies }
function Stage-NodeDeps         { Install-NodeDeps }
function Stage-Desktop          { Install-Desktop }
function Stage-Path             { Set-PathVariable }
function Stage-ConfigTemplates  { Copy-ConfigTemplates }
function Stage-PlatformSdks     { Resolve-UvCmd; Install-PlatformSdks }
function Stage-BootstrapMarker  { Write-BootstrapMarker }
function Stage-Configure        { Invoke-SetupWizard }
function Stage-Gateway          { Start-GatewayIfConfigured }

function Get-InstallStage {
    param([string]$Name)
    foreach ($s in $InstallStages) {
        if ($s.Name -eq $Name) { return $s }
    }
    return $null
}

function Step-OutOfInstallDir {
    # Windows refuses to delete a directory any shell is currently cd'd
    # inside -- and silently leaves orphan files behind, which then wedge
    # "is this a valid git repo" probes on re-install.  Harmless when the
    # caller ran the installer from somewhere else.
    try {
        $currentResolved = (Get-Location).ProviderPath
        $installResolved = $null
        if (Test-Path $InstallDir) {
            $installResolved = (Resolve-Path $InstallDir -ErrorAction SilentlyContinue).ProviderPath
        }
        if ($installResolved -and $currentResolved.ToLower().StartsWith($installResolved.ToLower())) {
            Write-Info "Stepping out of $InstallDir so Windows can replace files there if needed..."
            Set-Location $env:USERPROFILE
        }
    } catch {}
}

function Invoke-Stage {
    param(
        [Parameter(Mandatory=$true)] [hashtable]$StageDef
    )

    # Refresh PATH from registry so this stage sees binaries installed by
    # prior stages, even when each stage runs in its own powershell process.
    # No-op in cost-relevant cases (default invocation path syncs once per
    # foreach pass; cross-process drivers get the necessary freshening).
    Sync-EnvPath

    # Per-stage soft-skip channel.  A worker can populate
    # $script:_StageSkippedReason to surface "ran, but the thing it was
    # supposed to set up is not available" as skipped=true in the JSON
    # frame, without throwing.  Used by Stage-Node so the install flow
    # doesn't abort when an optional capability is missing while still
    # being honest in the protocol contract.  Reset before each stage so
    # a prior stage's reason can never leak into a later stage's frame.
    $script:_StageSkippedReason = $null

    $start = [DateTime]::UtcNow
    $result = @{
        stage        = $StageDef.Name
        ok           = $false
        skipped      = $false
        reason       = $null
        duration_ms  = 0
    }

    try {
        & $StageDef.Worker
        $result.ok = $true
        if ($script:_StageSkippedReason) {
            $result.skipped = $true
            $result.reason  = $script:_StageSkippedReason
        }
    } catch {
        $result.ok = $false
        $result.reason = "$_"
        throw
    } finally {
        $result.duration_ms = [int]([DateTime]::UtcNow - $start).TotalMilliseconds
        if ($Json -or $Stage) {
            # In stage-driver mode every stage emits a JSON line so the
            # caller can stream progress.  In default interactive mode we
            # stay silent here (the worker already wrote human output).
            $result | ConvertTo-Json -Compress | Write-Output
            # Tell the entry-point catch that we've already emitted a
            # frame for this failure (when $result.ok = $false), so it
            # doesn't double-emit a second JSON object and break the
            # one-line-per-stage contract the driver protocol promises.
            if (-not $result.ok) {
                $script:_StageEmittedErrorFrame = $true
            }
        }
    }
}

# ============================================================================
# Main
# ============================================================================

function Invoke-AllStages {
    Step-OutOfInstallDir
    foreach ($s in $InstallStages) {
        Invoke-Stage -StageDef $s
    }
}

function Invoke-EnsureMode {
    param([string]$Deps)
    $depList = $Deps -split ","
    foreach ($dep in $depList) {
        $dep = $dep.Trim()
        switch ($dep) {
            "node" {
                [void](Test-Node)
                if (-not $script:HasNode) {
                    Write-Err "Node.js could not be installed"
                    exit 1
                }
            }
            "browser" {
                [void](Test-Node)
                if ($script:HasNode) {
                    Install-AgentBrowser
                } else {
                    Write-Err "Node.js is required for browser tools but could not be installed"
                    exit 1
                }
            }
            "ripgrep" {
                Write-Info "ripgrep: install manually on Windows (scoop install ripgrep)"
            }
            "ffmpeg" {
                Write-Info "ffmpeg: install manually on Windows (scoop install ffmpeg)"
            }
            default {
                Write-Err "Unknown dependency: $dep"
                exit 1
            }
        }
    }
}

function Invoke-PostInstallMode {
    Write-Info "Running post-install setup..."
    Invoke-EnsureMode -Deps "node,browser"
    Write-Info "Post-install complete"
}

function Main {
    Write-Banner
    Invoke-AllStages
    if (-not $Json) {
        Write-Completion
    } else {
        @{ ok = $true; protocol_version = $InstallStageProtocolVersion } | ConvertTo-Json -Compress | Write-Output
    }
}

# ----------------------------------------------------------------------------
# Entry-point dispatch
# ----------------------------------------------------------------------------
#
# All branches funnel through one try/catch so errors don't kill an `irm |
# iex` PowerShell session, and so failures in stage-driver mode produce a
# structured JSON error frame instead of a bare exception.

try {
    if ($Ensure -ne "") {
        if ($PSBoundParameters.ContainsKey("Stage")) {
            Write-Err "Cannot use -Ensure and -Stage simultaneously"
            exit 1
        }
        Invoke-EnsureMode -Deps $Ensure
        exit 0
    }
    if ($PostInstall) {
        Invoke-PostInstallMode
        exit 0
    }

    if ($ProtocolVersion) {
        Write-Output $InstallStageProtocolVersion
        exit 0
    }

    if ($Manifest) {
        $payload = @{
            protocol_version = $InstallStageProtocolVersion
            stages = @($InstallStages | ForEach-Object {
                @{
                    name             = $_.Name
                    title            = $_.Title
                    category         = $_.Category
                    needs_user_input = $_.NeedsUserInput
                }
            })
        }
        $payload | ConvertTo-Json -Depth 5 -Compress | Write-Output
        exit 0
    }

    # Use PSBoundParameters rather than $Stage truthiness so that an
    # explicit `-Stage ""` from a misbehaving driver doesn't fall through
    # to the full-install Main path and silently kick off a destructive
    # operation.  Empty string is a contract violation; surface it as
    # unknown-stage exit 2 with a structured JSON frame.
    if ($PSBoundParameters.ContainsKey("Stage")) {
        $def = Get-InstallStage -Name $Stage
        if (-not $def) {
            $err = @{
                ok     = $false
                stage  = $Stage
                reason = "unknown stage: $Stage. Run install.ps1 -Manifest to list valid stages."
            }
            $err | ConvertTo-Json -Compress | Write-Output
            exit 2
        }
        Step-OutOfInstallDir
        Invoke-Stage -StageDef $def
        exit 0
    }

    # Default: full install (today's behavior, plus optional -NonInteractive
    # and -Json layered on by the params above).
    Main
} catch {
    if ($Json -or $Stage) {
        # Stage-driver mode: caller wants JSON they can parse.  Emit a
        # structured error frame and exit non-zero -- BUT only if
        # Invoke-Stage didn't already emit one for this same failure.
        # The inner finally emits the authoritative per-stage frame
        # (with duration_ms + skipped fields); a second emit here
        # would produce two concatenated JSON objects on stdout and
        # break drivers that parse one-line-per-invocation.
        if (-not $script:_StageEmittedErrorFrame) {
            $err = @{
                ok     = $false
                stage  = if ($Stage) { $Stage } else { $null }
                reason = "$_"
            }
            $err | ConvertTo-Json -Compress | Write-Output
        }
        exit 1
    }

    # Interactive mode: keep today's friendly recovery hint.
    Write-Host ""
    Write-Err "Installation failed: $_"
    Write-Host ""
    Write-Info "If the error is unclear, try downloading and running the script directly:"
    Write-Host "  Invoke-WebRequest -Uri 'https://hermes-agent.nousresearch.com/install.ps1' -OutFile install.ps1" -ForegroundColor Yellow
    Write-Host "  .\install.ps1" -ForegroundColor Yellow
    Write-Host ""
}
