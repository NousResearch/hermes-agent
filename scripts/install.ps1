# ============================================================================
# Hermes Agent Installer for Windows
# ============================================================================
# Installation script for Windows (PowerShell).
# Uses uv for fast Python provisioning and package management.
#
# Usage:
#   irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
#
# Or download and run with options:
#   .\install.ps1 -NoVenv -SkipSetup
#
# ============================================================================

param(
    [switch]$NoVenv,
    [switch]$SkipSetup,
    [string]$Branch = "main",
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [string]$InstallDir = "$env:LOCALAPPDATA\hermes\hermes-agent",

    # --- Stage protocol (additive; default invocation behaves as before) ----
    # See the "Stage protocol" section near the bottom of the file for the
    # full contract.  Intended for programmatic drivers (the desktop GUI's
    # onboarding wizard, CI, future install.sh parity, etc.).  CLI users
    # running the canonical `irm | iex` one-liner never touch these flags.
    [switch]$Manifest,
    [string]$Stage,
    [switch]$ProtocolVersion,
    [switch]$NonInteractive,
    [switch]$Json
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
# Configuration
# ============================================================================

$RepoUrlSsh = "git@github.com:NousResearch/hermes-agent.git"
$RepoUrlHttps = "https://github.com/NousResearch/hermes-agent.git"
$PythonVersion = "3.11"
$NodeVersion = "22"

# Stage-protocol version.  Bumped only for genuinely breaking changes to the
# manifest schema, stage-name set semantics, or stdout JSON shape.  Adding a
# new stage does NOT bump this -- drivers iterate the manifest dynamically.
$InstallStageProtocolVersion = 1

# ============================================================================
# Helper functions
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

# ============================================================================
# Dependency checks
# ============================================================================

function Install-Uv {
    Write-Info "Checking for uv package manager..."
    
    # Check if uv is already available
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $version = uv --version
        $script:UvCmd = "uv"
        Write-Success "uv found ($version)"
        return $true
    }
    
    # Check common install locations
    $uvPaths = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe"
    )
    foreach ($uvPath in $uvPaths) {
        if (Test-Path $uvPath) {
            $script:UvCmd = $uvPath
            $version = & $uvPath --version
            Write-Success "uv found at $uvPath ($version)"
            return $true
        }
    }
    
    # Install uv
    Write-Info "Installing uv (fast Python package manager)..."
    try {
        # Relax ErrorActionPreference around the nested astral installer.
        # The astral installer (a separate `powershell -c "irm ... | iex"`)
        # writes download progress to stderr.  With $ErrorActionPreference
        # = "Stop" set at the top of this script, PowerShell wraps stderr
        # lines from native commands (which `powershell -c` is, from our
        # perspective) as ErrorRecord objects when captured via 2>&1, then
        # throws a terminating exception on the first one -- even though
        # uv installs successfully and the child exits 0.  Same fix
        # pattern Test-Python uses for `uv python install`; verify success
        # via Test-Path on the expected binary afterwards, which is more
        # reliable than exit-code/stderr signal anyway.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null
        $ErrorActionPreference = $prevEAP

        # Find the installed binary
        $uvExe = "$env:USERPROFILE\.local\bin\uv.exe"
        if (-not (Test-Path $uvExe)) {
            $uvExe = "$env:USERPROFILE\.cargo\bin\uv.exe"
        }
        if (-not (Test-Path $uvExe)) {
            # Refresh PATH and try again
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                $uvExe = (Get-Command uv).Source
            }
        }
        
        if (Test-Path $uvExe) {
            $script:UvCmd = $uvExe
            $version = & $uvExe --version
            Write-Success "uv installed ($version)"
            return $true
        }
        
        Write-Err "uv installed but not found on PATH"
        Write-Info "Try restarting your terminal and re-running"
        return $false
    } catch {
        # Restore EAP in case the try block threw before the assignment
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
# at the top to populate $script:UvCmd from PATH or known install paths.
# Throws if uv is not findable -- the caller's stage then surfaces a
# clean error via the stage-driver's try/catch.  Fast path is a single
# Get-Command call when uv is on PATH (the common case after Stage-Uv
# ran path-modifying installs in a sibling process).
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

    # Try PATH first (covers `winget install astral.uv`, manual installs,
    # and the post-Install-Uv state where uv.exe lives in
    # %USERPROFILE%\.local\bin which the installer added to PATH).
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

    # Check the well-known install locations the astral.sh installer drops
    # uv into.  Mirrors the probe order Install-Uv uses.
    foreach ($uvPath in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $uvPath) {
            $script:UvCmd = $uvPath
            return
        }
    }

    throw "uv is not installed or not on PATH. Run install.ps1 -Stage uv first."
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
        $prevEAP = $ErrorActionPreference
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
    foreach ($fallbackVer in @("3.12", "3.13", "3.10")) {
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
        $arch = if ([Environment]::Is64BitOperatingSystem) {
            # Detect ARM64 vs x64 explicitly; PortableGit ships separate assets.
            if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64" -or $env:PROCESSOR_ARCHITEW6432 -eq "ARM64") {
                "arm64"
            } else {
                "64-bit"
            }
        } else {
            # PortableGit does not ship a 32-bit build -- fall back to MinGit 32-bit
            # with a warning that bash-based features will be unavailable.
            "32-bit-mingit"
        }

        $releaseApi = "https://api.github.com/repos/git-for-windows/git/releases/latest"
        $release = Invoke-RestMethod -Uri $releaseApi -UseBasicParsing -Headers @{ "User-Agent" = "hermes-installer" }

        if ($arch -eq "32-bit-mingit") {
            Write-Warn "32-bit Windows detected -- PortableGit is 64-bit only.  Installing MinGit 32-bit as a last resort; bash-dependent Hermes features (terminal tool, agent-browser) will not work on this machine."
            $assetPattern = "MinGit-*-32-bit.zip"
            $downloadIsZip = $true
        } elseif ($arch -eq "arm64") {
            $assetPattern = "PortableGit-*-arm64.7z.exe"
            $downloadIsZip = $false
        } else {
            $assetPattern = "PortableGit-*-64-bit.7z.exe"
            $downloadIsZip = $false
        }

        $asset = $release.assets | Where-Object { $_.name -like $assetPattern } | Select-Object -First 1

        if (-not $asset) {
            throw "Could not find $assetPattern in latest git-for-windows release"
        }

        $downloadUrl = $asset.browser_download_url
        $downloadExt = if ($downloadIsZip) { "zip" } else { "7z.exe" }
        $tmpFile = "$env:TEMP\$($asset.name)"
        $gitDir = "$HermesHome\git"

        Write-Info "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)..."
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

function Test-Node {
    Write-Info "Checking Node.js (for browser tools)..."

    if (Get-Command node -ErrorAction SilentlyContinue) {
        $version = node --version
        Write-Success "Node.js $version found"
        $script:HasNode = $true
        return $true
    }

    # Check our own managed install from a previous run
    $managedNode = "$HermesHome\node\node.exe"
    if (Test-Path $managedNode) {
        $version = & $managedNode --version
        $env:Path = "$HermesHome\node;$env:Path"
        Write-Success "Node.js $version found (Hermes-managed)"
        $script:HasNode = $true
        return $true
    }

    Write-Info "Node.js not found -- installing Node.js $NodeVersion LTS..."

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
        $arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
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
        try {
            # Relax EAP=Stop so stderr lines from winget don't get wrapped
            # as ErrorRecords and short-circuit the 2>&1 pipe before we can
            # check the post-condition.  See the long comment in Install-Uv
            # for the same pattern.
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            winget install OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
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
        foreach ($pkg in $wingetPkgs) {
            try {
                winget install $pkg --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Null
            } catch { }
        }
        # Refresh PATH and recheck
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed"
            $script:HasRipgrep = $true
            $needRipgrep = $false
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
        }
        if (-not $needRipgrep -and -not $needFfmpeg) { return }
    }

    # Fallback: choco
    if ($hasChoco -and ($needRipgrep -or $needFfmpeg)) {
        Write-Info "Trying Chocolatey..."
        foreach ($pkg in $chocoPkgs) {
            try { choco install $pkg -y 2>&1 | Out-Null } catch { }
        }
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
        # Validate the repo properly by asking git itself.  Two checks
        # belt-and-braces: rev-parse AND git status.  If either fails the
        # repo is broken and we fall through to a fresh clone.
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

                if ($revParseOk -and $statusOk) {
                    $repoValid = $true
                }
            } catch {}
            Pop-Location
        }

        if ($repoValid) {
            Write-Info "Existing installation found, updating..."
            Push-Location $InstallDir
            try {
                git -c windows.appendAtomically=false fetch origin
                if ($LASTEXITCODE -ne 0) { throw "git fetch failed (exit $LASTEXITCODE)" }
                git -c windows.appendAtomically=false checkout $Branch
                if ($LASTEXITCODE -ne 0) { throw "git checkout $Branch failed (exit $LASTEXITCODE)" }
                git -c windows.appendAtomically=false pull origin $Branch
                if ($LASTEXITCODE -ne 0) { throw "git pull failed (exit $LASTEXITCODE)" }
            } finally {
                Pop-Location
            }
            $didUpdate = $true
        } else {
            # Directory exists but isn't a usable git repo.  Wipe it and
            # fall through to a fresh clone.  A leftover ``.git`` stub from
            # a partial uninstall used to lock the installer into the
            # "update" branch forever, emitting three ``fatal: not a git
            # repository`` errors and failing with "not in a git directory".
            Write-Warn "Existing directory at $InstallDir is not a valid git repo -- replacing it."
            try {
                Remove-Item -Recurse -Force $InstallDir -ErrorAction Stop
            } catch {
                Write-Err "Could not remove $InstallDir : $_"
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
            git -c windows.appendAtomically=false clone --branch $Branch --recurse-submodules $RepoUrlSsh $InstallDir
            if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
        } catch { }
        $env:GIT_SSH_COMMAND = $null

        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Info "SSH failed, trying HTTPS..."
            try {
                git -c windows.appendAtomically=false clone --branch $Branch --recurse-submodules $RepoUrlHttps $InstallDir
                if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
            } catch { }
        }

        # Fallback: download ZIP archive (bypasses git file I/O issues entirely)
        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Warn "Git clone failed -- downloading ZIP archive instead..."
            try {
                $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/refs/heads/$Branch.zip"
                $zipPath = "$env:TEMP\hermes-agent-$Branch.zip"
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

    # Ensure submodules are initialized and updated
    Write-Info "Initializing submodules..."
    git -c windows.appendAtomically=false submodule update --init --recursive 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Submodule init failed (terminal/RL tools may need manual setup)"
    } else {
        Write-Success "Submodules ready"
    }
    Pop-Location

    Write-Success "Repository ready"
}

function Install-Venv {
    if ($NoVenv) {
        Write-Info "Skipping virtual environment (-NoVenv)"
        return
    }
    
    Write-Info "Creating virtual environment with Python $PythonVersion..."
    
    Push-Location $InstallDir
    
    if (Test-Path "venv") {
        Write-Info "Virtual environment already exists, recreating..."
        Remove-Item -Recurse -Force "venv"
    }
    
    # uv creates the venv and pins the Python version in one step
    & $UvCmd venv venv --python $PythonVersion
    
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
        & $UvCmd sync --extra all --locked
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
        & $UvCmd pip install -e $tier.Spec
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

function Copy-ConfigTemplates {
    Write-Info "Setting up configuration files..."
    
    # Create ~/.hermes directory structure
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
            Write-Success "Created ~/.hermes/.env from template"
        } else {
            New-Item -ItemType File -Force -Path $envPath | Out-Null
            Write-Success "Created ~/.hermes/.env"
        }
    } else {
        Write-Info "~/.hermes/.env already exists, keeping it"
    }
    
    # Create config.yaml
    $configPath = "$HermesHome\config.yaml"
    if (-not (Test-Path $configPath)) {
        $examplePath = "$InstallDir\cli-config.yaml.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $configPath
            Write-Success "Created ~/.hermes/config.yaml from template"
        }
    } else {
        Write-Info "~/.hermes/config.yaml already exists, keeping it"
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
        $soulContent = @"
# Hermes Agent Persona

<!--
This file defines the agent's personality and tone.
The agent will embody whatever you write here.
Edit this to customize how Hermes communicates with you.

Examples:
  - "You are a warm, playful assistant who uses kaomoji occasionally."
  - "You are a concise technical expert. No fluff, just facts."
  - "You speak like a friendly coworker who happens to know everything."

This file is loaded fresh each message -- no restart needed.
Delete the contents (or this file) to use the default personality.
-->
"@
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($soulPath, $soulContent, $utf8NoBom)
        Write-Success "Created ~/.hermes/SOUL.md (edit to customize personality)"
    }
    
    Write-Success "Configuration directory ready: ~/.hermes/"
    
    # Seed bundled skills into ~/.hermes/skills/ (manifest-based, one-time per skill)
    Write-Info "Syncing bundled skills to ~/.hermes/skills/ ..."
    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (Test-Path $pythonExe) {
        try {
            & $pythonExe "$InstallDir\tools\skills_sync.py" 2>$null
            Write-Success "Skills synced to ~/.hermes/skills/"
        } catch {
            # Fallback: simple directory copy
            $bundledSkills = "$InstallDir\skills"
            $userSkills = "$HermesHome\skills"
            if ((Test-Path $bundledSkills) -and -not (Get-ChildItem $userSkills -Exclude '.bundled_manifest' -ErrorAction SilentlyContinue)) {
                Copy-Item -Path "$bundledSkills\*" -Destination $userSkills -Recurse -Force -ErrorAction SilentlyContinue
                Write-Success "Skills copied to ~/.hermes/skills/"
            }
        }
    }
}

function Install-NodeDeps {
    if (-not $HasNode) {
        Write-Info "Skipping Node.js dependencies (Node not installed)"
        return
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
            $prevEAP = $ErrorActionPreference
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
                    $prevEAP = $ErrorActionPreference
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
    Write-Host "|              [OK] Installation Complete!                   |" -ForegroundColor Green
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
    @{ Name = "path";             Title = "Adding Hermes to PATH";                Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-Path" }
    @{ Name = "config-templates"; Title = "Writing configuration templates";      Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-ConfigTemplates" }
    @{ Name = "platform-sdks";    Title = "Installing messaging platform SDKs";   Category = "finalize";     NeedsUserInput = $false; Worker = "Stage-PlatformSdks" }
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
function Stage-Node             { [void](Test-Node) }
function Stage-SystemPackages   { Install-SystemPackages }
function Stage-Repository       { Install-Repository }
function Stage-Venv             { Resolve-UvCmd; Install-Venv }
function Stage-Dependencies     { Resolve-UvCmd; Install-Dependencies }
function Stage-NodeDeps         { Install-NodeDeps }
function Stage-Path             { Set-PathVariable }
function Stage-ConfigTemplates  { Copy-ConfigTemplates }
function Stage-PlatformSdks     { Resolve-UvCmd; Install-PlatformSdks }
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

    if ($Stage) {
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
        # structured error frame and exit non-zero.
        $err = @{
            ok     = $false
            stage  = if ($Stage) { $Stage } else { $null }
            reason = "$_"
        }
        $err | ConvertTo-Json -Compress | Write-Output
        exit 1
    }

    # Interactive mode: keep today's friendly recovery hint.
    Write-Host ""
    Write-Err "Installation failed: $_"
    Write-Host ""
    Write-Info "If the error is unclear, try downloading and running the script directly:"
    Write-Host "  Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1' -OutFile install.ps1" -ForegroundColor Yellow
    Write-Host "  .\install.ps1" -ForegroundColor Yellow
    Write-Host ""
}
