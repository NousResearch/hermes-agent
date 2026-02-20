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
    [string]$HermesHome = "$env:USERPROFILE\.hermes",
    [string]$InstallDir = "$env:USERPROFILE\.hermes\hermes-agent"
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================

$RepoUrlSsh = "git@github.com:NousResearch/hermes-agent.git"
$RepoUrlHttps = "https://github.com/NousResearch/hermes-agent.git"
$PythonVersion = "3.11"

# ============================================================================
# Helper functions
# ============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
    Write-Host "│             🦋 Hermes Agent Installer                   │" -ForegroundColor Magenta
    Write-Host "├─────────────────────────────────────────────────────────┤" -ForegroundColor Magenta
    Write-Host "│  An open source AI agent by Nous Research.              │" -ForegroundColor Magenta
    Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Magenta
    Write-Host ""
}

function Write-Info {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
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
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Out-Null
        
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
        Write-Err "Failed to install uv"
        Write-Info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    }
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
    
    # Python not found — use uv to install it (no admin needed!)
    Write-Info "Python $PythonVersion not found, installing via uv..."
    try {
        & $UvCmd python install $PythonVersion 2>&1 | Out-Null
        $pythonPath = & $UvCmd python find $PythonVersion 2>$null
        if ($pythonPath) {
            $ver = & $pythonPath --version 2>$null
            Write-Success "Python installed: $ver"
            return $true
        }
    } catch { }
    
    Write-Err "Failed to install Python $PythonVersion"
    Write-Info "Install Python $PythonVersion manually, then re-run this script"
    return $false
}

function Test-Git {
    Write-Info "Checking Git..."
    
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $version = git --version
        Write-Success "Git found ($version)"
        return $true
    }
    
    Write-Err "Git not found"
    Write-Info "Please install Git from:"
    Write-Info "  https://git-scm.com/download/win"
    return $false
}

function Test-Node {
    Write-Info "Checking Node.js (optional, for browser tools)..."
    
    if (Get-Command node -ErrorAction SilentlyContinue) {
        $version = node --version
        Write-Success "Node.js $version found"
        $script:HasNode = $true
        return $true
    }
    
    Write-Warn "Node.js not found (browser tools will be limited)"
    Write-Info "To install Node.js (optional):"
    Write-Info "  https://nodejs.org/en/download/"
    $script:HasNode = $false
    return $true  # Don't fail - Node is optional
}

function Test-Ripgrep {
    Write-Info "Checking ripgrep (optional, for faster file search)..."
    
    if (Get-Command rg -ErrorAction SilentlyContinue) {
        $version = rg --version | Select-Object -First 1
        Write-Success "$version found"
        $script:HasRipgrep = $true
        return $true
    }
    
    Write-Warn "ripgrep not found (file search will use findstr fallback)"
    
    # Check what package managers are available
    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    $hasChoco = Get-Command choco -ErrorAction SilentlyContinue
    $hasScoop = Get-Command scoop -ErrorAction SilentlyContinue
    
    # Offer to install
    Write-Host ""
    $response = Read-Host "Would you like to install ripgrep? (faster search, recommended) [Y/n]"
    
    if ($response -eq "" -or $response -match "^[Yy]") {
        Write-Info "Installing ripgrep..."
        
        if ($hasWinget) {
            try {
                winget install BurntSushi.ripgrep.MSVC --silent 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "ripgrep installed via winget"
                    $script:HasRipgrep = $true
                    return $true
                }
            } catch { }
        }
        
        if ($hasChoco) {
            try {
                choco install ripgrep -y 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "ripgrep installed via chocolatey"
                    $script:HasRipgrep = $true
                    return $true
                }
            } catch { }
        }
        
        if ($hasScoop) {
            try {
                scoop install ripgrep 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "ripgrep installed via scoop"
                    $script:HasRipgrep = $true
                    return $true
                }
            } catch { }
        }
        
        Write-Warn "Auto-install failed. You can install manually:"
    } else {
        Write-Info "Skipping ripgrep installation. To install manually:"
    }
    
    # Show manual install instructions
    Write-Info "  winget install BurntSushi.ripgrep.MSVC"
    Write-Info "  Or: choco install ripgrep"
    Write-Info "  Or: scoop install ripgrep"
    Write-Info "  Or download from: https://github.com/BurntSushi/ripgrep/releases"
    
    $script:HasRipgrep = $false
    return $true  # Don't fail - ripgrep is optional
}

function Test-Ffmpeg {
    Write-Info "Checking ffmpeg (optional, for TTS voice messages)..."
    
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        $version = ffmpeg -version 2>&1 | Select-Object -First 1
        Write-Success "ffmpeg found"
        $script:HasFfmpeg = $true
        return $true
    }
    
    Write-Warn "ffmpeg not found (TTS voice bubbles on Telegram will send as audio files instead)"
    Write-Info "  Install with: winget install ffmpeg"
    Write-Info "  Or: choco install ffmpeg"
    Write-Info "  Or download from: https://ffmpeg.org/download.html"
    
    $script:HasFfmpeg = $false
    return $true  # Don't fail - ffmpeg is optional
}

# ============================================================================
# Installation
# ============================================================================

function Install-Repository {
    Write-Info "Installing to $InstallDir..."
    
    if (Test-Path $InstallDir) {
        if (Test-Path "$InstallDir\.git") {
            Write-Info "Existing installation found, updating..."
            Push-Location $InstallDir
            git fetch origin
            git checkout $Branch
            git pull origin $Branch
            Pop-Location
        } else {
            Write-Err "Directory exists but is not a git repository: $InstallDir"
            Write-Info "Remove it or choose a different directory with -InstallDir"
            exit 1
        }
    } else {
        # Try SSH first (for private repo access), fall back to HTTPS
        Write-Info "Trying SSH clone..."
        $sshResult = git clone --branch $Branch --recurse-submodules $RepoUrlSsh $InstallDir 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Cloned via SSH"
        } else {
            Write-Info "SSH failed, trying HTTPS..."
            $httpsResult = git clone --branch $Branch --recurse-submodules $RepoUrlHttps $InstallDir 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Cloned via HTTPS"
            } else {
                Write-Err "Failed to clone repository"
                Write-Info "For private repo access, ensure your SSH key is added to GitHub:"
                Write-Info "  ssh-add ~/.ssh/id_rsa"
                Write-Info "  ssh -T git@github.com  # Test connection"
                exit 1
            }
        }
    }
    
    # Ensure submodules are initialized and updated
    Write-Info "Initializing submodules (mini-swe-agent, tinker-atropos)..."
    Push-Location $InstallDir
    git submodule update --init --recursive
    Pop-Location
    Write-Success "Submodules ready"
    
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
    
    # Install main package with all extras
    try {
        & $UvCmd pip install -e ".[all]" 2>&1 | Out-Null
    } catch {
        & $UvCmd pip install -e "." | Out-Null
    }
    
    Write-Success "Main package installed"
    
    # Install submodules
    Write-Info "Installing mini-swe-agent (terminal tool backend)..."
    if (Test-Path "mini-swe-agent\pyproject.toml") {
        try {
            & $UvCmd pip install -e ".\mini-swe-agent" 2>&1 | Out-Null
            Write-Success "mini-swe-agent installed"
        } catch {
            Write-Warn "mini-swe-agent install failed (terminal tools may not work)"
        }
    } else {
        Write-Warn "mini-swe-agent not found (run: git submodule update --init)"
    }
    
    Write-Info "Installing tinker-atropos (RL training backend)..."
    if (Test-Path "tinker-atropos\pyproject.toml") {
        try {
            & $UvCmd pip install -e ".\tinker-atropos" 2>&1 | Out-Null
            Write-Success "tinker-atropos installed"
        } catch {
            Write-Warn "tinker-atropos install failed (RL tools may not work)"
        }
    } else {
        Write-Warn "tinker-atropos not found (run: git submodule update --init)"
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
    
    # Create SOUL.md if it doesn't exist (global persona file)
    $soulPath = "$HermesHome\SOUL.md"
    if (-not (Test-Path $soulPath)) {
        @"
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
"@ | Set-Content -Path $soulPath -Encoding UTF8
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
    
    Push-Location $InstallDir
    
    if (Test-Path "package.json") {
        Write-Info "Installing Node.js dependencies..."
        try {
            npm install --silent 2>&1 | Out-Null
            Write-Success "Node.js dependencies installed"
        } catch {
            Write-Warn "npm install failed (browser tools may not work)"
        }
    }
    
    Pop-Location
}

function Invoke-SetupWizard {
    if ($SkipSetup) {
        Write-Info "Skipping setup wizard (-SkipSetup)"
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

    Write-Host ""
    Write-Info "Messaging platform token detected!"
    Write-Info "The gateway needs to be running for Hermes to send/receive messages."
    Write-Host ""
    $response = Read-Host "Would you like to start the gateway now? [Y/n]"

    if ($response -eq "" -or $response -match "^[Yy]") {
        $hermesCmd = "$InstallDir\venv\Scripts\hermes.exe"
        if (-not (Test-Path $hermesCmd)) {
            $hermesCmd = "hermes"
        }

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
    Write-Host "┌─────────────────────────────────────────────────────────┐" -ForegroundColor Green
    Write-Host "│              ✓ Installation Complete!                   │" -ForegroundColor Green
    Write-Host "└─────────────────────────────────────────────────────────┘" -ForegroundColor Green
    Write-Host ""
    
    # Show file locations
    Write-Host "📁 Your files (all in ~/.hermes/):" -ForegroundColor Cyan
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
    
    Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🚀 Commands:" -ForegroundColor Cyan
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
    Write-Host "Run messaging gateway"
    Write-Host "   hermes update       " -NoNewline -ForegroundColor Green
    Write-Host "Update to latest version"
    Write-Host ""
    
    Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚡ Restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $HasNode) {
        Write-Host "Note: Node.js was not found. Browser automation tools" -ForegroundColor Yellow
        Write-Host "will have limited functionality." -ForegroundColor Yellow
        Write-Host ""
    }
    
    if (-not $HasRipgrep) {
        Write-Host "Note: ripgrep (rg) was not found. File search will use" -ForegroundColor Yellow
        Write-Host "findstr as a fallback. For faster search:" -ForegroundColor Yellow
        Write-Host "  winget install BurntSushi.ripgrep.MSVC" -ForegroundColor Yellow
        Write-Host ""
    }
}

# ============================================================================
# Main
# ============================================================================

function Main {
    Write-Banner
    
    if (-not (Install-Uv)) { exit 1 }
    if (-not (Test-Python)) { exit 1 }
    if (-not (Test-Git)) { exit 1 }
    Test-Node      # Optional, doesn't fail
    Test-Ripgrep   # Optional, doesn't fail
    Test-Ffmpeg    # Optional, doesn't fail
    
    Install-Repository
    Install-Venv
    Install-Dependencies
    Install-NodeDeps
    Set-PathVariable
    Copy-ConfigTemplates
    Invoke-SetupWizard
    Start-GatewayIfConfigured
    
    Write-Completion
}

Main
