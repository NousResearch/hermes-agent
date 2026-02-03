# ============================================================================
# Hermes Agent Installer for Windows
# ============================================================================
# Installation script for Windows (PowerShell).
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

# ============================================================================
# Helper functions
# ============================================================================

function Write-Banner {
    Write-Host ""
    Write-Host "┌─────────────────────────────────────────────────────────┐" -ForegroundColor Magenta
    Write-Host "│             🦋 Hermes Agent Installer                   │" -ForegroundColor Magenta
    Write-Host "├─────────────────────────────────────────────────────────┤" -ForegroundColor Magenta
    Write-Host "│  I'm just a butterfly with a lot of tools.             │" -ForegroundColor Magenta
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

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# ============================================================================
# Dependency checks
# ============================================================================

function Test-Python {
    Write-Info "Checking Python..."
    
    # Try different python commands
    $pythonCmds = @("python3", "python", "py -3")
    
    foreach ($cmd in $pythonCmds) {
        try {
            $version = & $cmd.Split()[0] $cmd.Split()[1..99] -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($version) {
                $major, $minor = $version.Split('.')
                if ([int]$major -ge 3 -and [int]$minor -ge 10) {
                    $script:PythonCmd = $cmd
                    Write-Success "Python $version found"
                    return $true
                }
            }
        } catch {
            # Try next command
        }
    }
    
    Write-Error "Python 3.10+ not found"
    Write-Info "Please install Python 3.10 or newer from:"
    Write-Info "  https://www.python.org/downloads/"
    Write-Info ""
    Write-Info "Make sure to check 'Add Python to PATH' during installation"
    return $false
}

function Test-Git {
    Write-Info "Checking Git..."
    
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $version = git --version
        Write-Success "Git found ($version)"
        return $true
    }
    
    Write-Error "Git not found"
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
    
    Write-Warning "Node.js not found (browser tools will be limited)"
    Write-Info "To install Node.js (optional):"
    Write-Info "  https://nodejs.org/en/download/"
    $script:HasNode = $false
    return $true  # Don't fail - Node is optional
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
            Write-Error "Directory exists but is not a git repository: $InstallDir"
            Write-Info "Remove it or choose a different directory with -InstallDir"
            exit 1
        }
    } else {
        # Try SSH first (for private repo access), fall back to HTTPS
        Write-Info "Trying SSH clone..."
        $sshResult = git clone --branch $Branch $RepoUrlSsh $InstallDir 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Cloned via SSH"
        } else {
            Write-Info "SSH failed, trying HTTPS..."
            $httpsResult = git clone --branch $Branch $RepoUrlHttps $InstallDir 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Cloned via HTTPS"
            } else {
                Write-Error "Failed to clone repository"
                Write-Info "For private repo access, ensure your SSH key is added to GitHub:"
                Write-Info "  ssh-add ~/.ssh/id_rsa"
                Write-Info "  ssh -T git@github.com  # Test connection"
                exit 1
            }
        }
    }
    
    Write-Success "Repository ready"
}

function Install-Venv {
    if ($NoVenv) {
        Write-Info "Skipping virtual environment (-NoVenv)"
        return
    }
    
    Write-Info "Creating virtual environment..."
    
    Push-Location $InstallDir
    
    if (-not (Test-Path "venv")) {
        & $PythonCmd -m venv venv
    }
    
    # Activate
    & .\venv\Scripts\Activate.ps1
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools | Out-Null
    
    Pop-Location
    
    Write-Success "Virtual environment ready"
}

function Install-Dependencies {
    Write-Info "Installing dependencies..."
    
    Push-Location $InstallDir
    
    if (-not $NoVenv) {
        & .\venv\Scripts\Activate.ps1
    }
    
    try {
        pip install -e ".[all]" 2>&1 | Out-Null
    } catch {
        pip install -e "." | Out-Null
    }
    
    Pop-Location
    
    Write-Success "Dependencies installed"
}

function Set-PathVariable {
    Write-Info "Setting up PATH..."
    
    if ($NoVenv) {
        $binDir = "$InstallDir"
    } else {
        $binDir = "$InstallDir\venv\Scripts"
    }
    
    # Add to user PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($currentPath -notlike "*$binDir*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$binDir;$currentPath",
            "User"
        )
        Write-Success "Added to user PATH"
    } else {
        Write-Info "PATH already configured"
    }
    
    # Update current session
    $env:Path = "$binDir;$env:Path"
}

function Copy-ConfigTemplates {
    Write-Info "Setting up configuration files..."
    
    # Create ~/.hermes directory structure (config at top level, code in subdir)
    New-Item -ItemType Directory -Force -Path "$HermesHome\cron" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\sessions" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\logs" | Out-Null
    
    # Create .env at ~/.hermes/.env (top level, easy to find)
    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) {
        $examplePath = "$InstallDir\.env.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $envPath
            Write-Success "Created ~/.hermes/.env from template"
        } else {
            # Create empty .env if no example exists
            New-Item -ItemType File -Force -Path $envPath | Out-Null
            Write-Success "Created ~/.hermes/.env"
        }
    } else {
        Write-Info "~/.hermes/.env already exists, keeping it"
    }
    
    # Create config.yaml at ~/.hermes/config.yaml (top level, easy to find)
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
    
    Write-Success "Configuration directory ready: ~/.hermes/"
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
            Write-Warning "npm install failed (browser tools may not work)"
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
    
    if (-not $NoVenv) {
        & .\venv\Scripts\Activate.ps1
    }
    
    python -m hermes_cli.main setup
    
    Pop-Location
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
}

# ============================================================================
# Main
# ============================================================================

function Main {
    Write-Banner
    
    if (-not (Test-Python)) { exit 1 }
    if (-not (Test-Git)) { exit 1 }
    Test-Node  # Optional, doesn't fail
    
    Install-Repository
    Install-Venv
    Install-Dependencies
    Install-NodeDeps
    Set-PathVariable
    Copy-ConfigTemplates
    Invoke-SetupWizard
    
    Write-Completion
}

Main
