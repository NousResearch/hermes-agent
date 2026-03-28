
# ============================================================================
#  Hermes Agent - Windows Installer (builds from current branch)
#
#  One-liner:
#    irm https://raw.githubusercontent.com/claudlos/hermes-agent/windows-qol-local/scripts/install-windows.ps1 | iex
#
#  Or run locally:
#    .\scripts\install-windows.ps1
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Where to install
$INSTALL_DIR = "$env:LOCALAPPDATA\hermes-agent"
$VENV_DIR = "$INSTALL_DIR\.venv"
$BIN_DIR = "$env:LOCALAPPDATA\Programs\hermes"

# Detect: are we running from inside a repo checkout, or downloaded standalone?
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_ROOT = Split-Path -Parent $SCRIPT_DIR
$FROM_REPO = (Test-Path "$REPO_ROOT\.git") -and (Test-Path "$REPO_ROOT\pyproject.toml")

# If running from repo, use that repo and its current branch
if ($FROM_REPO) {
    $SOURCE_DIR = $REPO_ROOT
    $BRANCH = (& git -C $REPO_ROOT rev-parse --abbrev-ref HEAD 2>$null)
    if (-not $BRANCH) { $BRANCH = "unknown" }
} else {
    # Downloaded standalone - clone from GitHub
    $SOURCE_DIR = $null
    $REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
    $BRANCH = "main"
}

# -- Helpers -----------------------------------------------------------------
function Write-Step($n, $msg) { Write-Host "`n  [$n] " -NoNewline -ForegroundColor DarkYellow; Write-Host $msg }
function Write-Ok($msg)      { Write-Host "      $msg" -ForegroundColor Green }
function Write-Dim($msg)     { Write-Host "      $msg" -ForegroundColor DarkGray }
function Write-Err($msg)     { Write-Host "      $msg" -ForegroundColor Red }

# -- Banner ------------------------------------------------------------------
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkYellow
Write-Host "   Hermes Agent - Windows Installer" -ForegroundColor Yellow
Write-Host "  ============================================" -ForegroundColor DarkYellow
if ($FROM_REPO) {
    Write-Host "  Mode:   Local build (current branch)" -ForegroundColor DarkGray
    Write-Host "  Branch: $BRANCH" -ForegroundColor DarkGray
    Write-Host "  Source: $SOURCE_DIR" -ForegroundColor DarkGray
} else {
    Write-Host "  Mode:   Fresh install from GitHub" -ForegroundColor DarkGray
}

# -- Step 1: Python ----------------------------------------------------------
Write-Step 1 "Checking Python..."

$python = $null
foreach ($cmd in @("python3", "python", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                $python = $cmd
                Write-Ok "Found $ver"
                break
            }
        }
    } catch {}
}

if (-not $python) {
    Write-Err "Python 3.10+ required but not found."
    Write-Host ""
    Write-Host "  Download from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  Check 'Add Python to PATH' during install." -ForegroundColor DarkGray
    Write-Host ""
    exit 1
}

# -- Step 2: Git (only needed for clone mode) --------------------------------
if (-not $FROM_REPO) {
    Write-Step 2 "Checking Git..."
    try {
        $gitVer = & git --version 2>&1
        Write-Ok $gitVer
    } catch {
        Write-Err "Git not found. Install from https://git-scm.com/download/win"
        exit 1
    }
} else {
    Write-Step 2 "Using local repo"
    Write-Ok $SOURCE_DIR
}

# -- Step 3: Source code -----------------------------------------------------
Write-Step 3 "Preparing source..."

if ($FROM_REPO) {
    # Building from current checkout - copy to install dir if different
    if ($SOURCE_DIR -ne $INSTALL_DIR) {
        Write-Dim "Syncing to $INSTALL_DIR..."
        if (-not (Test-Path $INSTALL_DIR)) {
            New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
        }
        # Use robocopy for fast sync, exclude .git, .venv, __pycache__, build artifacts
        & robocopy $SOURCE_DIR $INSTALL_DIR /MIR /XD .git .venv __pycache__ .pytest_cache node_modules PCbuild externals /XF "*.pyc" /NFL /NDL /NJH /NJS /NP 2>&1 | Out-Null
        # Init git in install dir so editable install works
        Push-Location $INSTALL_DIR
        if (-not (Test-Path ".git")) {
            & git init --quiet 2>&1 | Out-Null
            & git add -A 2>&1 | Out-Null
            & git commit -m "install snapshot from $BRANCH" --quiet 2>&1 | Out-Null
        }
        Pop-Location
        Write-Ok "Source synced ($BRANCH)"
    } else {
        Write-Ok "Installing from current directory"
    }
} else {
    # Clone from GitHub
    if (Test-Path "$INSTALL_DIR\.git") {
        Write-Dim "Updating existing clone..."
        Push-Location $INSTALL_DIR
        & git fetch origin $BRANCH 2>&1 | Out-Null
        & git checkout $BRANCH 2>&1 | Out-Null
        & git pull origin $BRANCH 2>&1 | Out-Null
        Pop-Location
        Write-Ok "Updated to latest $BRANCH"
    } else {
        if (Test-Path $INSTALL_DIR) {
            Remove-Item $INSTALL_DIR -Recurse -Force
        }
        Write-Dim "Cloning..."
        & git clone --depth 1 --branch $BRANCH $REPO_URL $INSTALL_DIR 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Clone failed"; exit 1
        }
        Write-Ok "Cloned $BRANCH"
    }
}

# -- Step 4: Virtual environment ---------------------------------------------
Write-Step 4 "Setting up Python environment..."

if (-not (Test-Path "$VENV_DIR\Scripts\python.exe")) {
    Write-Dim "Creating virtual environment..."
    & $python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create venv"; exit 1
    }
}

$venvPython = "$VENV_DIR\Scripts\python.exe"
$venvPip = "$VENV_DIR\Scripts\pip.exe"

# Upgrade pip silently
& $venvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
Write-Ok "Virtual environment ready"

# -- Step 5: Install ---------------------------------------------------------
Write-Step 5 "Installing Hermes Agent..."
Write-Dim "This may take a minute on first install..."

& $venvPip install -e "$INSTALL_DIR" --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Dim "Retrying with full output..."
    & $venvPip install -e "$INSTALL_DIR"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Install failed"; exit 1
    }
}

# Verify the binary exists
$hermesExe = "$VENV_DIR\Scripts\hermes.exe"
if (-not (Test-Path $hermesExe)) {
    Write-Err "hermes.exe not found after install"
    exit 1
}

Write-Ok "Hermes Agent installed"

# -- Step 6: Create launcher -------------------------------------------------
Write-Step 6 "Creating launcher..."

if (-not (Test-Path $BIN_DIR)) {
    New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null
}

# hermes.bat - wrapper that activates venv and runs hermes
@"
@echo off
"$hermesExe" %*
"@ | Set-Content "$BIN_DIR\hermes.bat" -Encoding ASCII

Write-Ok "hermes.bat -> $BIN_DIR"

# -- Step 7: PATH ------------------------------------------------------------
Write-Step 7 "Configuring PATH..."

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BIN_DIR*") {
    [Environment]::SetEnvironmentVariable("Path", "$BIN_DIR;$userPath", "User")
    $env:Path = "$BIN_DIR;$env:Path"
    Write-Ok "Added to PATH"
    Write-Dim "Restart your terminal for PATH to take effect"
} else {
    Write-Ok "Already in PATH"
}

# -- Step 8: Quick verify ----------------------------------------------------
Write-Step 8 "Verifying install..."

$verifyOut = & $hermesExe --version 2>&1
if ($verifyOut) {
    Write-Ok $verifyOut
} else {
    Write-Ok "Binary runs"
}

# -- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "  ============================================" -ForegroundColor Green
Write-Host "   Hermes Agent installed successfully" -ForegroundColor Green
Write-Host "  ============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Next step — open a NEW terminal and run:" -ForegroundColor White
Write-Host ""
Write-Host "      hermes setup" -ForegroundColor Yellow
Write-Host ""
Write-Host "  This configures your API key and preferences." -ForegroundColor DarkGray
Write-Host ""
if ($FROM_REPO) {
    Write-Host "  Built from: $SOURCE_DIR ($BRANCH)" -ForegroundColor DarkGray
}
Write-Host "  Installed:  $INSTALL_DIR" -ForegroundColor DarkGray
Write-Host "  Launcher:   $BIN_DIR\hermes.bat" -ForegroundColor DarkGray
Write-Host ""
