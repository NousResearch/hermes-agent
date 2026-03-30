
# ============================================================================
#  Hermes Agent - Windows Installer (builds from current branch)
#
#  One-liner:
#    irm https://raw.githubusercontent.com/claudlos/hermes-agent/windows-qol-v2/scripts/install-windows.ps1 | iex
#
#  Or run locally:
#    .\scripts\install-windows.ps1
#
#  Optional custom Python:
#    .\scripts\install-windows.ps1 -PythonExe C:\path\to\python.exe
# ============================================================================

param(
    [string]$PythonExe = "",
    [ValidateSet("none", "nous", "staff")]
    [string]$DesktopIcon = "nous"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Where to install
$INSTALL_DIR = "$env:LOCALAPPDATA\hermes-agent"
$VENV_DIR = "$INSTALL_DIR\.venv"
$BIN_DIR = "$env:LOCALAPPDATA\Programs\hermes"

# Detect: are we running from inside a repo checkout, or downloaded standalone?
$SCRIPT_PATH = $MyInvocation.MyCommand.Path
$SCRIPT_DIR = if ($SCRIPT_PATH) { Split-Path -Parent $SCRIPT_PATH } elseif ($PSScriptRoot) { $PSScriptRoot } else { $pwd.Path }
$REPO_ROOT = Split-Path -Parent $SCRIPT_DIR
$FROM_REPO = (Test-Path "$REPO_ROOT\.git") -and (Test-Path "$REPO_ROOT\pyproject.toml")

# If running from repo, use that repo and its current branch
if ($FROM_REPO) {
    $SOURCE_DIR = $REPO_ROOT
    $BRANCH = (& git -C $REPO_ROOT rev-parse --abbrev-ref HEAD 2>$null)
    if (-not $BRANCH) { $BRANCH = "windows-qol-v2" }
    $REPO_URL = "https://github.com/claudlos/hermes-agent.git"
} else {
    # Downloaded standalone - clone the same branch/fork this installer comes from.
    $SOURCE_DIR = $null
    $REPO_URL = "https://github.com/claudlos/hermes-agent.git"
    $BRANCH = "windows-qol-v2"
}

# -- Helpers -----------------------------------------------------------------
function Write-Step($n, $msg) { Write-Host "`n  [$n] " -NoNewline -ForegroundColor DarkYellow; Write-Host $msg }
function Write-Ok($msg)      { Write-Host "      $msg" -ForegroundColor Green }
function Write-Dim($msg)     { Write-Host "      $msg" -ForegroundColor DarkGray }
function Write-Err($msg)     { Write-Host "      $msg" -ForegroundColor Red }
function Fail($msg)          { Write-Err $msg; exit 1 }

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
if ($PythonExe) {
    if (-not (Test-Path $PythonExe)) {
        Write-Err "Requested Python not found: $PythonExe"
        exit 1
    }
    try {
        $ver = & $PythonExe --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) {
                $python = $PythonExe
                Write-Ok "Using explicit Python: $ver"
            }
        }
    } catch {}
    if (-not $python) {
        Write-Err "Explicit Python must be 3.10+: $PythonExe"
        exit 1
    }
} else {
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
}

if (-not $python) {
    Write-Err "Python 3.10+ required but not found."
    Write-Host ""
    Write-Host "  Download from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  Check 'Add Python to PATH' during install." -ForegroundColor DarkGray
    Write-Host "  Or pass a custom interpreter: .\scripts\install-windows.ps1 -PythonExe C:\path\to\python.exe" -ForegroundColor DarkGray
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
    # Prefer a shallow clone over a full robocopy sync (much faster for large repos).
    # Fall back to robocopy only if git clone fails.
    if ($SOURCE_DIR -ne $INSTALL_DIR) {
        $clonedOk = $false
        if (Test-Path "$INSTALL_DIR\.git") {
            Write-Dim "Updating existing clone at $INSTALL_DIR..."
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            Push-Location $INSTALL_DIR
            $null = & git fetch origin $BRANCH --depth 1 2>&1
            $null = & git checkout $BRANCH 2>&1
            $null = & git reset --hard "origin/$BRANCH" 2>&1
            Pop-Location
            $clonedOk = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $prevEAP
        } else {
            Write-Dim "Shallow-cloning $REPO_URL ($BRANCH) to $INSTALL_DIR..."
            if (Test-Path $INSTALL_DIR) {
                Remove-Item $INSTALL_DIR -Recurse -Force
            }
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            $cloneOut = & git clone --depth 1 --branch $BRANCH $REPO_URL $INSTALL_DIR 2>&1
            $clonedOk = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $prevEAP
        }
        if ($clonedOk) {
            Write-Ok "Source ready via shallow clone ($BRANCH)"
        } else {
            Write-Dim "Shallow clone failed; falling back to robocopy sync..."
            if (-not (Test-Path $INSTALL_DIR)) {
                New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
            }
            & robocopy $SOURCE_DIR $INSTALL_DIR /MIR /XD .git .venv __pycache__ .pytest_cache node_modules PCbuild externals /XF "*.pyc" /NFL /NDL /NJH /NJS /NP 2>&1 | Out-Null
            $robocopyCode = $LASTEXITCODE
            if ($robocopyCode -ge 8) {
                Fail "robocopy sync failed with exit code $robocopyCode"
            }
            Push-Location $INSTALL_DIR
            if (-not (Test-Path ".git")) {
                & git init --quiet 2>&1 | Out-Null
                & git add -A 2>&1 | Out-Null
                & git commit -m "install snapshot from $BRANCH" --quiet 2>&1 | Out-Null
            }
            Pop-Location
            Write-Ok "Source synced via robocopy ($BRANCH)"
        }
    } else {
        Write-Ok "Installing from current directory"
    }
} else {
    # Clone from GitHub
    if (Test-Path "$INSTALL_DIR\.git") {
        Write-Dim "Updating existing clone..."
        Push-Location $INSTALL_DIR
        & git fetch origin $BRANCH 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Fail "git fetch failed for branch $BRANCH"
        }
        & git checkout $BRANCH 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Fail "git checkout failed for branch $BRANCH"
        }
        & git pull origin $BRANCH 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Fail "git pull failed for branch $BRANCH"
        }
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
$installTarget = "$INSTALL_DIR[keyring,pty]"

& $venvPip install -e $installTarget --quiet 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Dim "Retrying with full output..."
    & $venvPip install -e $installTarget
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
$launcherLines = @(
    '@echo off',
    ('"{0}" %*' -f $hermesExe)
)
Set-Content "$BIN_DIR\hermes.bat" -Value $launcherLines -Encoding ASCII

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

# -- Step 8: Desktop shortcut ------------------------------------------------
if ($DesktopIcon -ne 'none') {
    Write-Step 8 'Creating desktop shortcut...'

    $iconDir = $null
    if ($SCRIPT_PATH) {
        $iconDir = Join-Path (Split-Path -Parent $SCRIPT_PATH) 'icons'
    }
    if (-not $iconDir -or -not (Test-Path $iconDir)) {
        $iconDir = Join-Path $INSTALL_DIR 'scripts\icons'
    }

    $iconFile = if ($DesktopIcon -eq 'staff') { 'hermes-staff.ico' } else { 'hermes-nous.ico' }
    $iconPath = Join-Path $iconDir $iconFile

    $desktopPath = [Environment]::GetFolderPath('Desktop')
    $shortcutPath = Join-Path $desktopPath 'Hermes Agent.lnk'

    if (Test-Path $iconPath) {
        $ws = New-Object -ComObject WScript.Shell
        $sc = $ws.CreateShortcut($shortcutPath)
        $sc.TargetPath = 'cmd.exe'
        $sc.Arguments = '/k hermes'
        $sc.WorkingDirectory = '%USERPROFILE%'
        $sc.IconLocation = $iconPath
        $sc.Description = 'Hermes Agent'
        $sc.Save()
        Write-Ok ('Desktop shortcut created ({0})' -f $DesktopIcon)
    } else {
        Write-Dim ('Icon not found: {0} — skipping shortcut' -f $iconPath)
    }
}

# -- Step 9: Quick verify ----------------------------------------------------
Write-Step 9 'Verifying install...'

$verifyOut = & $hermesExe --version 2>&1
if ($verifyOut) {
    Write-Ok $verifyOut
} else {
    Write-Ok 'Binary runs'
}

# -- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "  ============================================" -ForegroundColor Green
Write-Host "   Hermes Agent installed successfully" -ForegroundColor Green
Write-Host "  ============================================" -ForegroundColor Green
Write-Host ""
Write-Host '  Next step — open a NEW terminal and run:' -ForegroundColor White
Write-Host ''
Write-Host '      hermes setup' -ForegroundColor Yellow
Write-Host ''
Write-Host '  This configures your API key and preferences.' -ForegroundColor DarkGray
Write-Host ''
if ($FROM_REPO) {
    Write-Host ('  Built from: {0} ({1})' -f $SOURCE_DIR, $BRANCH) -ForegroundColor DarkGray
}
Write-Host ('  Installed:  {0}' -f $INSTALL_DIR) -ForegroundColor DarkGray
Write-Host ('  Launcher:   {0}\hermes.bat' -f $BIN_DIR) -ForegroundColor DarkGray
Write-Host ''
