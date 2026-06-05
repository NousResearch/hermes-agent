<#
.SYNOPSIS
    Prepare offline deployment bundle for Hermes Agent (Linux x86_64 target).

.DESCRIPTION
    Downloads all dependencies needed for a fully offline Linux installation.
    Run on a Windows machine with internet access.

.PARAMETER OutputDir
    Output directory. Default: hermes-offline-bundle in project root.

.PARAMETER PythonVersion
    Target Python version. Default: 3.11

.PARAMETER NodeVersion
    Node.js major version. Default: 22

.PARAMETER SkipBrowser
    Skip Playwright Chromium download.

.PARAMETER SkipNpm
    Skip npm offline cache preparation.

.PARAMETER SkipDebs
    Skip deb system dependency download.
#>

[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [string]$PythonVersion = "3.11",
    [string]$NodeVersion = "22",
    [switch]$SkipBrowser,
    [switch]$SkipNpm,
    [switch]$SkipDebs
)

$ErrorActionPreference = "Continue"

# -- Helpers --

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Write-SubStep {
    param([string]$Message)
    Write-Host "    -> $Message" -ForegroundColor Gray
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "    [WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "    [ERROR] $Message" -ForegroundColor Red
}

function Test-CommandExists {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Get-FileSizeMB {
    param([string]$Path)
    if (Test-Path $Path) {
        return [math]::Round((Get-Item $Path).Length / 1MB, 1)
    }
    return 0
}

function Invoke-Download {
    param(
        [string]$Url,
        [string]$OutFile,
        [string]$Description
    )
    Write-SubStep "Downloading $Description ..."
    Write-SubStep "  URL: $Url"
    try {
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing -TimeoutSec 300
        $size = Get-FileSizeMB $OutFile
        Write-Ok "$Description ($size MB)"
        return $true
    } catch {
        Write-Err "Download failed: $_"
        return $false
    }
}

# -- Parameters --

if (-not $OutputDir) {
    $OutputDir = Join-Path (Join-Path $PSScriptRoot "..") "hermes-offline-bundle"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

# -- Banner --

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hermes Agent Offline Bundle Builder" -ForegroundColor Cyan
Write-Host "  Target: Linux x86_64 / Python $PythonVersion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# -- Prerequisites --

Write-Step "Checking prerequisites"

if (-not (Test-CommandExists "git")) {
    Write-Err "git is required. Install Git for Windows first."
    exit 1
}
Write-Ok "git found"

if (-not (Test-CommandExists "uv")) {
    Write-Err "uv is required. Install: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'"
    exit 1
}
Write-Ok "uv found"

if (-not (Test-CommandExists "node")) {
    Write-Err "Node.js is required (for npm pack and Playwright). Install from https://nodejs.org"
    exit 1
}
$nodeVer = & node --version 2>$null
Write-Ok "Node.js $nodeVer found"

# -- Create directory structure --

Write-Step "Creating directory structure: $OutputDir"

$dirs = @(
    $OutputDir,
    (Join-Path $OutputDir "binaries"),
    (Join-Path $OutputDir "python-wheels"),
    (Join-Path $OutputDir "playwright-browsers"),
    (Join-Path $OutputDir "npm-offline"),
    (Join-Path $OutputDir "deb-packages")
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Ok "Directories created"

# ============================================================
# 1/6 Download Linux binaries
# ============================================================

Write-Step "1/6 Downloading Linux binaries"

# --- Node.js ---
$nodeArch = "linux-x64"
$nodeListUrl = "https://nodejs.org/dist/latest-v$NodeVersion.x/"
try {
    $ProgressPreference = "SilentlyContinue"
    $html = Invoke-WebRequest -Uri $nodeListUrl -UseBasicParsing -TimeoutSec 30
    $pattern = "node-v(\d+\.\d+\.\d+)-$nodeArch\.tar\.xz"
    if ($html.Content -match $pattern) {
        $nodeFullVersion = $Matches[1]
        $nodeFileName = "node-v${nodeFullVersion}-${nodeArch}.tar.xz"
        $nodeUrl = "https://nodejs.org/dist/latest-v${NodeVersion}.x/$nodeFileName"
        $nodeOut = Join-Path (Join-Path $OutputDir "binaries") $nodeFileName
        Invoke-Download -Url $nodeUrl -OutFile $nodeOut -Description "Node.js v$nodeFullVersion ($nodeArch)"
    } else {
        Write-Warn "Could not parse Node.js version. Download manually: $nodeListUrl"
    }
} catch {
    Write-Warn "Node.js download failed: $_"
}

# --- ripgrep ---
Write-SubStep "Downloading ripgrep ..."
$rgApiUrl = "https://api.github.com/repos/BurntSushi/ripgrep/releases/latest"
try {
    $ProgressPreference = "SilentlyContinue"
    $rgRelease = Invoke-RestMethod -Uri $rgApiUrl -TimeoutSec 30
    $rgAsset = $rgRelease.assets | Where-Object {
        $_.name -match "x86_64-unknown-linux-musl" -and $_.name -match "\.tar\.gz$"
    } | Select-Object -First 1
    if ($rgAsset) {
        $rgOut = Join-Path (Join-Path $OutputDir "binaries") $rgAsset.name
        Invoke-Download -Url $rgAsset.browser_download_url -OutFile $rgOut -Description "ripgrep $($rgRelease.tag_name)"
    } else {
        Write-Warn "ripgrep linux-musl asset not found. Download manually: https://github.com/BurntSushi/ripgrep/releases"
    }
} catch {
    Write-Warn "ripgrep download failed (may need GitHub token): $_"
}

# --- ffmpeg ---
Write-SubStep "Downloading ffmpeg (static build) ..."
$ffmpegUrl = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
$ffmpegOut = Join-Path (Join-Path $OutputDir "binaries") "ffmpeg-release-amd64-static.tar.xz"
Invoke-Download -Url $ffmpegUrl -OutFile $ffmpegOut -Description "ffmpeg (amd64 static)"

# --- uv (Linux binary) ---
$uvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
$uvOut = Join-Path (Join-Path $OutputDir "binaries") "uv-x86_64-unknown-linux-gnu.tar.gz"
Invoke-Download -Url $uvUrl -OutFile $uvOut -Description "uv (Linux x86_64)"

# ============================================================
# 2/6 Download Python wheels (Linux x86_64)
# ============================================================

Write-Step "2/6 Downloading Python wheels (Linux x86_64)"

$wheelDir = Join-Path $OutputDir "python-wheels"

# Use uv export + uv pip install --target for cross-platform download
# uv pip install --python-platform linux downloads Linux wheels
$projectRoot = Join-Path $PSScriptRoot ".."
$projectRoot = [System.IO.Path]::GetFullPath($projectRoot)

# Generate requirements from lockfile
$requirementsFile = Join-Path $OutputDir "requirements.txt"
Write-SubStep "Exporting requirements from lockfile ..."
Push-Location $projectRoot
try {
    & uv export --extra all --no-hashes -o $requirementsFile 2>&1 | ForEach-Object { Write-SubStep "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "uv export failed, generating minimal requirements ..."
        # Fallback: create a minimal requirements file
        Set-Content -Path $requirementsFile -Value "hermes-agent[all]" -Encoding UTF8
    }
} finally {
    Pop-Location
}

# Download [all] extras wheels for Linux
Write-SubStep "Downloading hermes-agent[all] wheels for Linux x86_64 ..."
& uv pip install `
    --python-version $PythonVersion `
    --python-platform linux `
    --only-binary :all: `
    --target $wheelDir `
    --no-install `
    -r $requirementsFile 2>&1 | ForEach-Object { Write-SubStep "  $_" }

if ($LASTEXITCODE -ne 0) {
    Write-Warn "uv pip install --target failed, retrying without --only-binary ..."
    & uv pip install `
        --python-version $PythonVersion `
        --python-platform linux `
        --target $wheelDir `
        --no-install `
        -r $requirementsFile 2>&1 | ForEach-Object { Write-SubStep "  $_" }
}

# --- Lazy deps ---
Write-SubStep "Downloading lazy deps (all optional backends) ..."

# Complete package list from LAZY_DEPS in tools/lazy_deps.py (deduplicated)
$lazyPackages = @(
    "anthropic==0.87.0",
    "boto3==1.42.89",
    "azure-identity==1.25.3",
    "exa-py==2.10.2",
    "firecrawl-py==4.17.0",
    "parallel-web==0.4.2",
    "mistralai==2.4.8",
    "edge-tts==7.2.7",
    "elevenlabs==1.59.0",
    "faster-whisper==1.2.1",
    "sounddevice==0.5.5",
    "numpy==2.4.3",
    "fal-client==0.13.1",
    "honcho-ai==2.0.1",
    "hindsight-client==0.6.1",
    "python-telegram-bot[webhooks]==22.6",
    "discord.py[voice]==2.7.1",
    "brotlicffi==1.2.0.1",
    "slack-bolt==1.27.0",
    "slack-sdk==3.40.1",
    "aiohttp==3.13.4",
    "mautrix[encryption]==0.21.0",
    "Markdown==3.10.2",
    "aiosqlite==0.22.1",
    "asyncpg==0.31.0",
    "aiohttp-socks==0.11.0",
    "dingtalk-stream==0.24.3",
    "alibabacloud-dingtalk==2.2.42",
    "qrcode==7.4.2",
    "lark-oapi==1.5.3",
    "defusedxml==0.7.1",
    "modal==1.3.4",
    "daytona==0.155.0",
    "google-api-python-client==2.194.0",
    "google-auth-oauthlib==1.3.1",
    "google-auth-httplib2==0.3.1",
    "youtube-transcript-api==1.2.4",
    "agent-client-protocol==0.9.0"
)

foreach ($pkg in $lazyPackages) {
    Write-SubStep "  $pkg"
}

# Write lazy deps to a requirements file
$lazyReqFile = Join-Path $OutputDir "requirements-lazy.txt"
Set-Content -Path $lazyReqFile -Value ($lazyPackages -join "`n") -Encoding UTF8

# Download lazy deps wheels for Linux
Write-SubStep "Downloading lazy deps wheels for Linux x86_64 ..."
& uv pip install `
    --python-version $PythonVersion `
    --python-platform linux `
    --only-binary :all: `
    --target $wheelDir `
    --no-install `
    -r $lazyReqFile 2>&1 | ForEach-Object { Write-SubStep "  $_" }

if ($LASTEXITCODE -ne 0) {
    Write-Warn "uv pip install lazy deps failed, retrying without --only-binary ..."
    & uv pip install `
        --python-version $PythonVersion `
        --python-platform linux `
        --target $wheelDir `
        --no-install `
        -r $lazyReqFile 2>&1 | ForEach-Object { Write-SubStep "  $_" }
}

# Statistics
$wheelCount = (Get-ChildItem $wheelDir -Filter "*.whl" -ErrorAction SilentlyContinue).Count
$wheelSizeMB = [math]::Round((Get-ChildItem $wheelDir -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Ok "Python wheels: $wheelCount packages, $wheelSizeMB MB"

# ============================================================
# 3/6 Package source code
# ============================================================

Write-Step "3/6 Packaging Hermes source code"

$tarOut = Join-Path $OutputDir "hermes-agent.tar.gz"

Push-Location $projectRoot
try {
    & git archive --format=tar.gz -o $tarOut HEAD 2>&1 | ForEach-Object { Write-SubStep "  $_" }
    if ($LASTEXITCODE -eq 0) {
        $tarSize = Get-FileSizeMB $tarOut
        Write-Ok "Source packaged ($tarSize MB)"
    } else {
        Write-Err "git archive failed"
    }
} finally {
    Pop-Location
}

# ============================================================
# 4/6 Download Playwright Chromium (Linux)
# ============================================================
# NOTE: npx playwright install on Windows downloads Windows Chromium.
# We need Linux Chromium. Use Docker to download the Linux version.

if (-not $SkipBrowser) {
    Write-Step "4/6 Downloading Playwright Chromium (Linux x86_64)"

    $pwDir = Join-Path $OutputDir "playwright-browsers"

    if (Test-CommandExists "docker") {
        Write-SubStep "Using Docker to download Linux Chromium ..."
        $pwDockerCmd = "set -e && npm install -g playwright@latest 2>&1 && PLAYWRIGHT_BROWSERS_PATH=/browsers npx playwright install chromium 2>&1"
        try {
            & docker run --rm -v "${pwDir}:/browsers" --name hermes-pw-downloader node:22-slim bash -c $pwDockerCmd 2>&1 | ForEach-Object { Write-SubStep "  $_" }

            $chromiumDir = Get-ChildItem $pwDir -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($chromiumDir) {
                Write-Ok "Playwright Chromium downloaded: $($chromiumDir.Name)"
            } else {
                Write-Warn "Docker download may have failed. Check playwright-browsers/ directory."
            }
        } catch {
            Write-Warn "Docker Playwright download failed: $_"
            Write-SubStep "Install manually on Linux target: npx playwright install chromium"
        }
    } else {
        Write-Warn "Docker not found. Cannot download Linux Chromium on Windows."
        Write-SubStep "Playwright Chromium will need to be installed on the Linux target:"
        Write-SubStep "  npx playwright install chromium"
        Write-SubStep "  (requires temporary internet access or pre-downloaded binary)"
    }
} else {
    Write-Step "4/6 Skipping Playwright Chromium (-SkipBrowser)"
}

# ============================================================
# 5/6 Create npm offline cache
# ============================================================

if (-not $SkipNpm) {
    Write-Step "5/6 Creating npm offline cache"

    $npmOfflineDir = Join-Path $OutputDir "npm-offline"

    # Install all deps online first (ensure node_modules is complete)
    Write-SubStep "Installing root workspace dependencies ..."
    Push-Location $projectRoot
    try {
        & npm install --silent 2>&1 | ForEach-Object { Write-SubStep "  $_" }
    } catch {
        Write-Warn "npm install failed: $_"
    }
    Pop-Location

    # Collect all non-workspace dependencies
    Write-SubStep "Packing npm dependencies to offline cache ..."

    $allNpmDeps = @{}

    # Read dependencies from each workspace package.json
    $pkgFiles = @(
        (Join-Path $projectRoot "package.json"),
        (Join-Path (Join-Path $projectRoot "web") "package.json"),
        (Join-Path (Join-Path $projectRoot "ui-tui") "package.json"),
        (Join-Path (Join-Path (Join-Path (Join-Path $projectRoot "ui-tui") "packages") "hermes-ink") "package.json")
    )

    foreach ($pkgFile in $pkgFiles) {
        if (Test-Path $pkgFile) {
            $pkg = Get-Content $pkgFile -Raw | ConvertFrom-Json
            if ($pkg.dependencies) {
                foreach ($dep in $pkg.dependencies.PSObject.Properties) {
                    if (-not ($dep.Value -match "^file:")) {
                        $allNpmDeps[$dep.Name] = $dep.Value
                    }
                }
            }
        }
    }

    Write-SubStep "Found $($allNpmDeps.Count) npm dependencies"

    # Pack each dependency from node_modules
    $packed = 0
    foreach ($depName in $allNpmDeps.Keys) {
        # Handle scoped packages (@scope/name)
        $depPath = Join-Path (Join-Path $projectRoot "node_modules") $depName
        if (Test-Path $depPath) {
            Push-Location $depPath
            try {
                $tarballName = & npm pack --silent 2>$null
                if ($tarballName) {
                    $tarballName = $tarballName.Trim()
                    $src = Join-Path $depPath $tarballName
                    $dst = Join-Path $npmOfflineDir $tarballName
                    if (Test-Path $src) {
                        Move-Item $src $dst -Force -ErrorAction SilentlyContinue
                        $packed++
                    }
                }
            } catch {
                # Skip failed packages
            } finally {
                Pop-Location
            }
        }
    }

    $npmSizeMB = [math]::Round((Get-ChildItem $npmOfflineDir -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Ok "npm offline packages: $packed, $npmSizeMB MB"
} else {
    Write-Step "5/6 Skipping npm offline cache (-SkipNpm)"
}

# ============================================================
# 6/6 Download Playwright system dependencies (deb packages)
# ============================================================

if (-not $SkipDebs) {
    Write-Step "6/6 Downloading Playwright system dependencies (deb packages)"

    $debDir = Join-Path $OutputDir "deb-packages"

    # Playwright Chromium requires these Debian/Ubuntu system libraries
    # Derived from Arch pacman list in scripts/install.sh
    $debPackages = @(
        "libnss3",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libdrm2",
        "libxkbcommon0",
        "libgbm1",
        "libpango-1.0-0",
        "libcairo2",
        "libasound2",
        "libatspi2.0-0",
        "libxcomposite1",
        "libxdamage1",
        "libxfixes3",
        "libxrandr2",
        "libwayland-client0",
        "libwayland-cursor0",
        "libwayland-egl1",
        "libnspr4",
        "libfontconfig1",
        "libfreetype6",
        "libxshmfence1"
    )

    Write-SubStep "Need to download $($debPackages.Count) deb packages"

    # Try Docker first
    if (Test-CommandExists "docker") {
        Write-SubStep "Docker detected, downloading deb packages automatically ..."

        # Build the docker command as a single string
        $pkgList = $debPackages -join " "
        $dockerCmd = "set -e && apt-get update -qq && cd /debs && apt-get download -y $pkgList 2>&1 || true"

        $containerName = "hermes-deb-downloader-$(Get-Random)"
        try {
            & docker run --rm -v "${debDir}:/debs" --name $containerName debian:bookworm-slim bash -c $dockerCmd 2>&1 | ForEach-Object { Write-SubStep "  $_" }

            $debCount = (Get-ChildItem $debDir -Filter "*.deb" -ErrorAction SilentlyContinue).Count
            if ($debCount -gt 0) {
                $debSizeMB = [math]::Round((Get-ChildItem $debDir -Filter "*.deb" | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
                Write-Ok "deb packages: $debCount, $debSizeMB MB"
            } else {
                Write-Warn "Docker download got no deb packages. Download manually in WSL."
            }
        } catch {
            Write-Warn "Docker download failed: $_"
            Write-SubStep "Download manually in WSL (see OFFLINE-README.md)"
        }
    } else {
        Write-Warn "Docker not found. Download deb packages manually in WSL."
        Write-SubStep "See OFFLINE-README.md for manual download instructions."
        Write-SubStep "Package list: $($debPackages -join ' ')"
    }
} else {
    Write-Step "6/6 Skipping deb packages (-SkipDebs)"
}

# ============================================================
# Copy install script
# ============================================================

Write-Step "Copying install script"

$installSh = Join-Path (Join-Path $projectRoot "scripts") "install-offline.sh"
if (Test-Path $installSh) {
    Copy-Item $installSh (Join-Path $OutputDir "install-offline.sh") -Force
    Write-Ok "install-offline.sh copied"
} else {
    Write-Warn "install-offline.sh not found"
}

# ============================================================
# Generate manifest
# ============================================================

Write-Step "Generating manifest"

$manifestLines = @()
$manifestLines += "# Hermes Agent Offline Bundle Manifest"
$manifestLines += "# Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$manifestLines += "# Target: Linux x86_64 / Python $PythonVersion"
$manifestLines += ""
$manifestLines += "## Binaries"
Get-ChildItem (Join-Path $OutputDir "binaries") -ErrorAction SilentlyContinue | ForEach-Object {
    $manifestLines += "- $($_.Name) ($([math]::Round($_.Length / 1MB, 1)) MB)"
}
$manifestLines += ""
$manifestLines += "## Python Wheels"
$manifestLines += "- Count: $(Get-ChildItem (Join-Path $OutputDir 'python-wheels') -Filter '*.whl' -ErrorAction SilentlyContinue).Count"
$manifestLines += "- Size: $([math]::Round((Get-ChildItem (Join-Path $OutputDir 'python-wheels') -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)) MB"
$manifestLines += ""
$manifestLines += "## NPM Offline"
$manifestLines += "- Count: $(Get-ChildItem (Join-Path $OutputDir 'npm-offline') -Filter '*.tgz' -ErrorAction SilentlyContinue).Count"
$manifestLines += ""
$manifestLines += "## Deb Packages"
Get-ChildItem (Join-Path $OutputDir "deb-packages") -Filter "*.deb" -ErrorAction SilentlyContinue | ForEach-Object {
    $manifestLines += "- $($_.Name)"
}
$manifestLines += ""
$manifestLines += "## Source"
$manifestLines += "- hermes-agent.tar.gz: $(Get-FileSizeMB (Join-Path $OutputDir 'hermes-agent.tar.gz')) MB"

$manifestPath = Join-Path $OutputDir "MANIFEST.txt"
Set-Content -Path $manifestPath -Value ($manifestLines -join "`n") -Encoding UTF8
Write-Ok "Manifest written to MANIFEST.txt"

# ============================================================
# Done
# ============================================================

$totalSizeMB = [math]::Round((Get-ChildItem $OutputDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB, 1)

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Offline bundle ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Output:  $OutputDir" -ForegroundColor Green
Write-Host "  Size:    $totalSizeMB MB" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Green
Write-Host "  1. Transfer hermes-offline-bundle/ to Linux target" -ForegroundColor Green
Write-Host "  2. On target: chmod +x install-offline.sh" -ForegroundColor Green
Write-Host "  3. On target: sudo ./install-offline.sh" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
