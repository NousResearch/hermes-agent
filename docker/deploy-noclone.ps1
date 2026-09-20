<#
.SYNOPSIS
    docker/deploy-noclone.ps1 — No-clone installer for the JZKK720/hermes-agent fork (Windows)

.DESCRIPTION
    Installs the Hermes stack on a fresh Windows machine WITHOUT cloning the
    full repository. Downloads only the 2 files the compose file bind-mounts,
    then pulls and recreates the stack from the fork's published GHCR image.

.USAGE
    Fresh machine (one command, no repo needed):
        irm https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy-noclone.ps1 | iex

    Or save and run locally:
        powershell -ExecutionPolicy Bypass -File docker/deploy-noclone.ps1

.PREREQUISITES
    - Docker Desktop (with Compose v2): https://docs.docker.com/desktop/setup/windows-install/
    - Ollama running on host port 11434 with a model pulled:
        ollama pull nemotron-3.5-lightning:30b-a3b

.WHAT IT DOES
    1. Creates a working directory (default: .\hermes-agent, override with -InstallDir)
    2. Downloads docker-compose.upstream.yml and docker/hermes-config.yaml from GitHub
    3. Creates data/.env with a minimal working template
    4. Pulls and recreates the stack from the fork's GHCR-published image:
       docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans

.PARAMETER InstallDir
    Directory to install into. Defaults to ".\hermes-agent" in the current location.

.PARAMETER Force
    Overwrite an existing install directory if it already has a compose file.

.EXAMPLE
    # Default install
    irm https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy-noclone.ps1 | iex

.EXAMPLE
    # Custom install directory
    irm https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy-noclone.ps1 | iex -InstallDir C:\hermes

.EXAMPLE
    # Run from a saved copy
    powershell -ExecutionPolicy Bypass -File docker/deploy-noclone.ps1 -InstallDir C:\hermes
#>

#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = '.\hermes-agent',
    [string]$RepoRawUrl = 'https://raw.githubusercontent.com/JZKK720/hermes-agent/main',
    [string]$ComposeFile = 'docker-compose.upstream.yml',
    [string]$ConfigFile = 'docker/hermes-config.yaml',
    [string]$EnvTemplateUrl = 'https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/hermes-env.example',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Log   { param([string]$Msg) Write-Host       "[hermes] $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host       "[hermes] $Msg" -ForegroundColor Green }
function Write-Warn2 { param([string]$Msg) Write-Host       "[hermes] $Msg" -ForegroundColor Yellow }
function Write-Die   { param([string]$Msg) Write-Host "[hermes] ERROR: $Msg" -ForegroundColor Red; throw $Msg }

function Invoke-Native {
    param([scriptblock]$Block, [string]$Description = 'command')
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Die "$Description exited with code $LASTEXITCODE"
    }
}

function Save-FileFromUrl {
    param([string]$Url, [string]$OutPath)
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing -TimeoutSec 30 -ErrorAction Stop
    } catch {
        Write-Die "Failed to download $Url : $($_.Exception.Message)"
    }
}

# ── Preflight checks ──────────────────────────────────────────────────────────
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Die "docker not found. Install Docker Desktop: https://docs.docker.com/desktop/setup/windows-install/"
}
try {
    $composeVersion = docker compose version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Die "docker compose (v2) not found. Upgrade Docker Desktop or install the Compose plugin."
    }
} catch {
    Write-Die "docker compose (v2) not found. Upgrade Docker Desktop or install the Compose plugin."
}

# ── Confirm Ollama is reachable ───────────────────────────────────────────────
$ollamaOk = $false
try {
    $null = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    $ollamaOk = $true
} catch {
    $ollamaOk = $false
}
if ($ollamaOk) {
    Write-Ok 'Ollama is running on :11434'
} else {
    Write-Warn2 'Ollama not detected on :11434 — containers will start but model calls will fail.'
    Write-Warn2 'Start Ollama and run: ollama pull nemotron-3.5-lightning:30b-a3b'
}

# ── Create install directory ──────────────────────────────────────────────────
$InstallDirFull = (Resolve-Path -Path '.' -ErrorAction SilentlyContinue).Path
if (-not $InstallDirFull) { $InstallDirFull = (Get-Location).Path }
$InstallDirFull = Join-Path $InstallDirFull ($InstallDir -replace '^\.\\', '' -replace '^\./', '')

$composePath = Join-Path $InstallDirFull $ComposeFile
$dockerDir = Join-Path $InstallDirFull 'docker'
$configPath = Join-Path $InstallDirFull $ConfigFile
$dataDir = Join-Path $InstallDirFull 'data'
$envPath = Join-Path $dataDir '.env'

# Check if already installed
if ((Test-Path $composePath) -and -not $Force) {
    Write-Log "Found existing install at $InstallDirFull"
    Write-Log 'Refreshing the fork GHCR image and recreating services...'
    Push-Location $InstallDirFull
    try {
        Invoke-Native {
            docker compose -f $ComposeFile up -d --pull always --force-recreate --remove-orphans
        } -Description 'docker compose up'
    } finally {
        Pop-Location
    }
    Write-Host ''
    Write-Ok 'Hermes-Agent refreshed!'
    Write-Host ''
    Write-Host '  Web UI              : http://localhost:9119' -ForegroundColor Cyan
    Write-Host '  Stop all            : docker compose -f docker-compose.upstream.yml down' -ForegroundColor Cyan
    Write-Host '  View logs           : docker compose -f docker-compose.upstream.yml logs -f' -ForegroundColor Cyan
    return
}

if ($Force -and (Test-Path $InstallDirFull)) {
    Write-Log "Force mode — removing existing $InstallDirFull (data/ will be preserved if it exists)"
    $existingData = $null
    if (Test-Path $dataDir) {
        $existingData = $dataDir
    }
    if ($existingData) {
        $tempData = Join-Path $InstallDirFull '..\_hermes_data_backup'
        if (Test-Path $tempData) { Remove-Item $tempData -Recurse -Force }
        Move-Item $existingData $tempData -Force
    }
    Remove-Item $InstallDirFull -Recurse -Force
    New-Item -ItemType Directory -Force -Path $InstallDirFull | Out-Null
    if ($existingData) {
        Move-Item $tempData $dataDir -Force
    }
} else {
    New-Item -ItemType Directory -Force -Path $InstallDirFull | Out-Null
}

Write-Log "Installing to: $InstallDirFull"

# ── Download required files ───────────────────────────────────────────────────
Write-Log "Downloading $ComposeFile ..."
Save-FileFromUrl -Url "$RepoRawUrl/$ComposeFile" -OutPath $composePath

Write-Log "Downloading $ConfigFile ..."
New-Item -ItemType Directory -Force -Path $dockerDir | Out-Null
Save-FileFromUrl -Url "$RepoRawUrl/$ConfigFile" -OutPath $configPath

# ── Seed data directory ───────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

if (-not (Test-Path $envPath)) {
    Write-Log 'Downloading .env template ...'
    try {
        Save-FileFromUrl -Url $EnvTemplateUrl -OutPath $envPath
        Write-Ok 'Created data/.env from template'
    } catch {
        # Fallback: write a minimal .env if the template download fails
        Write-Warn2 'Could not download .env template — writing minimal defaults.'
        $minimalEnv = @(
            '# Minimal .env — see https://github.com/JZKK720/hermes-agent for full template',
            'WEIXIN_ALLOW_ALL_USERS=1'
        ) -join "`n"
        Set-Content -Path $envPath -Value $minimalEnv -Encoding UTF8
        Write-Ok 'Created data/.env with minimal defaults'
    }
    Write-Host ''
    Write-Warn2 'Review data/.env and set any API keys before starting.'
    Write-Warn2 '  - POSTGRES_PASSWORD: must match what you set in compose if you change it'
    Write-Warn2 '  - Messaging tokens: TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, etc.'
    Write-Warn2 '  - WeChat (weixin) does not need a token — use the QR wizard in the dashboard.'
    Write-Host ''
} else {
    Write-Log 'data/.env already exists — skipping template copy.'
}

# ── Pull + recreate ───────────────────────────────────────────────────────────
Write-Log "Refreshing the fork's GHCR-published Hermes image and recreating services..."
Push-Location $InstallDirFull
try {
    Invoke-Native {
        docker compose -f $ComposeFile up -d --pull always --force-recreate --remove-orphans
    } -Description 'docker compose up'
} finally {
    Pop-Location
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Ok  'Hermes-Agent is up!'
Write-Host ''
Write-Host '  Web UI              : http://localhost:9119' -ForegroundColor Cyan
Write-Host '  WeChat gateway      : hermes-gateway (outbound only, no host port)' -ForegroundColor Cyan
Write-Host '  PostgreSQL          : localhost:5433' -ForegroundColor Cyan
Write-Host ''
Write-Host "  Install directory   : $InstallDirFull" -ForegroundColor Cyan
Write-Host '  Connect WeChat      : open http://localhost:9119 -> Channels -> Set up with QR' -ForegroundColor Cyan
Write-Host '  View logs           : docker compose -f docker-compose.upstream.yml logs -f' -ForegroundColor Cyan
Write-Host '  Stop all            : docker compose -f docker-compose.upstream.yml down' -ForegroundColor Cyan
Write-Host '  Routine update      : docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans' -ForegroundColor Cyan
Write-Host ''
Write-Warn2 'Config lives in data/config.yaml — edit the model name or settings there.'
Write-Warn2 'Default model: nemotron-3.5-lightning:30b-a3b — change it in data/config.yaml'
Write-Warn2 'Vision model: qwen3.8:27b (for image analysis) — pull it with: ollama pull qwen3.8:27b'
Write-Warn2 'Ollama is reached at http://host.docker.internal:11434 — make sure it is running on the host.'