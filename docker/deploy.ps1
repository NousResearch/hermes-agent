<#
.SYNOPSIS
    docker/deploy.ps1 — One-liner installer for the JZKK720/hermes-agent fork (Windows)

.DESCRIPTION
    Windows PowerShell mirror of docker/deploy.sh. Installs the Hermes stack
    on a fresh Windows machine using the fork's published GHCR image.

.USAGE
    Fresh machine (piped from GitHub):
        irm https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy.ps1 | iex

    Or clone first, then run:
        powershell -ExecutionPolicy Bypass -File docker/deploy.ps1

.PREREQUISITES
    - Docker Desktop (with Compose v2): https://docs.docker.com/desktop/setup/windows-install/
    - Ollama running on host port 11434 with a model pulled:
        ollama pull nemotron-3.5-lightning:30b-a3b

.WHAT IT DOES
    1. Clones JZKK720/hermes-agent (skipped if already cloned)
    2. Creates the data/ directory and seeds data/.env from template
    3. Pins the weixin base_url in data/config.yaml (if present)
    4. Recreates the stack from the fork's GHCR-published image with:
       docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
#>

#Requires -Version 5.1

$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/JZKK720/hermes-agent.git'
$RepoDir = 'hermes-agent'
$ComposeFile = 'docker-compose.upstream.yml'
$EnvTemplate = 'docker/hermes-env.example'

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Log  { param([string]$Msg) Write-Host       "[hermes] $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host       "[hermes] $Msg" -ForegroundColor Green }
function Write-Warn2 { param([string]$Msg) Write-Host      "[hermes] $Msg" -ForegroundColor Yellow }
function Write-Die  { param([string]$Msg) Write-Host "[hermes] ERROR: $Msg" -ForegroundColor Red; throw $Msg }

# Treat non-zero exit codes from native commands as errors (PS7+ does this
# automatically when $PSNativeCommandUseErrorActionPreference is true; for
# PS5.1 we check $LASTEXITCODE explicitly after each native call).
function Invoke-Native {
    param([scriptblock]$Block)
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Die "Command exited with code $LASTEXITCODE"
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
    Write-Ok "Ollama is running on :11434"
} else {
    Write-Warn2 "Ollama not detected on :11434 — containers will start but model calls will fail."
    Write-Warn2 "Start Ollama and run: ollama pull nemotron-3.5-lightning:30b-a3b"
}

# ── Clone (if not already inside the repo) ────────────────────────────────────
if (-not (Test-Path 'docker-compose.yml')) {
    Write-Log "Cloning $RepoUrl ..."
    Invoke-Native { git clone $RepoUrl $RepoDir }
    Set-Location $RepoDir
} else {
    Write-Log "Already inside hermes-agent repo — skipping clone."
}

# ── Seed data directory ───────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path 'data' | Out-Null

if (-not (Test-Path 'data/.env')) {
    if (Test-Path $EnvTemplate) {
        Copy-Item $EnvTemplate 'data/.env'
        Write-Ok "Created data/.env from template"
        Write-Host ''
        Write-Warn2 'Review data/.env and set any API keys before starting.'
        Write-Warn2 '  - POSTGRES_PASSWORD: must match what you set in compose if you change it'
        Write-Warn2 '  - Messaging tokens: TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, etc.'
        Write-Warn2 '  - WeChat (weixin) does not need a token — use the QR wizard below.'
        Write-Host ''
    } else {
        Write-Warn2 "Template $EnvTemplate not found — skipping data/.env seed."
    }
} else {
    Write-Log 'data/.env already exists — skipping template copy.'
}

# ── Pin weixin base_url in data/config.yaml ───────────────────────────────────
# A stale WEIXIN_BASE_URL in data/.env (e.g. https://ilinkai.wechat.com instead
# of https://ilinkai.weixin.qq.com) causes silent "Session expired" errors.
# The adapter resolution order is: extra.base_url -> WEIXIN_BASE_URL env -> constant.
# Pinning it in config.yaml wins over any stale env var.
if (Test-Path 'data/config.yaml') {
    $configContent = Get-Content 'data/config.yaml' -Raw -ErrorAction SilentlyContinue
    if ($configContent -and ($configContent -notmatch 'ilinkai\.weixin\.qq\.com')) {
        Write-Log 'Pinning weixin base_url in data/config.yaml ...'
        $pyCode = @"
import yaml
with open('/opt/data/config.yaml') as f:
    cfg = yaml.safe_load(f)
wx = cfg.setdefault('platforms', {}).setdefault('weixin', {})
wx.setdefault('extra', {})['base_url'] = 'https://ilinkai.weixin.qq.com'
with open('/opt/data/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('weixin base_url pinned')
"@
        Invoke-Native { docker compose -f $ComposeFile run --rm --no-deps --entrypoint '' hermes-gateway python3 -c $pyCode }
    } else {
        Write-Log 'weixin base_url already pinned in data/config.yaml — skipping.'
    }
}

# ── Pull + recreate ───────────────────────────────────────────────────────────
Write-Log "Refreshing the fork's GHCR-published Hermes image and recreating services..."
Invoke-Native { docker compose -f $ComposeFile up -d --pull always --force-recreate --remove-orphans }

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ''
Write-Ok  'Hermes-Agent is up!'
Write-Host ''
Write-Host '  Web UI              : http://localhost:9119' -ForegroundColor Cyan
Write-Host '  WeChat gateway      : hermes-gateway (outbound only, no host port)' -ForegroundColor Cyan
Write-Host '  PostgreSQL          : localhost:5433' -ForegroundColor Cyan
Write-Host ''
Write-Host '  Interactive CLI     : docker exec -it hermes-web hermes' -ForegroundColor Cyan
Write-Host '  Connect WeChat      : open http://localhost:9119 -> Channels -> Set up with QR' -ForegroundColor Cyan
Write-Host '  View logs           : docker compose -f docker-compose.upstream.yml logs -f' -ForegroundColor Cyan
Write-Host '  Stop all            : docker compose -f docker-compose.upstream.yml down' -ForegroundColor Cyan
Write-Host ''
Write-Warn2 'Config lives in data/config.yaml — edit the model name or settings there.'
Write-Warn2 'Default model: nemotron-3.5-lightning:30b-a3b — change it in data/config.yaml'
Write-Warn2 'Vision model: qwen3.8:27b (for image analysis) — pull it with: ollama pull qwen3.8:27b'
Write-Warn2 'Ollama is reached at http://host.docker.internal:11434 — make sure it is running on the host.'