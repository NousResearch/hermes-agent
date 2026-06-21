#!/usr/bin/env bash
# docker/deploy.sh — One-liner installer for the JZKK720/hermes-agent fork
#
# Usage (fresh machine):
#   curl -fsSL https://raw.githubusercontent.com/JZKK720/hermes-agent/main/docker/deploy.sh | bash
#
# Or clone first, then run:
#   bash docker/deploy.sh
#
# Prerequisites:
#   - Docker + Docker Compose v2 (https://docs.docker.com/get-docker/)
#   - Ollama running on host port 11434 with the target model pulled:
#       ollama pull gemma4:e4b-it-q8_0
#
# What this script does:
#   1. Clones JZKK720/hermes-agent (skipped if already cloned)
#   2. Creates the data/ directory and seeds data/.env from template
#   3. Recreates the stack from the published upstream Docker Hub image with:
#      docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans
# ============================================================================

set -euo pipefail

REPO_URL="https://github.com/JZKK720/hermes-agent.git"
REPO_DIR="hermes-agent"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[hermes]${NC} $*"; }
ok()   { echo -e "${GREEN}[hermes]${NC} $*"; }
warn() { echo -e "${YELLOW}[hermes]${NC} $*"; }
die()  { echo -e "${RED}[hermes] ERROR:${NC} $*" >&2; exit 1; }

# ── Preflight checks ──────────────────────────────────────────────────────────
command -v docker  >/dev/null 2>&1 || die "docker not found. Install from https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || die "docker compose (v2) not found. Upgrade Docker Desktop or install the plugin."

# ── Confirm Ollama is reachable ───────────────────────────────────────────────
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is running on :11434"
else
    warn "Ollama not detected on :11434 — containers will start but model calls will fail."
    warn "Start Ollama and run: ollama pull gemma4:e4b-it-q8_0"
fi

# ── Clone (if not already inside the repo) ───────────────────────────────────
if [ ! -f "docker-compose.yml" ]; then
    log "Cloning ${REPO_URL} ..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
else
    log "Already inside hermes-agent repo — skipping clone."
fi

# ── Seed data directory ───────────────────────────────────────────────────────
mkdir -p data

if [ ! -f "data/.env" ]; then
    cp docker/hermes-env.example data/.env
    ok "Created data/.env from template"
    echo ""
    warn "Review data/.env and set any API keys before starting."
    warn "  - POSTGRES_PASSWORD: must match what you set in compose if you change it"
    warn "  - Messaging tokens: TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, etc."
    warn "  - WeChat (weixin) does not need a token — use the QR wizard below."
    echo ""
else
    log "data/.env already exists — skipping template copy."
fi

# ── Pull + recreate ───────────────────────────────────────────────────────────
log "Refreshing the published upstream Docker Hub Hermes image and recreating services..."
docker compose -f docker-compose.upstream.yml up -d --pull always --force-recreate --remove-orphans

echo ""
ok "Hermes-Agent is up!"
echo ""
echo -e "  ${CYAN}Web UI${NC}              : http://localhost:9119"
echo -e "  ${CYAN}WeChat gateway${NC}      : hermes-gateway (outbound only, no host port)"
echo -e "  ${CYAN}PostgreSQL${NC}          : localhost:5433"
echo ""
echo -e "  ${CYAN}Interactive CLI${NC}     : docker exec -it hermes-web hermes"
echo -e "  ${CYAN}Connect WeChat${NC}      : docker compose -f docker-compose.upstream.yml run --rm hermes-gateway hermes gateway setup  (then pick item 13: Weixin / WeChat)"
echo -e "  ${CYAN}View logs${NC}           : docker compose -f docker-compose.upstream.yml logs -f"
echo -e "  ${CYAN}Stop all${NC}            : docker compose -f docker-compose.upstream.yml down"
echo ""
warn "Config lives in data/config.yaml — edit the model name or settings there."
warn "Default model: gemma4:e4b-it-q8_0 — change it in data/config.yaml"
warn "Ollama is reached at http://host.docker.internal:11434 — make sure it is running on the host."
