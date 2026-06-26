#!/usr/bin/env bash
# setup_local_web_stack.sh — one-command local web stack for Hermes
#
# Sets up and starts a privacy-preserving, self-hosted web stack:
#   SearXNG  -> web_search
#   Firecrawl -> web_extract / crawl (Docker)
#   Camofox   -> browser extraction / automation
#
# Usage:
#   ./scripts/setup_local_web_stack.sh [command]
#
# Commands:
#   setup     Install dependencies and start all services (default)
#   start     Start all services
#   stop      Stop all services
#   status    Print service status
#   env       Print export commands for Hermes env vars

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEARXNG_USER="searxng"
SEARXNG_HOME="/usr/local/searxng"
SEARXNG_VENV="${SEARXNG_HOME}/searx-pyenv"
SEARXNG_SRC="${SEARXNG_HOME}/searxng-src"
SEARXNG_PORT="${SEARXNG_PORT:-8080}"

CAMOFOX_DIR="${CAMOFOX_DIR:-/opt/camofox-browser}"
CAMOFOX_PORT="${CAMOFOX_PORT:-9377}"

FIRECRAWL_DIR="${FIRECRAWL_DIR:-/opt/firecrawl}"
FIRECRAWL_PORT="${FIRECRAWL_PORT:-3002}"

HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
HERMES_ENV="${HERMES_HOME}/.env"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[local-web-stack] $*"; }
warn() { echo "[local-web-stack] WARNING: $*" >&2; }

cmd_exists() { command -v "$1" >/dev/null 2>&1; }

ensure_docker() {
    if cmd_exists docker; then
        return 0
    fi
    log "Docker not found; installing..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "${USER}"
    log "Docker installed. You may need to log out and back in for group changes to take effect."
}

ensure_searxng() {
    if [ -d "${SEARXNG_SRC}" ]; then
        log "SearXNG already installed at ${SEARXNG_SRC}"
        return 0
    fi
    log "Installing SearXNG..."
    sudo apt-get update
    sudo apt-get install -y python3-dev python3-babel python3-venv python-is-python3 \
        git build-essential libxslt-dev zlib1g-dev libffi-dev libssl-dev \
        curl redis-server

    sudo useradd --shell /bin/bash --system --home-dir "${SEARXNG_HOME}" \
        --comment 'Privacy-respecting metasearch engine' "${SEARXNG_USER}" 2>/dev/null || true
    sudo mkdir -p "${SEARXNG_HOME}"
    sudo chown -R "${SEARXNG_USER}:${SEARXNG_USER}" "${SEARXNG_HOME}"

    sudo -H -u "${SEARXNG_USER}" git clone https://github.com/searxng/searxng "${SEARXNG_SRC}"
    sudo -H -u "${SEARXNG_USER}" python3 -m venv "${SEARXNG_VENV}"
    sudo -H -u "${SEARXNG_USER}" "${SEARXNG_VENV}/bin/pip" install -U pip setuptools wheel pyyaml msgspec typing-extensions pybind11
    sudo -H -u "${SEARXNG_USER}" "${SEARXNG_VENV}/bin/pip" install --use-pep517 --no-build-isolation -e "${SEARXNG_SRC}"

    sudo mkdir -p /etc/searxng
    cat <<EOF | sudo tee /etc/searxng/settings.yml >/dev/null
server:
  port: ${SEARXNG_PORT}
  bind_address: "0.0.0.0"
  secret_key: "$(openssl rand -hex 32)"
search:
  formats:
    - html
    - json
EOF
    sudo chown -R "${SEARXNG_USER}:${SEARXNG_USER}" /etc/searxng
    log "SearXNG installed."
}

start_searxng() {
    if ss -tlnp 2>/dev/null | grep -q ":${SEARXNG_PORT}"; then
        log "SearXNG already running on port ${SEARXNG_PORT}"
        return 0
    fi
    log "Starting SearXNG on port ${SEARXNG_PORT}..."
    sudo -H -u "${SEARXNG_USER}" -s /bin/sh -c "cd ${SEARXNG_HOME} && SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml nohup ${SEARXNG_VENV}/bin/python -m searx.webapp > /tmp/searxng.log 2>&1 &"
    for i in {1..30}; do
        if ss -tlnp 2>/dev/null | grep -q ":${SEARXNG_PORT}"; then
            log "SearXNG is listening."
            return 0
        fi
        sleep 1
    done
    warn "SearXNG did not start within 30s; check /tmp/searxng.log"
    return 1
}

ensure_camofox() {
    if [ -d "${CAMOFOX_DIR}" ]; then
        log "Camofox already installed at ${CAMOFOX_DIR}"
        return 0
    fi
    log "Installing Camofox to ${CAMOFOX_DIR}..."
    sudo mkdir -p "$(dirname "${CAMOFOX_DIR}")"
    sudo git clone https://github.com/jo-inc/camofox-browser.git "${CAMOFOX_DIR}"
    sudo chown -R "${USER}:${USER}" "${CAMOFOX_DIR}"
    cd "${CAMOFOX_DIR}"
    npm install
    log "Camofox installed."
}

start_camofox() {
    if ss -tlnp 2>/dev/null | grep -q ":${CAMOFOX_PORT}"; then
        log "Camofox already running on port ${CAMOFOX_PORT}"
        return 0
    fi
    log "Starting Camofox on port ${CAMOFOX_PORT}..."
    cd "${CAMOFOX_DIR}"
    nohup env CAMOFOX_PORT="${CAMOFOX_PORT}" node --max-old-space-size=128 server.js > /tmp/camofox.log 2>&1 &
    for i in {1..30}; do
        if ss -tlnp 2>/dev/null | grep -q ":${CAMOFOX_PORT}"; then
            log "Camofox is listening."
            return 0
        fi
        sleep 1
    done
    warn "Camofox did not start within 30s; check /tmp/camofox.log"
    return 1
}

ensure_firecrawl() {
    if [ -d "${FIRECRAWL_DIR}" ]; then
        log "Firecrawl already installed at ${FIRECRAWL_DIR}"
        return 0
    fi
    log "Installing Firecrawl to ${FIRECRAWL_DIR}..."
    sudo mkdir -p "$(dirname "${FIRECRAWL_DIR}")"
    sudo git clone https://github.com/firecrawl/firecrawl.git "${FIRECRAWL_DIR}"
    sudo chown -R "${USER}:${USER}" "${FIRECRAWL_DIR}"

    cat > "${FIRECRAWL_DIR}/.env" <<'EOF'
PORT=3002
HOST=0.0.0.0
NUM_WORKERS_PER_QUEUE=1
CRAWL_CONCURRENT_REQUESTS=2
MAX_CONCURRENT_JOBS=2
BROWSER_POOL_SIZE=2
USE_DB_AUTHENTICATION=false
X402_ENABLED=false
REDIS_URL=redis://redis:6379
REDIS_RATE_LIMIT_URL=redis://redis:6379
EOF

    # Prefer pre-built images so the user does not need Go/Rust/Node native builds.
    sed -i 's|# image: ghcr.io/firecrawl/firecrawl|image: ghcr.io/firecrawl/firecrawl:latest|; s|build: apps/api|# build: apps/api|' "${FIRECRAWL_DIR}/docker-compose.yaml"
    sed -i 's|# image: ghcr.io/firecrawl/playwright-service:latest|image: ghcr.io/firecrawl/playwright-service:latest|; s|build: apps/playwright-service-ts|# build: apps/playwright-service-ts|' "${FIRECRAWL_DIR}/docker-compose.yaml"
    # Tighten limits for small VPSes.
    sed -i 's|cpus: 4.0|cpus: 1.5|; s|mem_limit: 8G|mem_limit: 3G|; s|memswap_limit: 8G|memswap_limit: 3G|' "${FIRECRAWL_DIR}/docker-compose.yaml"
    log "Firecrawl installed."
}

start_firecrawl() {
    if ss -tlnp 2>/dev/null | grep -q ":${FIRECRAWL_PORT}"; then
        log "Firecrawl already running on port ${FIRECRAWL_PORT}"
        return 0
    fi
    log "Starting Firecrawl (Docker) on port ${FIRECRAWL_PORT}..."
    cd "${FIRECRAWL_DIR}"
    docker compose pull
    docker compose up -d
    for i in {1..60}; do
        if curl -s --max-time 2 "http://127.0.0.1:${FIRECRAWL_PORT}/" >/dev/null 2>&1; then
            log "Firecrawl API is responding."
            return 0
        fi
        sleep 2
    done
    warn "Firecrawl API did not respond within 2m; check 'docker compose logs -f api'"
    return 1
}

stop_firecrawl() {
    if [ -d "${FIRECRAWL_DIR}" ]; then
        log "Stopping Firecrawl..."
        (cd "${FIRECRAWL_DIR}" && docker compose down) || true
    fi
}

write_hermes_env() {
    mkdir -p "${HERMES_HOME}"
    touch "${HERMES_ENV}"
    # Idempotent update: remove old local-web-stack entries, then append.
    grep -vE '^(SEARXNG_URL|CAMOFOX_URL|FIRECRAWL_API_URL|FIRECRAWL_API_KEY)=' "${HERMES_ENV}" > "${HERMES_ENV}.tmp" || true
    cat >> "${HERMES_ENV}.tmp" <<EOF
SEARXNG_URL=http://127.0.0.1:${SEARXNG_PORT}
CAMOFOX_URL=http://127.0.0.1:${CAMOFOX_PORT}
FIRECRAWL_API_URL=http://127.0.0.1:${FIRECRAWL_PORT}
FIRECRAWL_API_KEY=***
EOF
    mv "${HERMES_ENV}.tmp" "${HERMES_ENV}"
    chmod 600 "${HERMES_ENV}"
    log "Wrote local web-stack env vars to ${HERMES_ENV}"
}

configure_hermes() {
    log "Configuring Hermes to use local web stack..."
    hermes config set web.search_backend searxng 2>/dev/null || true
    hermes config set web.extract_backend firecrawl 2>/dev/null || true
    hermes config set web.backend searxng 2>/dev/null || true
}

print_env() {
    cat <<EOF
export SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}"
export CAMOFOX_URL="http://127.0.0.1:${CAMOFOX_PORT}"
export FIRECRAWL_API_URL="http://127.0.0.1:${FIRECRAWL_PORT}"
export FIRECRAWL_API_KEY="***"
EOF
}

print_status() {
    echo "SearXNG  (port ${SEARXNG_PORT}): $(ss -tlnp 2>/dev/null | grep -q ":${SEARXNG_PORT}" && echo 'up' || echo 'down')"
    echo "Camofox  (port ${CAMOFOX_PORT}): $(ss -tlnp 2>/dev/null | grep -q ":${CAMOFOX_PORT}" && echo 'up' || echo 'down')"
    echo "Firecrawl (port ${FIRECRAWL_PORT}): $(curl -s --max-time 2 "http://127.0.0.1:${FIRECRAWL_PORT}/" >/dev/null 2>&1 && echo 'up' || echo 'down')"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMAND="${1:-setup}"

case "${COMMAND}" in
setup)
    ensure_docker
    ensure_searxng
    ensure_camofox
    ensure_firecrawl
    start_searxng
    start_camofox
    start_firecrawl
    write_hermes_env
    configure_hermes
    log "Local web stack is ready."
    print_status
    log "Run 'source ${HERMES_ENV}' or restart Hermes to load env vars."
    ;;
start)
    start_searxng
    start_camofox
    start_firecrawl
    ;;
stop)
    stop_firecrawl
    warn "SearXNG and Camofox were started as background processes; stop them manually if needed."
    ;;
status)
    print_status
    ;;
env)
    print_env
    ;;
*)
    echo "Usage: $0 {setup|start|stop|status|env}"
    exit 1
    ;;
esac
