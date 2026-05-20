#!/usr/bin/env bash
# docker/entrypoint.saas.sh — SaaS-mode container entrypoint.
#
# Differences from docker/entrypoint.sh (community image):
#   - No EFS volume, no hermes-state mount. HERMES_HOME is /tmp/hermes-runtime.
#   - No UID remapping / gosu: container runs as hermes (uid 10001) from the start.
#   - Starts the health HTTP server (gateway/health_server.py) in the background
#     before exec-ing the main gateway process.
#   - Creates ephemeral HERMES_HOME subdirectory structure in /tmp.
#   - Bootstraps .env and config.yaml from defaults — no persistent state.
#
# Environment variables consumed:
#   HERMES_HOME          — defaults to /tmp/hermes-runtime
#   HERMES_GATEWAY_PLATFORM — gateway platform (default: slack)
#   HEALTH_PORT          — port for health HTTP server (default: 8080)
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-/tmp/hermes-runtime}"
HEALTH_PORT="${HEALTH_PORT:-8080}"
INSTALL_DIR="/app"

# Create ephemeral directory structure.
# These paths are written by the Hermes runtime during a session — they are
# recreated fresh on every container start. No persistent state expected here.
mkdir -p "${HERMES_HOME}"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# Seed .env from example if absent (ephemeral — reseeded every start).
if [ ! -f "${HERMES_HOME}/.env" ] && [ -f "${INSTALL_DIR}/.env.example" ]; then
    cp "${INSTALL_DIR}/.env.example" "${HERMES_HOME}/.env"
fi

# Seed config.yaml from example if absent.
if [ ! -f "${HERMES_HOME}/config.yaml" ] && [ -f "${INSTALL_DIR}/cli-config.yaml.example" ]; then
    cp "${INSTALL_DIR}/cli-config.yaml.example" "${HERMES_HOME}/config.yaml"
fi

# Start the health probe HTTP server in the background.
# gateway/health_server.py listens on HEALTH_PORT and responds to GET /health.
# This must start before the ECS container health check fires (startPeriod=60s).
echo "[saas-entrypoint] Starting health server on :${HEALTH_PORT}"
python -m gateway.health_server --port "${HEALTH_PORT}" &
HEALTH_PID=$!

# Give the health server a moment to bind the port before proceeding.
# This avoids a race where ECS health checks fire before the server is up.
sleep 2

# Verify the health server bound successfully.
if ! kill -0 "${HEALTH_PID}" 2>/dev/null; then
    echo "[saas-entrypoint] ERROR: health server failed to start. Aborting."
    exit 1
fi
echo "[saas-entrypoint] Health server running (pid ${HEALTH_PID})"

# Exec the gateway. This replaces the current shell so signals (SIGTERM from
# ECS task stop) are delivered directly to the hermes process.
echo "[saas-entrypoint] Starting: hermes $*"
exec hermes "$@"
