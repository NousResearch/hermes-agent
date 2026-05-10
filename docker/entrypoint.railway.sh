#!/bin/bash
# Railway entrypoint for Hermes Agent - no privilege dropping

set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

echo "[hermes-railway] Starting Hermes Agent on Railway"
echo "[hermes-railway] HERMES_HOME=$HERMES_HOME"

# --- Create directory structure ---
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# --- If HERMES_HOME is empty (first boot), try to bootstrap from git ---
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    echo "[hermes-railway] First boot — checking for config..."

    if [ -n "$HARNESS_GIT_URL" ]; then
        echo "[hermes-railway] Cloning config from git..."
        git clone -b "${HARNESS_BRANCH:-main}" "$HARNESS_GIT_URL" "$HERMES_HOME" 2>&1 | tail -3 || true
    fi

    if [ ! -f "$HERMES_HOME/config.yaml" ]; then
        echo "[hermes-railway] Using default config..."
        cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
        cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
        cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md" 2>/dev/null || true
    fi
fi

# --- Bootstrap minimal files if still missing ---
[ ! -f "$HERMES_HOME/.env" ] && cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
[ ! -f "$HERMES_HOME/config.yaml" ] && cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
[ ! -f "$HERMES_HOME/SOUL.md" ] && cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md" 2>/dev/null || true

# --- Fix ownership (Railway mounts as root) ---
if [ -d "$HERMES_HOME" ]; then
    chown -R 10000:10000 "$HERMES_HOME" 2>/dev/null || true
fi

# --- Sync bundled skills ---
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py" 2>/dev/null || true
fi

echo "[hermes-railway] Config ready. Starting hermes gateway..."

# --- Start gateway ---
source "${INSTALL_DIR}/.venv/bin/activate"
exec hermes gateway run
