#!/bin/bash
# Railway-compatible entrypoint for Hermes Agent
# Runs BOTH gateway (background) + Web Dashboard (foreground on $PORT)

set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"
PORT="${PORT:-8080}"

# --- Privilege dropping via gosu ---
if [ "$(id -u)" = "0" ]; then
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
        usermod -u "$HERMES_UID" hermes
    fi
    if [ -n "$HERMES_GID" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
        groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
    fi

    actual_hermes_uid=$(id -u hermes)
    if [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
        chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || true
        chown -R hermes:hermes "$INSTALL_DIR/.venv" 2>/dev/null || true
    fi

    if [ -f "$HERMES_HOME/config.yaml" ]; then
        chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
        chmod 640 "$HERMES_HOME/config.yaml" 2>/dev/null || true
    fi

    echo "Dropping root privileges"
    exec gosu hermes "$0" "$@"
fi

# --- Running as hermes ---
source "${INSTALL_DIR}/.venv/bin/activate"

echo "docker" > "${HERMES_HOME}/.install_method" 2>/dev/null || true

mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# Bootstrap config files
[ ! -f "$HERMES_HOME/.env" ] && cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
[ ! -f "$HERMES_HOME/config.yaml" ] && cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
[ ! -f "$HERMES_HOME/SOUL.md" ] && cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"

if [ ! -f "$HERMES_HOME/auth.json" ] && [ -n "$HERMES_AUTH_JSON_BOOTSTRAP" ]; then
    printf '%s' "$HERMES_AUTH_JSON_BOOTSTRAP" > "$HERMES_HOME/auth.json"
    chmod 600 "$HERMES_HOME/auth.json"
fi

# Sync bundled skills
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

echo "═══════════════════════════════════════════════════════════"
echo "Hermes Agent on Railway"
echo "Dashboard: $PORT  |  Gateway running in background"
echo "═══════════════════════════════════════════════════════════"

# Start Gateway in background (for Telegram/Discord/etc)
hermes gateway run > "$HERMES_HOME/logs/gateway.log" 2>&1 &

# Small delay so gateway initializes
sleep 2

# Start Web Dashboard in foreground on Railway's assigned port
exec hermes dashboard \
    --host 0.0.0.0 \
    --port "$PORT" \
    --insecure \
    --no-open