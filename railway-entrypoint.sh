#!/bin/bash
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"
PORT="${PORT:-8080}"

# gosu privilege drop block (keep your existing one)
if [ "$(id -u)" = "0" ]; then
    # ... your existing gosu block ...
    exec gosu hermes "$0" "$@"
fi

source "${INSTALL_DIR}/.venv/bin/activate"

# Bootstrap + skills sync (keep your existing code)
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}
[ ! -f "$HERMES_HOME/.env" ] && cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
[ ! -f "$HERMES_HOME/config.yaml" ] && cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
[ ! -f "$HERMES_HOME/SOUL.md" ] && cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"

if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

echo "═══════════════════════════════════════════════════════════"
echo "Hermes Agent + Dashboard (behind Caddy Auth Proxy)"
echo "═══════════════════════════════════════════════════════════"

# Start Gateway in background
hermes gateway run > "$HERMES_HOME/logs/gateway.log" 2>&1 &

sleep 2

# Start Dashboard (must use --insecure + 0.0.0.0 because Caddy proxies to it)
exec hermes dashboard \
    --host 0.0.0.0 \
    --port "$PORT" \
    --insecure \
    --no-open