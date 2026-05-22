#!/bin/bash
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"
DASHBOARD_PORT=9119          # Fixed internal port

# gosu privilege drop (keep your existing block)
if [ "$(id -u)" = "0" ]; then
    # ... your existing gosu block stays here ...
    exec gosu hermes "$0" "$@"
fi

source "${INSTALL_DIR}/.venv/bin/activate"

# Bootstrap
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}
[ ! -f "$HERMES_HOME/.env" ] && cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
[ ! -f "$HERMES_HOME/config.yaml" ] && cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
[ ! -f "$HERMES_HOME/SOUL.md" ] && cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"

if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

echo "═══════════════════════════════════════════════════════════"
echo "Hermes Agent (Robust Mode) - Dashboard on :$DASHBOARD_PORT"
echo "═══════════════════════════════════════════════════════════"

# Start Gateway in background with proper signal handling
hermes gateway run > "$HERMES_HOME/logs/gateway.log" 2>&1 &
GATEWAY_PID=$!

# Start Dashboard on fixed port
exec hermes dashboard \
    --host 0.0.0.0 \
    --port $DASHBOARD_PORT \
    --insecure \
    --no-open