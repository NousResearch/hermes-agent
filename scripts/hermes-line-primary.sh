#!/usr/bin/env bash
set -euo pipefail

# Hermes LINE primary-host helper for macOS.
# Manages:
#   - Hermes gateway launchd service
#   - Cloudflared named tunnel launch agent

ACTION="${1:-status}"
HERMES_DIR="${HERMES_DIR:-$HOME/.hermes/hermes-agent}"
HERMES_HOME_DIR="${HERMES_HOME_DIR:-$HOME/.hermes}"
TUNNEL_PLIST_PATH="${TUNNEL_PLIST_PATH:-$HOME/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist}"
TUNNEL_LABEL="${TUNNEL_LABEL:-com.soichiyo.cloudflared.hermes-line}"
GATEWAY_LABEL="${GATEWAY_LABEL:-ai.hermes.gateway}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|restart|status|logs]

start   Start Hermes gateway service and Cloudflare named tunnel
stop    Stop Hermes gateway service and Cloudflare named tunnel
restart Restart Hermes gateway service and Cloudflare named tunnel
status  Show service status
logs    Tail Hermes + cloudflared logs
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "❌ Missing file: $path" >&2
    exit 1
  fi
}

gateway_status_line() {
  if launchctl list | grep -q "$GATEWAY_LABEL"; then
    echo "loaded"
  else
    echo "not loaded"
  fi
}

tunnel_status_line() {
  if launchctl list | grep -q "$TUNNEL_LABEL"; then
    echo "loaded"
  else
    echo "not loaded"
  fi
}

start_tunnel() {
  require_file "$TUNNEL_PLIST_PATH"
  launchctl bootout "gui/$(id -u)/$TUNNEL_LABEL" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$TUNNEL_PLIST_PATH"
  launchctl kickstart "gui/$(id -u)/$TUNNEL_LABEL"
}

stop_tunnel() {
  launchctl bootout "gui/$(id -u)/$TUNNEL_LABEL" >/dev/null 2>&1 || true
}

case "$ACTION" in
  start)
    echo "==> Starting Hermes gateway..."
    cd "$HERMES_DIR"
    "$HERMES_DIR/venv/bin/hermes" gateway install >/dev/null 2>&1 || true
    "$HERMES_DIR/venv/bin/hermes" gateway start
    echo "==> Starting Cloudflare named tunnel..."
    start_tunnel
    echo "✅ Hermes LINE primary stack started"
    ;;
  stop)
    echo "==> Stopping Hermes gateway..."
    cd "$HERMES_DIR"
    "$HERMES_DIR/venv/bin/hermes" gateway stop || true
    echo "==> Stopping Cloudflare named tunnel..."
    stop_tunnel
    echo "✅ Hermes LINE primary stack stopped"
    ;;
  restart)
    "$0" stop
    sleep 2
    "$0" start
    ;;
  status)
    echo "[Hermes gateway]"
    echo "  launchd: $(gateway_status_line)"
    "$HERMES_DIR/venv/bin/hermes" gateway status || true
    echo
    echo "[Cloudflare tunnel]"
    echo "  launchd: $(tunnel_status_line)"
    echo "  plist:   $TUNNEL_PLIST_PATH"
    if [[ -f "$HOME/.cloudflared/config.yml" ]]; then
      echo "  config:  $HOME/.cloudflared/config.yml"
      sed -n '1,80p' "$HOME/.cloudflared/config.yml" | sed 's/^/    /'
    else
      echo "  config:  missing"
    fi
    ;;
  logs)
    touch "$HERMES_HOME_DIR/logs/gateway.log" "$HERMES_HOME_DIR/logs/cloudflared-line.log"
    tail -f "$HERMES_HOME_DIR/logs/gateway.log" "$HERMES_HOME_DIR/logs/cloudflared-line.log"
    ;;
  *)
    usage
    exit 1
    ;;
esac
