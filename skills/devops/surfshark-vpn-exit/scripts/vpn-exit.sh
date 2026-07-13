#!/usr/bin/env bash
#
# vpn-exit.sh — manage a userspace WireGuard->SOCKS5 egress proxy (wireproxy)
# exposed as socks5h://127.0.0.1:<port>, and point systemd --user services at it.
#
# All values are env-driven; no personal data, no hardcoded hosts or keys.
#
#   SOCKS_PORT      local SOCKS5 port                (default: 1080)
#   WIREPROXY_BIN   wireproxy binary on PATH         (default: wireproxy)
#   WIREPROXY_CONF  WireGuard config w/ [Socks5]     (default: ~/.config/wireproxy/exit.conf)
#   SERVICE_NAME    systemd --user unit name         (default: hermes-wireproxy)
#   VERIFY_URL      IP/ASN echo endpoint             (default: https://ipinfo.io/json)
#
# Usage: vpn-exit.sh {install-service|up|down|status|verify|wire-service <unit>|persist}
set -euo pipefail

SOCKS_PORT="${SOCKS_PORT:-1080}"
WIREPROXY_BIN="${WIREPROXY_BIN:-wireproxy}"
WIREPROXY_CONF="${WIREPROXY_CONF:-$HOME/.config/wireproxy/exit.conf}"
SERVICE_NAME="${SERVICE_NAME:-hermes-wireproxy}"
VERIFY_URL="${VERIFY_URL:-https://ipinfo.io/json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$(cd "$SCRIPT_DIR/../templates" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
PROXY_URL="socks5h://127.0.0.1:${SOCKS_PORT}"

log()  { printf '[vpn-exit] %s\n' "$*" >&2; }
die()  { printf '[vpn-exit] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

# Extract a JSON string field without requiring jq.
json_field() { # <field> reads stdin
  if command -v jq >/dev/null 2>&1; then
    jq -r ".${1} // empty"
  else
    grep -o "\"${1}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" | head -1 | sed 's/.*: *"\(.*\)"/\1/'
  fi
}

public_ip() { # [proxy_url]
  if [ -n "${1:-}" ]; then
    curl --proxy "$1" -fsS --max-time 20 "$VERIFY_URL"
  else
    curl -fsS --max-time 20 "$VERIFY_URL"
  fi
}

render() { # <template> -> stdout, expands the documented vars only
  sed \
    -e "s|@WIREPROXY_BIN@|${WIREPROXY_BIN}|g" \
    -e "s|@WIREPROXY_CONF@|${WIREPROXY_CONF}|g" \
    -e "s|@SOCKS_PORT@|${SOCKS_PORT}|g" \
    -e "s|@PROXY_URL@|${PROXY_URL}|g" \
    "$1"
}

cmd_install_service() {
  need "$WIREPROXY_BIN"
  [ -f "$WIREPROXY_CONF" ] || log "note: $WIREPROXY_CONF not found yet — create it from templates/wireproxy.conf.template"
  mkdir -p "$UNIT_DIR"
  render "$TEMPLATES_DIR/wireproxy.service.template" > "$UNIT_DIR/${SERVICE_NAME}.service"
  systemctl --user daemon-reload
  log "installed $UNIT_DIR/${SERVICE_NAME}.service (proxy=$PROXY_URL)"
}

cmd_up() {
  systemctl --user start "${SERVICE_NAME}.service"
  for _ in $(seq 1 20); do
    if curl --proxy "$PROXY_URL" -fsS --max-time 3 "$VERIFY_URL" >/dev/null 2>&1; then
      log "proxy is up on $PROXY_URL"; return 0
    fi
    sleep 0.5
  done
  die "proxy did not come up on $PROXY_URL — check: systemctl --user status ${SERVICE_NAME}"
}

cmd_down() {
  systemctl --user stop "${SERVICE_NAME}.service" || true
  log "stopped ${SERVICE_NAME}"
}

cmd_status() {
  systemctl --user --no-pager status "${SERVICE_NAME}.service" || true
}

cmd_verify() {
  need curl
  local direct proxied dip pip dorg porg
  direct="$(public_ip "" || true)"
  proxied="$(public_ip "$PROXY_URL" || true)"
  [ -n "$proxied" ] || die "no response through $PROXY_URL — tunnel down (refusing to fall back to datacenter IP)"
  dip="$(printf '%s' "$direct"  | json_field ip)"
  pip="$(printf '%s' "$proxied" | json_field ip)"
  dorg="$(printf '%s' "$direct"  | json_field org)"
  porg="$(printf '%s' "$proxied" | json_field org)"
  log "direct : ip=${dip:-?} org=${dorg:-?}"
  log "proxied: ip=${pip:-?} org=${porg:-?}"
  [ -n "$pip" ] || die "could not read proxied IP from $VERIFY_URL"
  if [ -n "$dip" ] && [ "$dip" = "$pip" ]; then
    die "proxied IP equals direct IP ($pip) — egress is NOT routed through the VPN"
  fi
  log "OK: egress is routed through the VPN exit"
}

cmd_wire_service() {
  local unit="${1:-}"
  [ -n "$unit" ] || die "usage: vpn-exit.sh wire-service <systemd-user-unit>"
  local dropin="$UNIT_DIR/${unit}.service.d"
  mkdir -p "$dropin"
  render "$TEMPLATES_DIR/proxy-env.conf.template" > "$dropin/10-proxy.conf"
  systemctl --user daemon-reload
  log "wired ${unit} egress to $PROXY_URL — restart it: systemctl --user restart ${unit}"
}

cmd_persist() {
  systemctl --user enable "${SERVICE_NAME}.service"
  loginctl enable-linger "$USER" || log "could not enable linger (needs polkit/root) — start manually after login"
  log "enabled ${SERVICE_NAME} on boot + linger for $USER"
}

main() {
  local sub="${1:-}"; shift || true
  case "$sub" in
    install-service) cmd_install_service ;;
    up)              cmd_up ;;
    down)            cmd_down ;;
    status)          cmd_status ;;
    verify)          cmd_verify ;;
    wire-service)    cmd_wire_service "${1:-}" ;;
    persist)         cmd_persist ;;
    *) die "usage: vpn-exit.sh {install-service|up|down|status|verify|wire-service <unit>|persist}" ;;
  esac
}

main "$@"
