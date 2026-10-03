#!/bin/bash
# From-scratch VPS (or peer SOCKS) install. Run as the agent user.
# Usage:
#   ./install.sh check   # no sudo: print what is missing (agent first step)
#   ./install.sh vps     # this VPS: Camoufox + Xvfb + loopback VNC + wg noVNC
#   ./install.sh peer    # other WireGuard host: SOCKS5 on PEER_WG_IP:1080
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TAKEOVER_DEST:-$HOME/.hermes/takeover}"
MODE="${1:-}"

need_env() {
  local f="$DEST/env"
  if [ ! -f "$f" ]; then
    mkdir -p "$DEST"
    cp "$ROOT/templates/env.example" "$f"
    echo "Fill $f (at least BIND_IP) and re-run." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a
  # shellcheck disable=SC1091
  . "$f"
  set +a
}

reject_public() {
  local ip="$1" label="$2"
  if [ -z "$ip" ] || [ "$ip" = "0.0.0.0" ] || [ "$ip" = "::" ]; then
    echo "$label must be a WireGuard address, never 0.0.0.0" >&2
    exit 1
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

load_env_optional() {
  local f="$DEST/env"
  if [ -f "$f" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$f"
    set +a
  fi
}

check_host() {
  load_env_optional
  local missing=()
  have_cmd xvfb-run || have_cmd Xvfb || missing+=(xvfb)
  have_cmd x11vnc || missing+=(x11vnc)
  have_cmd websockify || missing+=(websockify)
  [ -d /usr/share/novnc ] || missing+=(novnc)
  have_cmd python3 || missing+=(python3)
  local iface="${WG_IFACE:-wg0}"
  local wgip
  wgip="$(ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
  [ -n "$wgip" ] || missing+=(wg)
  local bind="${BIND_IP:-$wgip}"
  if [ -z "$bind" ] || [ "$bind" = "0.0.0.0" ] || [ "$bind" = "::" ]; then
    missing+=(bind_ip)
  fi
  if [ ! -x "$DEST/venv/bin/python" ] || ! "$DEST/venv/bin/python" -c "import camoufox" 2>/dev/null; then
    missing+=(camoufox)
  fi
  printf 'WG_IFACE=%s\n' "$iface"
  printf 'BIND_IP=%s\n' "${bind:-}"
  printf 'CAMOUFOX_VENV=%s\n' "$DEST/venv"
  if [ "${#missing[@]}" -eq 0 ]; then
    echo "MISSING="
    echo "READY=yes"
    return 0
  fi
  echo "MISSING=${missing[*]}"
  echo "READY=no"
  echo "NEXT=fill ~/.hermes/takeover/env BIND_IP if needed, then: $ROOT/scripts/install.sh vps  (sudo for apt)"
  return 1
}

apt_vps() {
  local need=()
  have_cmd Xvfb || need+=(xvfb)
  have_cmd x11vnc || need+=(x11vnc)
  have_cmd websockify || need+=(websockify)
  [ -d /usr/share/novnc ] || need+=(novnc)
  have_cmd python3 || need+=(python3 python3-venv python3-pip)
  have_cmd xdpyinfo || need+=(x11-utils)
  if [ "${#need[@]}" -gt 0 ]; then
    need+=(libgtk-3-0 libdbus-glib-1-2 libxt6)
  fi
  if [ "${#need[@]}" -eq 0 ]; then
    echo "apt packages already present"
    return 0
  fi
  if ! have_cmd apt-get; then
    echo "Need apt-get (Ubuntu/Debian VPS)." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y "${need[@]}"
  sudo apt-get install -y libasound2t64 2>/dev/null || sudo apt-get install -y libasound2 || true
}

venv_vps() {
  mkdir -p "$HOME/tmp" "$DEST"
  if [ ! -x "$DEST/venv/bin/python" ]; then
    python3 -m venv "$DEST/venv"
  fi
  "$DEST/venv/bin/pip" install -U pip
  "$DEST/venv/bin/pip" install -U 'camoufox[geoip]' playwright
  if "$DEST/venv/bin/python" -m camoufox path >/dev/null 2>&1; then
    echo "Camoufox browser already fetched"
    return 0
  fi
  TMPDIR="$HOME/tmp" "$DEST/venv/bin/python" -m camoufox fetch
}

copy_scripts() {
  mkdir -p "$DEST"
  install -m 755 "$ROOT/scripts/camoufox_server.py" "$DEST/camoufox_server.py"
  install -m 755 "$ROOT/scripts/hold_page.py" "$DEST/hold_page.py"
  install -m 755 "$ROOT/scripts/wg_socks5.py" "$DEST/wg_socks5.py"
  install -m 755 "$ROOT/templates/takeover_view.sh" "$DEST/takeover_view.sh"
}

write_user_unit() {
  local name="$1" body="$2"
  mkdir -p "$HOME/.config/systemd/user"
  printf '%s\n' "$body" > "$HOME/.config/systemd/user/$name"
}

install_vps_units() {
  reject_public "${BIND_IP:-}" BIND_IP
  write_user_unit camoufox-server.service "$(cat <<EOF
[Unit]
Description=Headed Camoufox Playwright server (Xvfb)
After=network.target

[Service]
Type=simple
ExecStart=$DEST/venv/bin/python $DEST/camoufox_server.py
Restart=always
RestartSec=5
Environment=DISPLAY=:98
Environment=TMPDIR=$HOME/tmp
EnvironmentFile=-$DEST/env

[Install]
WantedBy=default.target
EOF
)"
  write_user_unit camoufox-hold.service "$(cat <<EOF
[Unit]
Description=Hold one headed page so noVNC is not black
After=camoufox-server.service
Wants=camoufox-server.service

[Service]
Type=simple
ExecStart=$DEST/venv/bin/python $DEST/hold_page.py
Restart=on-failure
RestartSec=3
EnvironmentFile=-$DEST/env

[Install]
WantedBy=default.target
EOF
)"
  write_user_unit hermes-takeover-vnc.service "$(cat <<'EOF'
[Unit]
Description=x11vnc loopback mirror of Xvfb
After=camoufox-server.service

[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -display :98 -forever -shared -nopw -quiet -rfbport 5900 -nossl -listen 127.0.0.1 -localhost
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
)"
  write_user_unit hermes-takeover-novnc.service "$(cat <<EOF
[Unit]
Description=noVNC on WireGuard BIND_IP only
After=hermes-takeover-vnc.service
Wants=hermes-takeover-vnc.service

[Service]
Type=simple
EnvironmentFile=$DEST/env
ExecStart=/usr/bin/websockify --web /usr/share/novnc \${BIND_IP}:\${NOVNC_PORT} 127.0.0.1:\${VNC_PORT}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
)"
  systemctl --user daemon-reload
  systemctl --user enable --now camoufox-server.service camoufox-hold.service \
    hermes-takeover-vnc.service hermes-takeover-novnc.service
  if command -v loginctl >/dev/null 2>&1; then
    sudo loginctl enable-linger "$USER" || true
  fi
  echo "Owner URL: http://${BIND_IP}:${NOVNC_PORT:-6080}/vnc.html"
}

install_peer() {
  need_env
  reject_public "${PEER_WG_IP:-}" PEER_WG_IP
  mkdir -p "$DEST" "$HOME/.config/systemd/user"
  install -m 755 "$ROOT/scripts/wg_socks5.py" "$DEST/wg_socks5.py"
  write_user_unit wg-socks5.service "$(cat <<EOF
[Unit]
Description=WireGuard-peer SOCKS5 for browser egress
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 $DEST/wg_socks5.py
Restart=on-failure
RestartSec=3
EnvironmentFile=-$DEST/env

[Install]
WantedBy=default.target
EOF
)"
  systemctl --user daemon-reload
  systemctl --user enable --now wg-socks5.service
  if command -v loginctl >/dev/null 2>&1; then
    sudo loginctl enable-linger "$USER" || true
  fi
  echo "SOCKS on ${PEER_WG_IP}:${SOCKS_PORT:-1080} — point VPS HERMES_BROWSER_PROXY=socks5://${PEER_WG_IP}:${SOCKS_PORT:-1080}"
}

case "$MODE" in
  check)
    check_host
    ;;
  vps)
    need_env
    reject_public "${BIND_IP:-}" BIND_IP
    apt_vps
    copy_scripts
    venv_vps
    install_vps_units
    echo "Next: $ROOT/scripts/verify.sh"
    ;;
  peer)
    install_peer
    ;;
  *)
    echo "usage: $0 check|vps|peer" >&2
    exit 1
    ;;
esac
