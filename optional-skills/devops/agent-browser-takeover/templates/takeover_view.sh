#!/bin/bash
# Mirror an Xvfb headed browser via x11vnc -> noVNC on a WireGuard IP only.
# Usage: takeover_view.sh start|stop|status
# Env: DISPLAY_NUM VNC_PORT NOVNC_PORT BIND_IP WG_IFACE
set -u
BASE="${TAKEOVER_BASE:-$HOME/.hermes/takeover}"
PIDDIR="$BASE/pids"
DISPLAY_NUM="${DISPLAY_NUM:-98}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
WG_IFACE="${WG_IFACE:-wg0}"
BIND_IP="${BIND_IP:-}"
mkdir -p "$PIDDIR"

resolve_bind_ip() {
  if [ -z "$BIND_IP" ]; then
    BIND_IP="$(ip -4 -o addr show dev "$WG_IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
  fi
  if [ -z "$BIND_IP" ] || [ "$BIND_IP" = "0.0.0.0" ] || [ "$BIND_IP" = "::" ]; then
    echo "Set BIND_IP to your WireGuard address (or WG_IFACE). Never 0.0.0.0." >&2
    exit 1
  fi
}

is_up() { [ -f "$PIDDIR/$1.pid" ] && kill -0 "$(cat "$PIDDIR/$1.pid")" 2>/dev/null; }

start() {
  resolve_bind_ip
  if is_up x11vnc; then echo "x11vnc already running"; else
    # Never proxy to "localhost" (can resolve to ::1).
    x11vnc -display ":$DISPLAY_NUM" -forever -shared -nopw -quiet \
           -rfbport "$VNC_PORT" -nossl -listen 127.0.0.1 -localhost \
           >"$BASE/x11vnc.log" 2>&1 &
    echo $! > "$PIDDIR/x11vnc.pid"
    echo "x11vnc up on 127.0.0.1:${VNC_PORT} (display :$DISPLAY_NUM)"
  fi
  if is_up websockify; then echo "websockify already running"; else
    websockify --web /usr/share/novnc "$BIND_IP:$NOVNC_PORT" "127.0.0.1:$VNC_PORT" \
               >"$BASE/websockify.log" 2>&1 &
    echo $! > "$PIDDIR/websockify.pid"
    echo "noVNC ready: http://${BIND_IP}:${NOVNC_PORT}/vnc.html"
  fi
}

stop() {
  for s in x11vnc websockify; do
    if is_up "$s"; then kill "$(cat "$PIDDIR/$s.pid")" 2>/dev/null; rm -f "$PIDDIR/$s.pid"; echo "$s stopped"; else echo "$s not running"; fi
  done
}

status() {
  for s in x11vnc websockify; do
    if is_up "$s"; then echo "$s: RUNNING (pid $(cat "$PIDDIR/$s.pid"))"; else echo "$s: stopped"; fi
  done
  ss -tlnp 2>/dev/null | grep -E ":(${VNC_PORT}|${NOVNC_PORT}) " || true
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
