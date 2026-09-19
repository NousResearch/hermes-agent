#!/bin/bash
# Prove the stack is takeover-ready. Source ~/.hermes/takeover/env
set -euo pipefail
ENVF="${TAKEOVER_DEST:-$HOME/.hermes/takeover}/env"
# shellcheck disable=SC1090
set -a
. "$ENVF"
set +a

if [ -z "${BIND_IP:-}" ] || [ "$BIND_IP" = "0.0.0.0" ]; then
  echo "BIND_IP missing or public" >&2
  exit 1
fi

echo "== listeners =="
ss -tln | grep -E '127.0.0.1:5900' || { echo "RFB not on 127.0.0.1:5900" >&2; exit 1; }
ss -tln | grep -F "${BIND_IP}:${NOVNC_PORT:-6080}" || { echo "noVNC not on ${BIND_IP}:${NOVNC_PORT:-6080}" >&2; exit 1; }
if ss -tln | grep -q '0.0.0.0:6080'; then
  echo "noVNC must not bind 0.0.0.0:6080" >&2
  exit 1
fi

echo "== RFB banner =="
python3 - <<'PY'
import os, socket
s = socket.create_connection(("127.0.0.1", int(os.environ.get("VNC_PORT", "5900"))), 5)
banner = s.recv(12)
s.close()
print(banner)
if not banner.startswith(b"RFB "):
    raise SystemExit("no RFB banner")
PY

echo "== Camoufox WS =="
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:9377/json/version || true)"
echo "json/version $code"
[ "$code" = "200" ] || { echo "Camoufox not up" >&2; exit 1; }

if [ -n "${HERMES_BROWSER_PROXY:-}" ]; then
  echo "== peer SOCKS vs VPS WAN =="
  proxy="${HERMES_BROWSER_PROXY#socks5://}"
  via="$(curl -4 -sS --max-time 10 --socks5-hostname "$proxy" https://api.ipify.org || true)"
  wan="$(curl -4 -sS --max-time 10 https://api.ipify.org || true)"
  echo "proxy_origin=$via"
  echo "vps_wan=$wan"
  if [ -z "$via" ] || [ -z "$wan" ] || [ "$via" = "$wan" ]; then
    echo "peer SOCKS did not change public origin (or failed)" >&2
    exit 1
  fi
fi

echo "OK  viewer=http://${BIND_IP}:${NOVNC_PORT:-6080}/vnc.html"
