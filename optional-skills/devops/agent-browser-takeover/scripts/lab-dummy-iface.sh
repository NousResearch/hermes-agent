#!/usr/bin/env bash
# Give this box a VPN-shaped IPv4 so noVNC binds like the POC.
# Owner URL: http://10.13.37.1:6080/vnc.html
# Loopback is raw x11vnc only — do not bind the viewer there.
set -euo pipefail

IFACE="${TAKEOVER_LAB_IFACE:-takeover0}"
ADDR="${TAKEOVER_LAB_ADDR:-10.13.37.1}"
MODE="${1:-print}"

usage() {
  cat <<EOF
Usage: $0 [print|apply]
  print  (default) show the commands; no sudo
  apply  create dummy iface ${IFACE} with ${ADDR}/32 (needs sudo)

noVNC must bind ${ADDR}:6080. Do not bind the viewer to 127.0.0.1.
EOF
}

cmds() {
  cat <<EOF
ip link add ${IFACE} type dummy
ip addr add ${ADDR}/32 dev ${IFACE}
ip link set ${IFACE} up
# then in ~/.hermes/takeover/env:
#   WG_IFACE=${IFACE}
#   BIND_IP=${ADDR}
# owner URL: http://${ADDR}:6080/vnc.html
EOF
}

case "$MODE" in
  -h|--help|help) usage; exit 0 ;;
  print)
    cmds
    ;;
  apply)
    if [[ "$(id -u)" -ne 0 ]]; then
      echo "apply needs sudo" >&2
      exit 1
    fi
    if [[ "$ADDR" == "127.0.0.1" || "$ADDR" == "0.0.0.0" ]]; then
      echo "Never bind the lab viewer to ${ADDR}" >&2
      exit 1
    fi
    if ip link show "$IFACE" >/dev/null 2>&1; then
      echo "iface ${IFACE} already exists" >&2
    else
      ip link add "$IFACE" type dummy
    fi
    ip addr replace "${ADDR}/32" dev "$IFACE"
    ip link set "$IFACE" up
    echo "READY iface=${IFACE} BIND_IP=${ADDR}"
    echo "Owner URL: http://${ADDR}:6080/vnc.html"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
