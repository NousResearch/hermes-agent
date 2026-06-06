#!/usr/bin/env bash
set -euo pipefail

VPS_HOST="${HERMES_VPS_HOST:-linux-nat@103.142.150.185}"
REMOTE_ENTRY="${HERMES_NAT_ENTRY:-/home/linux-nat/bin/hermes-nat}"
LOCAL_PORT="${HERMES_DASHBOARD_LOCAL_PORT:-9119}"
REMOTE_PORT="${HERMES_DASHBOARD_REMOTE_PORT:-9119}"

usage() {
  cat <<'EOF'
Usage:
  scripts/enter_vps_nat.sh
  scripts/enter_vps_nat.sh --tunnel
  scripts/enter_vps_nat.sh --status

Options:
  --tunnel  Open the Hermes dashboard SSH tunnel, then enter the nat worktree.
  --status  Check the nat worktree and dashboard service without opening a shell.
EOF
}

case "${1:-}" in
  "")
    exec ssh -t "$VPS_HOST" "$REMOTE_ENTRY"
    ;;
  --tunnel)
    exec ssh -t -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$VPS_HOST" "$REMOTE_ENTRY"
    ;;
  --status)
    exec ssh "$VPS_HOST" "$REMOTE_ENTRY --status"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
