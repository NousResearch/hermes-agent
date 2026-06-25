#!/usr/bin/env bash
set -euo pipefail

SWAP_SIZE="${1:-6G}"
SWAP_FILE="${SECOND_BRAIN_SWAP_FILE:-/swapfile-second-brain}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root." >&2
  exit 1
fi

if swapon --show=NAME --noheadings | grep -qx "$SWAP_FILE"; then
  echo "Swap already active at $SWAP_FILE"
  swapon --show
  exit 0
fi

if [[ ! -f "$SWAP_FILE" ]]; then
  if command -v fallocate >/dev/null 2>&1; then
    fallocate -l "$SWAP_SIZE" "$SWAP_FILE"
  else
    dd if=/dev/zero of="$SWAP_FILE" bs=1M count="$(numfmt --from=iec "$SWAP_SIZE" | awk '{print int($1 / 1024 / 1024)}')"
  fi
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
fi

swapon "$SWAP_FILE"

if ! grep -qE "^$SWAP_FILE\\s" /etc/fstab; then
  printf '%s none swap sw 0 0\n' "$SWAP_FILE" >> /etc/fstab
fi

sysctl vm.swappiness=20
cat >/etc/sysctl.d/99-second-brain-swap.conf <<EOF
vm.swappiness=20
EOF

swapon --show
