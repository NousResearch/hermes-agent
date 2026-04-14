#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Example cron entry (every 5 minutes):
# */5 * * * * /home/yeqiuqiu/clawd-architect/scripts/cron_walletdb_helius_phase1.sh >>/tmp/walletdb_phase1_cron.log 2>&1

if [[ -z "${HELIUS_API_KEY:-}" ]]; then
  echo "HELIUS_API_KEY is not set" >&2
  exit 2
fi

# Provide one or more addresses via WALLETDB_PHASE1_ADDRESSES (comma-separated)
ADDRS="${WALLETDB_PHASE1_ADDRESSES:-}"
ARGS=()
IFS="," read -r -a ADDR_LIST <<< "$ADDRS"
for addr in "${ADDR_LIST[@]}"; do
  if [[ -n "$addr" ]]; then
    ARGS+=("--address" "$addr")
  fi
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "No addresses provided. Set WALLETDB_PHASE1_ADDRESSES=addr1,addr2" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR/src" \
  python -m walletdb.bundles.helius_phase1_cli \
  "${ARGS[@]}" \
  --limit "${WALLETDB_PHASE1_LIMIT:-50}" \
  --min-wallets "${WALLETDB_PHASE1_MIN_WALLETS:-5}" \
  --min-edges "${WALLETDB_PHASE1_MIN_EDGES:-4}" \
  --max-hop "${WALLETDB_PHASE1_MAX_HOP:-2}" \
  --cooldown-min "${WALLETDB_PHASE1_COOLDOWN_MIN:-60}"
