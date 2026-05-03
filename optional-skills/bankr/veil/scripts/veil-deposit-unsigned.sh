#!/usr/bin/env bash
# Build a Bankr-compatible unsigned deposit tx JSON (no signing).
# Usage: veil-deposit-unsigned.sh <asset> <amount> [extra flags...]
#   asset: ETH or USDC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ASSET="${1:?asset required (ETH, USDC)}"
AMOUNT="${2:?amount required}"
shift 2 || true

veil_cli deposit "$ASSET" "$AMOUNT" --unsigned --quiet "$@"
