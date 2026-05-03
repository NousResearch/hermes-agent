#!/usr/bin/env bash
# Merge/consolidate UTXOs by self-transfer.
# Usage: veil-merge.sh <asset> <amount>
#   asset: ETH or USDC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ASSET="${1:?asset required (ETH, USDC)}"
AMOUNT="${2:?amount required}"

veil_cli merge "$ASSET" "$AMOUNT" --quiet
