#!/usr/bin/env bash
# Withdraw from private pool to a public address (executes locally using VEIL_KEY).
# Usage: veil-withdraw.sh <asset> <amount> <recipient>
#   asset: ETH or USDC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ASSET="${1:?asset required (ETH, USDC)}"
AMOUNT="${2:?amount required}"
RECIPIENT="${3:?recipient address required}"

veil_cli withdraw "$ASSET" "$AMOUNT" "$RECIPIENT" --quiet
