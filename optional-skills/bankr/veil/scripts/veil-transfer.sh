#!/usr/bin/env bash
# Private transfer within the pool to another registered address.
# Usage: veil-transfer.sh <asset> <amount> <recipient>
#   asset: ETH or USDC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ASSET="${1:?asset required (ETH, USDC)}"
AMOUNT="${2:?amount required}"
RECIPIENT="${3:?recipient address required}"

veil_cli transfer "$ASSET" "$AMOUNT" "$RECIPIENT" --quiet
