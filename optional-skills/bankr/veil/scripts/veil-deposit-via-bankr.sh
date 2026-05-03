#!/usr/bin/env bash
# Build unsigned deposit tx JSON and submit via Bankr.
# For ETH: single tx. For USDC: approve tx first, then deposit tx.
# Usage: veil-deposit-via-bankr.sh <asset> <amount> [extra flags...]
#   asset: ETH or USDC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

need_bin jq

ASSET="${1:?asset required (ETH, USDC)}"
AMOUNT="${2:?amount required}"
shift 2 || true

PAYLOAD=$(veil_cli deposit "$ASSET" "$AMOUNT" --unsigned --quiet "$@")

# Validate CLI output is valid JSON
if ! echo "$PAYLOAD" | jq empty 2>/dev/null; then
  echo "CLI did not return valid JSON:" >&2
  echo "$PAYLOAD" >&2
  exit 1
fi

# Check if payload is an array (ERC20: approve + deposit) or single object (ETH)
IS_ARRAY=$(echo "$PAYLOAD" | jq -r 'if type == "array" then "true" else "false" end')

if [[ "$IS_ARRAY" == "true" ]]; then
  ARRAY_LEN=$(echo "$PAYLOAD" | jq 'length')
  if [[ "$ARRAY_LEN" -ne 2 ]]; then
    echo "Expected 2 transactions (approve + deposit), got $ARRAY_LEN" >&2
    echo "$PAYLOAD" >&2
    exit 1
  fi

  APPROVE_TX=$(echo "$PAYLOAD" | jq -c '.[0]')
  DEPOSIT_TX=$(echo "$PAYLOAD" | jq -c '.[1]')

  echo "Submitting approval transaction..." >&2
  APPROVE_RESULT=$(echo "$APPROVE_TX" | "$SCRIPT_DIR/veil-bankr-submit-tx.sh") || {
    echo "Approval transaction failed" >&2
    echo "$APPROVE_RESULT" >&2
    exit 1
  }

  # Validate approval result is JSON
  if ! echo "$APPROVE_RESULT" | jq empty 2>/dev/null; then
    echo "Approval result is not valid JSON:" >&2
    echo "$APPROVE_RESULT" >&2
    exit 1
  fi

  APPROVE_STATUS=$(echo "$APPROVE_RESULT" | jq -r '.status // empty')
  if [[ "$APPROVE_STATUS" != "completed" ]]; then
    echo "Approval transaction not completed (status: ${APPROVE_STATUS:-unknown})" >&2
    ERROR_MSG=$(echo "$APPROVE_RESULT" | jq -r '.error // .message // .reason // empty')
    [[ -n "$ERROR_MSG" ]] && echo "Error: $ERROR_MSG" >&2
    echo "$APPROVE_RESULT" >&2
    exit 1
  fi

  echo "Submitting deposit transaction..." >&2
  echo "$DEPOSIT_TX" | "$SCRIPT_DIR/veil-bankr-submit-tx.sh"
else
  # ETH deposit: single tx
  echo "$PAYLOAD" | "$SCRIPT_DIR/veil-bankr-submit-tx.sh"
fi
