#!/usr/bin/env bash
# publish-signal.sh - Publish a trading signal to Bankr Signals
#
# Requires: node, a wallet private key in environment
#
# Usage:
#   export PRIVATE_KEY="0x..."
#   ./publish-signal.sh LONG ETH 2650.00 5 0xTX_HASH 100 "RSI oversold, MACD crossover"
#
# Args: action token entryPrice leverage txHash collateralUsd [reasoning]

set -euo pipefail

API_URL="${BANKR_SIGNALS_URL:-https://bankrsignals.com}/api/signals"

ACTION="${1:?Usage: publish-signal.sh ACTION TOKEN ENTRY_PRICE LEVERAGE TX_HASH COLLATERAL_USD [REASONING]}"
TOKEN="${2:?Missing TOKEN}"
ENTRY_PRICE="${3:?Missing ENTRY_PRICE}"
LEVERAGE="${4:-1}"
TX_HASH="${5:?Missing TX_HASH}"
COLLATERAL_USD="${6:?Missing COLLATERAL_USD}"
REASONING="${7:-}"

if [ -z "${PRIVATE_KEY:-}" ]; then
  echo "Error: PRIVATE_KEY environment variable required" >&2
  exit 1
fi

# Derive wallet address and sign message using node + viem
TIMESTAMP=$(date +%s)

RESULT=$(node -e "
const { privateKeyToAccount } = require('viem/accounts');
const account = privateKeyToAccount(process.env.PRIVATE_KEY);
const provider = account.address.toLowerCase();
const msg = 'bankr-signals:signal:' + provider + ':${ACTION}:${TOKEN}:${TIMESTAMP}';
account.signMessage({ message: msg }).then(sig => {
  console.log(JSON.stringify({ provider, message: msg, signature: sig }));
});
" 2>/dev/null)

PROVIDER=$(echo "$RESULT" | jq -r .provider)
MESSAGE=$(echo "$RESULT" | jq -r .message)
SIGNATURE=$(echo "$RESULT" | jq -r .signature)

BODY=$(jq -n \
  --arg provider "$PROVIDER" \
  --arg action "$ACTION" \
  --arg token "$TOKEN" \
  --argjson entryPrice "$ENTRY_PRICE" \
  --argjson leverage "$LEVERAGE" \
  --arg txHash "$TX_HASH" \
  --argjson collateralUsd "$COLLATERAL_USD" \
  --arg reasoning "$REASONING" \
  --arg message "$MESSAGE" \
  --arg signature "$SIGNATURE" \
  '{provider: $provider, action: $action, token: $token, entryPrice: $entryPrice,
    leverage: $leverage, txHash: $txHash, collateralUsd: $collateralUsd,
    reasoning: $reasoning, message: $message, signature: $signature}')

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "$BODY")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY_OUT=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
  echo "Published: $ACTION $TOKEN @ \$$ENTRY_PRICE (${LEVERAGE}x, \$${COLLATERAL_USD} collateral)"
  echo "$BODY_OUT" | jq -r '.id // .signal.id // "ok"' 2>/dev/null
else
  echo "Error ($HTTP_CODE): $BODY_OUT" >&2
  exit 1
fi
