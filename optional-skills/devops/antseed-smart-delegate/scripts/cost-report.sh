#!/usr/bin/env bash
# antseed-smart-delegate/cost-report.sh
# Show AntSeed spending, channel state, and wallet balance.
# Usage: bash cost-report.sh [--json]
set -euo pipefail

ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /usr/local/bin/antseed)"
OUTPUT_FORMAT="${1:-text}"

# Parse AntSeed table output using Python (handles Unicode box-drawing │)
parse_table() {
  python3 -c "
import sys
text = sys.stdin.read()
key = sys.argv[1]
for line in text.splitlines():
    if key.lower() in line.lower():
        parts = line.split('\u2502')
        if len(parts) >= 3:
            val = parts[2].strip()  # value is after 2nd │
            if val and val.lower() != key.lower():
                print(val)
                break
" "$1" <<< "$2"
}

# --- Collect data ---
STATUS_OUTPUT=$("$ANTSEED_BIN" buyer status 2>/dev/null || true)
METERING_OUTPUT=$("$ANTSEED_BIN" buyer metering 2>/dev/null || true)

wallet=$(parse_table "Wallet address" "$STATUS_OUTPUT")
deposits_avail=$(parse_table "Deposits available" "$STATUS_OUTPUT" | grep -oP '[\d.]+' | head -1 || echo "0")
deposits_reserved=$(parse_table "Deposits reserved" "$STATUS_OUTPUT" | grep -oP '[\d.]+' | head -1 || echo "0")
channels=$(parse_table "Active channels" "$STATUS_OUTPUT" | grep -oP '\d+' | head -1 || echo "0")
peer_pinned=$(parse_table "Pinned peer" "$STATUS_OUTPUT" | tr -d '[:space:]' || echo "none")
peer_name=$(parse_table "Pinned service" "$STATUS_OUTPUT" || echo "")

# Parse metering
total_in_tokens="0"
total_out_tokens="0"
total_usd_spent="0"
channel_peer=""
channel_requests="0"

if [ -n "$METERING_OUTPUT" ]; then
  total_usd_spent=$(echo "$METERING_OUTPUT" | grep -oiP '[\d.]+\s*USDC' | grep -oP '[\d.]+' | head -1 || echo "0")
  total_in_tokens=$(echo "$METERING_OUTPUT" | grep -oiP '[\d.]+\s*(tokens?|in)' | grep -oP '[\d.]+' | head -1 || echo "0")
  channel_requests=$(echo "$METERING_OUTPUT" | grep -oiP '(requests?:?\s*|count\s*)\K[\d]+' | head -1 || echo "0")
  channel_peer=$(echo "$METERING_OUTPUT" | grep -i "peer\|seller" | head -1 | grep -oP '[0-9a-f]{8,}' | head -1 || echo "")
fi

available_for_new=$(python3 -c "print(max(0, ${deposits_avail:-0} - ${deposits_reserved:-0}))" 2>/dev/null || echo "0")

# --- Output ---
if [ "$OUTPUT_FORMAT" = "--json" ] || [ "$OUTPUT_FORMAT" = "json" ]; then
  cat <<JSONEOF
{
  "wallet": "${wallet:-unknown}",
  "deposits": { "available": ${deposits_avail:-0}, "reserved": ${deposits_reserved:-0}, "available_for_new_channels": $available_for_new },
  "channels": { "active": ${channels:-0}, "peer": "${peer_pinned:0:16}...", "peer_name": "${peer_name:-}", "requests": $channel_requests },
  "metering": { "tokens_in_est": $total_in_tokens, "usd_est": $total_usd_spent },
  "timestamp": "$(date -Iseconds)"
}
JSONEOF
else
  cat <<TEXTEOF

🐝 AntSeed Cost Report
═══════════════════════════════════════
Wallet:     ${wallet:0:10}...${wallet: -6}
Deposits:   ${deposits_avail:-0} USDC (${deposits_reserved:-0} reserved)
Available:  ~$available_for_new USDC for new channels

Channel${channels:+(s)} (${peer_name:-no peer}):
  Status:    $([ "${channels:-0}" -gt 0 ] && echo "✅ active" || echo "⚬ none")
  Requests:  $channel_requests
  Peer:      ${peer_pinned:0:16}...

Spending (session):
  Est. spent: ~$total_usd_spent USDC
  Tokens in: ~$total_in_tokens

───────────────────────────────────────
$(date '+%Y-%m-%d %H:%M:%S %Z')
TEXTEOF
fi
