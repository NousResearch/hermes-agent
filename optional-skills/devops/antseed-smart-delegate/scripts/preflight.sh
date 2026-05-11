#!/usr/bin/env bash
# antseed-smart-delegate/preflight.sh
# Health check before delegating through AntSeed P2P network.
# Outputs JSON on stdout, human-readable on stderr.
# Exit: 0=ok, 1=warning, 2=cannot-delegate
set -euo pipefail

PROXY_URL="http://127.0.0.1:8377"
ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /usr/local/bin/antseed)"
MAX_WAIT=5

# Parse AntSeed table output using Python (handles Unicode box-drawing │)
# Args: status_output, field_name → value or ""
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

# --- Collect status ---
issues=()
proxy_up="false"
peer_pinned="none"
peer_name=""
deposits_avail="0"
deposits_reserved="0"
wallet="unknown"
channels="0"

# 1. Check buyer proxy (any HTTP response = up)
http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$MAX_WAIT" "$PROXY_URL/v1/models" 2>/dev/null || echo "000")
if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 600 ]; then
  proxy_up="true"
else
  issues+=("proxy_down")
fi

# 2. Check antseed CLI + buyer status
if [ -x "$ANTSEED_BIN" ]; then
  STATUS_OUTPUT=$("$ANTSEED_BIN" buyer status 2>/dev/null) || true

  wallet=$(parse_table "Wallet address" "$STATUS_OUTPUT")
  deposits_avail=$(parse_table "Deposits available" "$STATUS_OUTPUT" | grep -oP '[\d.]+' | head -1 || echo "0")
  deposits_reserved=$(parse_table "Deposits reserved" "$STATUS_OUTPUT" | grep -oP '[\d.]+' | head -1 || echo "0")
  peer_pinned_raw=$(parse_table "Pinned peer" "$STATUS_OUTPUT" | tr -d '[:space:]')
  peer_name=$(parse_table "Pinned service" "$STATUS_OUTPUT")
  channels=$(parse_table "Active channels" "$STATUS_OUTPUT" | grep -oP '\d+' | head -1 || echo "0")

  if [ -z "$peer_pinned_raw" ] || [ "$peer_pinned_raw" = "none" ] || [ "$peer_pinned_raw" = "│" ]; then
    peer_pinned="none"
    issues+=("no_peer")
  else
    peer_pinned="$peer_pinned_raw"
  fi

  [ "${deposits_avail:-0}" = "0" ] && issues+=("no_funds")
else
  issues+=("no_antseed_cli")
fi

# Determine overall status
can_delegate="true"
[ "$proxy_up" != "true" ] && can_delegate="false"
[ "$peer_pinned" = "none" ] && can_delegate="false"
[ "${deposits_avail:-0}" = "0" ] && can_delegate="false"

exit_code=0
[ "$can_delegate" != "true" ] && exit_code=2
[ ${#issues[@]} -gt 0 ] && [ "$can_delegate" = "true" ] && exit_code=1

# Build issues JSON array
issues_json=""
for i in "${issues[@]+"${issues[@]}"}"; do
  [ -n "$issues_json" ] && issues_json="${issues_json},"
  issues_json="${issues_json}\"$i\""
done

cat <<JSONEOF
{
  "ok": $can_delegate,
  "proxy_up": $proxy_up,
  "proxy_http": "${http_code:-000}",
  "peer_pinned": "$peer_pinned",
  "peer_name": "${peer_name:-}",
  "deposits_usdc": ${deposits_avail:-0},
  "reserved_usdc": ${deposits_reserved:-0},
  "wallet": "${wallet:-unknown}",
  "active_channels": ${channels:-0},
  "issues": [$issues_json],
  "can_delegate": $can_delegate
}
JSONEOF

# Human-readable summary to stderr
if [ "$can_delegate" = "true" ]; then
  echo "✅ Ready to delegate via AntSeed" >&2
  echo "   Peer: ${peer_name:-unknown} (${peer_pinned:0:12}...)" >&2
  echo "   Deposits: ${deposits_avail:-0} USDC available (${deposits_reserved:-0} reserved)" >&2
  echo "   Channels: ${channels:-0} active" >&2
else
  echo "❌ Cannot delegate through AntSeed:" >&2
  for i in "${issues[@]+"${issues[@]}"}"; do
    case "$i" in
      proxy_down)     echo "   • Buyer proxy not responding at $PROXY_URL (HTTP ${http_code:-000})" >&2 ;;
      no_peer)       echo "   • No peer pinned — run: antseed buyer connection set --peer <id>" >&2 ;;
      no_funds)      echo "   • No deposits — run: antseed buyer deposit <amount>" >&2 ;;
      no_antseed_cli) echo "   • antseed CLI not found at $ANTSEED_BIN" >&2 ;;
      *)             echo "   • $i" >&2 ;;
    esac
  done
fi

exit $exit_code
