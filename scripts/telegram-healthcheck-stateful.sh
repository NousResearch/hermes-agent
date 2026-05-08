#!/bin/bash
set -euo pipefail

# Stateful Telegram watchdog for Hermes gateway.
#
# Unlike the legacy log-scraping helper, this reads the current runtime state
# from ~/.hermes/gateway_state.json and only restarts the gateway when Telegram
# is presently unhealthy and that state has remained stale for a while.

STATE_FILE="${HERMES_HOME:-$HOME/.hermes}/gateway_state.json"
LOG_FILE="${HERMES_HOME:-$HOME/.hermes}/logs/healthcheck.log"
MAX_STALE_SECONDS="${HERMES_TELEGRAM_HEALTHCHECK_MAX_STALE_SECONDS:-600}"
NOW=$(date -u +%s)

mkdir -p "$(dirname "$LOG_FILE")"

log_msg() {
  echo "$(date -u): $*" >> "$LOG_FILE"
}

if [[ ! -f "$STATE_FILE" ]]; then
  log_msg "state file missing; skipping restart"
  exit 0
fi

result=$(python3 - "$STATE_FILE" "$NOW" "$MAX_STALE_SECONDS" <<"PY"
import datetime
import json
import sys

state_file, now_s, max_stale = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

try:
    data = json.load(open(state_file))
except Exception as exc:
    print(f"state-unreadable:{exc.__class__.__name__}")
    sys.exit(0)

telegram = (data.get("platforms") or {}).get("telegram") or {}
state = (telegram.get("state") or "unknown").strip().lower() or "unknown"
updated_at = telegram.get("updated_at")

if state == "connected":
    print("healthy")
    sys.exit(0)

if not updated_at:
    print(f"unhealthy:no-updated-at:{state}")
    sys.exit(0)

try:
    ts = datetime.datetime.fromisoformat(updated_at.replace("Z", "+00:00")).timestamp()
except Exception:
    print(f"unhealthy:bad-updated-at:{state}")
    sys.exit(0)

age = now_s - int(ts)
if age < max_stale:
    print(f"recently-changed:{state}:{age}")
else:
    print(f"unhealthy:{state}:{age}")
PY
)

case "$result" in
  healthy)
    exit 0
    ;;
  recently-changed:*)
    log_msg "telegram not connected yet but state is recent ($result); skipping restart"
    exit 0
    ;;
  unhealthy:*)
    log_msg "telegram unhealthy ($result); restarting gateway"
    ~/.local/bin/hermes gateway restart >/dev/null 2>&1 || log_msg "restart command failed"
    exit 0
    ;;
  *)
    log_msg "healthcheck inconclusive ($result); skipping restart"
    exit 0
    ;;
esac
