#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERRED_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ROOT="${OPENCLAW_ROOT:-${INFERRED_ROOT:-/home/yeqiuqiu/clawd-architect}}"
TEXTFILE_DIR="${OPENCLAW_TEXTFILE_DIR:-$ROOT/ops/telemetry/textfile}"
EVENT_ROUTER="${OPENCLAW_EVENT_ROUTER_SCRIPT:-$ROOT/ops/openclaw/event_router.sh}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"

CANARY_FILE="$TEXTFILE_DIR/openclaw_hourly_canary_last_success_epoch.prom"
AUTOPILOT_FILE="$TEXTFILE_DIR/openclaw_autopilot_tick_last_success_epoch.prom"
LIVE_FILE="$TEXTFILE_DIR/openclaw_hl_terminal_live_last_success_epoch.prom"
PARITY_FILE="$TEXTFILE_DIR/openclaw_competitive_parity_last_success_epoch.prom"

CANARY_STALE_SEC="${OPENCLAW_SLO_CANARY_STALE_SEC:-5400}"       # 90m for hourly canary
AUTOPILOT_STALE_SEC="${OPENCLAW_SLO_AUTOPILOT_STALE_SEC:-2700}" # 45m for 15m tick
LIVE_STALE_SEC="${OPENCLAW_SLO_LIVE_STALE_SEC:-1800}"           # 30m for 5m live watchdog
PARITY_STALE_SEC="${OPENCLAW_SLO_PARITY_STALE_SEC:-777600}"     # 9d for weekly parity harness
EVENT_COOLDOWN_SEC="${OPENCLAW_SLO_EVENT_COOLDOWN_SEC:-1800}"   # 30m dedupe cooldown

# shellcheck source=ops/openclaw/lib/blocker_routing.sh
source "$BLOCKER_ROUTING_LIB"

OPENCLAW_BLOCKER_EVENT_ROUTER="$EVENT_ROUTER"
OPENCLAW_BLOCKER_SOURCE="watchdog.slo_staleness"
OPENCLAW_BLOCKER_COOLDOWN_SEC="$EVENT_COOLDOWN_SEC"
OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF="$TEXTFILE_DIR"

if [[ ! -x "$EVENT_ROUTER" ]]; then
  openclaw_watchdog_route_blocker "event_router_missing" "task=slo_staleness; reason=missing_event_router" "$EVENT_ROUTER"
  exit 0
fi

extract_epoch() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo 0
    return 0
  fi
  awk '{for(i=1;i<=NF;i++){if($i ~ /^[0-9]+$/){v=$i}}} END{if(v=="") v=0; print v}' "$file"
}

now_epoch="$(date +%s)"
canary_epoch="$(extract_epoch "$CANARY_FILE")"
autopilot_epoch="$(extract_epoch "$AUTOPILOT_FILE")"
live_epoch="$(extract_epoch "$LIVE_FILE")"
parity_epoch="$(extract_epoch "$PARITY_FILE")"

issues=()
issue_keys=()

if [[ "$canary_epoch" -le 0 ]]; then
  issues+=("canary_metric_missing")
  issue_keys+=("canary_missing")
else
  canary_age=$(( now_epoch - canary_epoch ))
  if (( canary_age > CANARY_STALE_SEC )); then
    issues+=("canary_stale=${canary_age}s>${CANARY_STALE_SEC}s")
    issue_keys+=("canary_stale")
  fi
fi

if [[ "$autopilot_epoch" -le 0 ]]; then
  issues+=("autopilot_metric_missing")
  issue_keys+=("autopilot_missing")
else
  autopilot_age=$(( now_epoch - autopilot_epoch ))
  if (( autopilot_age > AUTOPILOT_STALE_SEC )); then
    issues+=("autopilot_stale=${autopilot_age}s>${AUTOPILOT_STALE_SEC}s")
    issue_keys+=("autopilot_stale")
  fi
fi

if [[ "$live_epoch" -le 0 ]]; then
  issues+=("live_metric_missing")
  issue_keys+=("live_missing")
else
  live_age=$(( now_epoch - live_epoch ))
  if (( live_age > LIVE_STALE_SEC )); then
    issues+=("live_stale=${live_age}s>${LIVE_STALE_SEC}s")
    issue_keys+=("live_stale")
  fi
fi

if [[ "$parity_epoch" -le 0 ]]; then
  issues+=("parity_metric_missing")
  issue_keys+=("parity_missing")
else
  parity_age=$(( now_epoch - parity_epoch ))
  if (( parity_age > PARITY_STALE_SEC )); then
    issues+=("parity_stale=${parity_age}s>${PARITY_STALE_SEC}s")
    issue_keys+=("parity_stale")
  fi
fi

if [[ ${#issues[@]} -eq 0 ]]; then
  exit 0
fi

sig_input="$(IFS='|'; printf '%s' "${issue_keys[*]}")"
summary="$(IFS='; '; echo "${issues[*]}")"

openclaw_route_blocker \
  --event-router "$OPENCLAW_BLOCKER_EVENT_ROUTER" \
  --source "$OPENCLAW_BLOCKER_SOURCE" \
  --key "slo_staleness" \
  --severity critical \
  --summary "slo_staleness ${summary}" \
  --evidence-ref "$TEXTFILE_DIR" \
  --fingerprint-input "$sig_input" \
  --cooldown-sec "$OPENCLAW_BLOCKER_COOLDOWN_SEC"
exit 0
