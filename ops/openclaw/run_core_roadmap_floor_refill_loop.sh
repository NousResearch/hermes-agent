#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CONTRACT_PATH="${OPENCLAW_CORE_ROADMAP_REFILL_CONTRACT_PATH:-$ROOT/ops/openclaw/contract_no_nudge_continuity_watchdog.sh}"
WINDOW_SEC="${OPENCLAW_CORE_ROADMAP_REFILL_WINDOW_SEC:-240}"
INTERVAL_SEC="${OPENCLAW_CORE_ROADMAP_REFILL_INTERVAL_SEC:-120}"
LOCK_PATH="${OPENCLAW_CORE_ROADMAP_REFILL_LOCK_PATH:-$ROOT/state/continuity/locks/core_roadmap_floor_refill.lock}"
ERR_PATH="/tmp/core_roadmap_floor_refill_loop_${$}.err"

sanitize_inline() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

first_non_empty_line() {
  awk 'NF { print; exit }'
}

if ! [[ "$WINDOW_SEC" =~ ^[0-9]+$ ]] || [[ "$WINDOW_SEC" -le 0 ]]; then
  echo "BLOCKER: core_roadmap_floor_refill_loop_invalid_window_sec; value=${WINDOW_SEC}"
  exit 0
fi
if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SEC" -le 0 ]]; then
  echo "BLOCKER: core_roadmap_floor_refill_loop_invalid_interval_sec; value=${INTERVAL_SEC}"
  exit 0
fi
if [[ "$INTERVAL_SEC" -gt "$WINDOW_SEC" ]]; then
  echo "BLOCKER: core_roadmap_floor_refill_loop_invalid_interval_window_relation; interval_sec=${INTERVAL_SEC}; window_sec=${WINDOW_SEC}"
  exit 0
fi
if [[ ! -x "$CONTRACT_PATH" ]]; then
  echo "BLOCKER: core_roadmap_floor_refill_loop_missing_contract; path=${CONTRACT_PATH}"
  exit 0
fi

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "READY: core_roadmap_floor_refill_loop_already_active; lock_path=${LOCK_PATH}"
  exit 0
fi

cleanup() {
  rm -f "$ERR_PATH" 2>/dev/null || true
}
trap cleanup EXIT

start_ts="$(date +%s)"
deadline_ts=$((start_ts + WINDOW_SEC))
attempts=0

while :; do
  attempts=$((attempts + 1))

  set +e
  contract_out="$("$CONTRACT_PATH" 2>"$ERR_PATH")"
  contract_rc=$?
  set -e

  first_line="$(printf '%s\n' "$contract_out" | first_non_empty_line)"

  if [[ "$contract_rc" -ne 0 ]]; then
    err_raw="$(cat "$ERR_PATH" 2>/dev/null || true)"
    err_txt="$(sanitize_inline "$err_raw")"
    if [[ -z "$err_txt" ]]; then
      err_txt="$(sanitize_inline "$first_line")"
    fi
    if [[ -z "$err_txt" ]]; then
      err_txt="no_stderr"
    fi
    echo "BLOCKER: core_roadmap_floor_refill_loop_contract_exec_failed; rc=${contract_rc}; contract_path=${CONTRACT_PATH}; err=${err_txt:0:180}"
    exit 0
  fi

  if [[ -z "$first_line" || "$first_line" == "NO_REPLY" ]]; then
    :
  elif [[ "$first_line" == BLOCKER:* ]]; then
    echo "$first_line"
    exit 0
  elif [[ "$first_line" == READY:* || "$first_line" == PROGRESS:* ]]; then
    :
  else
    safe_line="$(sanitize_inline "$first_line")"
    echo "BLOCKER: core_roadmap_floor_refill_loop_invalid_first_line; contract_path=${CONTRACT_PATH}; first_line=${safe_line:0:180}"
    exit 0
  fi

  now_ts="$(date +%s)"
  if [[ "$now_ts" -ge "$deadline_ts" ]]; then
    break
  fi

  remaining_sec=$((deadline_ts - now_ts))
  sleep_sec="$INTERVAL_SEC"
  if [[ "$sleep_sec" -gt "$remaining_sec" ]]; then
    sleep_sec="$remaining_sec"
  fi
  if [[ "$sleep_sec" -gt 0 ]]; then
    sleep "$sleep_sec"
  fi
done

echo "READY: core_roadmap_floor_refill_loop_window_complete; attempts=${attempts}; window_sec=${WINDOW_SEC}; interval_sec=${INTERVAL_SEC}; contract_path=${CONTRACT_PATH}"
exit 0
