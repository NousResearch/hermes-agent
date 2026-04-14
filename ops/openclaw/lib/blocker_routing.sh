#!/usr/bin/env bash
# shellcheck shell=bash

# Shared event/blocker routing helpers for watchdog/wrapper scripts.
# - openclaw_route_event: thin normalized event_router invocation (returns router rc)
# - openclaw_route_blocker: blocker convenience wrapper with consistent fallback output
# - openclaw_watchdog_route_blocker: preconfigured wrapper using OPENCLAW_BLOCKER_* defaults

openclaw_route_event() {
  local event_router=""
  local source=""
  local key=""
  local severity="info"
  local summary=""
  local evidence_ref=""
  local cooldown_sec="1800"
  local fingerprint_input=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --event-router)
        event_router="${2:-}"; shift 2 ;;
      --source)
        source="${2:-}"; shift 2 ;;
      --key)
        key="${2:-}"; shift 2 ;;
      --severity)
        severity="${2:-}"; shift 2 ;;
      --summary)
        summary="${2:-}"; shift 2 ;;
      --evidence-ref)
        evidence_ref="${2:-}"; shift 2 ;;
      --cooldown-sec)
        cooldown_sec="${2:-}"; shift 2 ;;
      --fingerprint-input)
        fingerprint_input="${2:-}"; shift 2 ;;
      *)
        shift ;;
    esac
  done

  if [[ -z "$event_router" || ! -x "$event_router" || -z "$source" || -z "$key" ]]; then
    return 127
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  local -a router_cmd=(
    "$event_router"
    --source "$source"
    --key "$key"
    --severity "$severity"
    --summary "$summary"
    --evidence-ref "$evidence_ref"
    --cooldown-sec "$cooldown_sec"
  )
  if [[ -n "$fingerprint_input" ]]; then
    router_cmd+=(--fingerprint-input "$fingerprint_input")
  fi

  "${router_cmd[@]}" >/dev/null
  local rc=$?

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  return "$rc"
}

openclaw_route_blocker() {
  local event_router=""
  local source=""
  local key=""
  local severity="critical"
  local summary=""
  local evidence_ref=""
  local cooldown_sec="1800"
  local fingerprint_input=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --event-router)
        event_router="${2:-}"; shift 2 ;;
      --source)
        source="${2:-}"; shift 2 ;;
      --key)
        key="${2:-}"; shift 2 ;;
      --severity)
        severity="${2:-}"; shift 2 ;;
      --summary)
        summary="${2:-}"; shift 2 ;;
      --evidence-ref)
        evidence_ref="${2:-}"; shift 2 ;;
      --cooldown-sec)
        cooldown_sec="${2:-}"; shift 2 ;;
      --fingerprint-input)
        fingerprint_input="${2:-}"; shift 2 ;;
      *)
        shift ;;
    esac
  done

  if [[ -z "$summary" ]]; then
    summary="unspecified_blocker"
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  openclaw_route_event \
    --event-router "$event_router" \
    --source "$source" \
    --key "$key" \
    --severity "$severity" \
    --summary "$summary" \
    --evidence-ref "$evidence_ref" \
    --cooldown-sec "$cooldown_sec" \
    --fingerprint-input "$fingerprint_input"
  local rc=$?

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  if [[ "$rc" -eq 0 ]]; then
    printf 'BLOCKER: %s\n' "$summary"
  elif [[ "$rc" -eq 20 ]]; then
    :
  elif [[ "$rc" -eq 127 ]]; then
    printf 'BLOCKER: %s\n' "$summary"
  else
    printf 'BLOCKER: %s; router_error=%s\n' "$summary" "$rc"
  fi

  return 0
}

openclaw_watchdog_route_blocker() {
  local key="${1:-}"
  local summary="${2:-}"
  local evidence_ref="${3:-${OPENCLAW_BLOCKER_DEFAULT_EVIDENCE_REF:-}}"
  local severity="${4:-critical}"

  openclaw_route_blocker \
    --event-router "${OPENCLAW_BLOCKER_EVENT_ROUTER:-}" \
    --source "${OPENCLAW_BLOCKER_SOURCE:-}" \
    --key "$key" \
    --severity "$severity" \
    --summary "$summary" \
    --evidence-ref "$evidence_ref" \
    --cooldown-sec "${OPENCLAW_BLOCKER_COOLDOWN_SEC:-1800}"
}

openclaw_blocker_summary_from_line() {
  local line="${1:-}"
  local summary=""

  if [[ "$line" == BLOCKER_JSON:* ]]; then
    local payload="${line#BLOCKER_JSON: }"
    if command -v python3 >/dev/null 2>&1; then
      summary="$(python3 - "$payload" <<'PY' 2>/dev/null || true
import json
import sys

raw = sys.argv[1]
summary = ""
try:
    obj = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)

if isinstance(obj, dict):
    step = str(obj.get("step") or "").strip()
    reason = str(obj.get("reason") or "").strip()
    if step and reason:
      summary = f"step={step}; reason={reason}"
    elif step:
      summary = f"step={step}"
    elif reason:
      summary = reason

    if not summary:
      for key in ("summary", "message", "detail"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
          summary = val.strip()
          break

print(summary)
PY
)"
    fi
  fi

  if [[ -z "$summary" ]]; then
    summary="$line"
    summary="${summary#BLOCKER: }"
    summary="${summary#BLOCKER_JSON: }"
  fi

  summary="$(printf '%s' "$summary" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//')"
  if [[ -z "$summary" ]]; then
    summary="unspecified_blocker"
  fi

  printf '%.240s' "$summary"
}
