#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
BLOCKER_ROUTING_LIB="$ROOT/ops/openclaw/lib/blocker_routing.sh"
PROTOCOL_ACCEPT_LIB="$ROOT/ops/openclaw/lib/protocol_accept_list.sh"
EVENT_ROUTER_DEFAULT="$ROOT/ops/openclaw/event_router.sh"
TASK=""
STDERR_MAX=180
BLOCKER_SOURCE_PREFIX="${OPENCLAW_CRON_PROTOCOL_BLOCKER_SOURCE_PREFIX:-watchdog.cron}"
BLOCKER_COOLDOWN_SEC="${OPENCLAW_CRON_PROTOCOL_BLOCKER_COOLDOWN_SEC:-1800}"
BLOCKER_EVIDENCE_REF_DEFAULT="${OPENCLAW_CRON_PROTOCOL_BLOCKER_EVIDENCE_REF:-}"

usage() {
  cat <<'EOF'
Usage: cron_protocol_outcome.sh --task <task_name> -- <command> [args...]

Deterministic watchdog adapter for cron agent-turn rails.
- Executes command and inspects first non-empty stdout line.
- Emits exactly one line on stdout:
  - BLOCKER: ...   (only when command reports blocker or fails)
  - NO_REPLY       (all non-blocker outcomes)
- Routes blocker side-effects through event_router when available so authority
  decisions do not depend on model-wrapper forwarding.

Options:
  --task <name>        Logical task name used in fallback blocker summary (required)
  --stderr-max <n>     Max stderr chars included in fallback blocker (default: 180)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --stderr-max)
      STDERR_MAX="${2:-180}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK" ]]; then
  echo "--task is required" >&2
  exit 2
fi
if [[ "$#" -eq 0 ]]; then
  echo "command is required after --" >&2
  exit 2
fi
if ! [[ "$STDERR_MAX" =~ ^[0-9]+$ ]]; then
  echo "--stderr-max must be an integer" >&2
  exit 2
fi

if [[ -f "$BLOCKER_ROUTING_LIB" ]]; then
  # shellcheck source=ops/openclaw/lib/blocker_routing.sh
  source "$BLOCKER_ROUTING_LIB"
fi

if [[ ! -f "$PROTOCOL_ACCEPT_LIB" ]]; then
  printf 'BLOCKER: task=%s; reason=watchdog_protocol_helper_missing; path=%s\n' "$TASK" "$PROTOCOL_ACCEPT_LIB"
  exit 0
fi
# shellcheck source=ops/openclaw/lib/protocol_accept_list.sh
source "$PROTOCOL_ACCEPT_LIB"

sanitize_blocker_summary() {
  local line="${1:-}"
  local summary=""

  if declare -F openclaw_blocker_summary_from_line >/dev/null 2>&1; then
    summary="$(openclaw_blocker_summary_from_line "$line")"
  fi

  if [[ -z "$summary" ]]; then
    summary="$line"
    summary="${summary#BLOCKER: }"
    summary="${summary#BLOCKER_JSON: }"
    summary="$(openclaw_protocol_sanitize_inline "$summary")"
  fi

  if [[ -z "$summary" ]]; then
    summary="task=$TASK; reason=unknown_blocker"
  fi

  printf '%.240s' "$summary"
}

route_blocker_side_effect() {
  local key="${1:-task_blocker}"
  local summary="${2:-}"
  local evidence_ref="${3:-}"

  if ! declare -F openclaw_route_event >/dev/null 2>&1; then
    return 0
  fi

  local source="${BLOCKER_SOURCE_PREFIX}.${TASK}"
  local router="$EVENT_ROUTER_DEFAULT"
  if [[ ! -x "$router" ]]; then
    return 0
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  openclaw_route_event \
    --event-router "$router" \
    --source "$source" \
    --key "$key" \
    --severity "critical" \
    --summary "$summary" \
    --evidence-ref "$evidence_ref" \
    --cooldown-sec "$BLOCKER_COOLDOWN_SEC" >/dev/null
  local _rc=$?

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  return 0
}

emit_blocker() {
  local line="${1:-}"
  local key="${2:-task_blocker}"
  local evidence_ref="${3:-}"
  local summary
  summary="$(sanitize_blocker_summary "$line")"
  route_blocker_side_effect "$key" "$summary" "$evidence_ref"
  printf 'BLOCKER: %s\n' "$summary"
}

err_file="/tmp/cron_protocol_outcome_${TASK}.err"
cmd=("$@")
set +e
cmd_out="$("${cmd[@]}" 2>"$err_file")"
cmd_rc=$?
set -e

first_line="$(openclaw_protocol_first_non_empty_line "$cmd_out")"

if [[ "$cmd_rc" -ne 0 ]]; then
  if openclaw_protocol_line_is_blocker "$first_line"; then
    emit_blocker "$first_line" "task_blocker" "$err_file"
    exit 0
  fi

  err_raw="$(cat "$err_file" 2>/dev/null || true)"
  err="$(openclaw_protocol_sanitize_inline "$err_raw")"
  if [[ -z "$err" ]]; then
    err="$(openclaw_protocol_sanitize_inline "$first_line")"
  fi
  if [[ -z "$err" ]]; then
    err="no_stderr"
  fi

  emit_blocker "BLOCKER: task=$TASK; reason=watchdog_exec_failed; rc=$cmd_rc; err=${err:0:STDERR_MAX}" "watchdog_exec_failed" "$err_file"
  exit 0
fi

if openclaw_protocol_line_is_blocker "$first_line"; then
  emit_blocker "$first_line" "task_blocker" "$err_file"
  exit 0
fi

if [[ -z "$first_line" ]]; then
  printf 'NO_REPLY\n'
  exit 0
fi

if openclaw_protocol_line_is_cron_quiet_success "$first_line"; then
  printf 'NO_REPLY\n'
  exit 0
fi

first_line_sanitized="$(openclaw_protocol_sanitize_inline "$first_line")"
emit_blocker "BLOCKER: task=$TASK; reason=watchdog_protocol_invalid_first_line; first_line=${first_line_sanitized:0:STDERR_MAX}" "protocol_invalid_first_line" "$err_file"
exit 0
