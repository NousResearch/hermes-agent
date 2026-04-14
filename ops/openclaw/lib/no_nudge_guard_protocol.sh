#!/usr/bin/env bash
# shellcheck shell=bash

# Validate no-nudge cron guard runtime execution + first-line protocol.
# Returns exactly one first line:
# - READY: ...
# - BLOCKER: ... (fail-closed on guard execution/protocol issues)

OPENCLAW_PROTOCOL_ACCEPT_LIB="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}/ops/openclaw/lib/protocol_accept_list.sh"
if [[ -f "$OPENCLAW_PROTOCOL_ACCEPT_LIB" ]]; then
  # shellcheck source=ops/openclaw/lib/protocol_accept_list.sh
  source "$OPENCLAW_PROTOCOL_ACCEPT_LIB"
fi

openclaw_no_nudge_guard_first_line() {
  local guard_script="${1:-}"
  local err_path="${2:-/tmp/openclaw_no_nudge_guard.err}"
  shift 2 || true
  local guard_args=("$@")

  if ! declare -F openclaw_protocol_first_non_empty_line >/dev/null 2>&1 \
    || ! declare -F openclaw_protocol_sanitize_inline >/dev/null 2>&1 \
    || ! declare -F openclaw_protocol_line_is_guard_ready_or_blocker >/dev/null 2>&1; then
    echo "BLOCKER: no_nudge_cron_guard_invalid_protocol; reason=protocol_accept_helper_missing"
    return 0
  fi

  if [[ -z "$guard_script" ]]; then
    echo "BLOCKER: no_nudge_cron_guard_exec_failed; reason=guard_script_missing"
    return 0
  fi

  if [[ ! -x "$guard_script" ]]; then
    echo "BLOCKER: no_nudge_cron_guard_exec_failed; reason=guard_script_not_executable; path=$guard_script"
    return 0
  fi

  local had_errexit=0
  if [[ "$-" == *e* ]]; then
    had_errexit=1
    set +e
  fi

  local guard_stdout=""
  guard_stdout="$($guard_script "${guard_args[@]}" 2>"$err_path")"
  local guard_rc=$?

  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  local first_line=""
  first_line="$(openclaw_protocol_first_non_empty_line "$guard_stdout")"

  if [[ "$guard_rc" -ne 0 ]]; then
    local err=""
    err="$(cat "$err_path" 2>/dev/null || true)"
    err="$(openclaw_protocol_sanitize_inline "$err")"
    echo "BLOCKER: no_nudge_cron_guard_exec_failed; rc=$guard_rc; err=${err:0:180}"
    return 0
  fi

  if [[ -z "$first_line" ]]; then
    echo "BLOCKER: no_nudge_cron_guard_invalid_protocol; reason=empty_first_line"
    return 0
  fi

  if ! openclaw_protocol_line_is_guard_ready_or_blocker "$first_line"; then
    local sanitized_first_line=""
    sanitized_first_line="$(openclaw_protocol_sanitize_inline "$first_line")"
    echo "BLOCKER: no_nudge_cron_guard_invalid_protocol; reason=unexpected_first_line; first_line=${sanitized_first_line:0:180}"
    return 0
  fi

  printf '%s\n' "$first_line"
  return 0
}
