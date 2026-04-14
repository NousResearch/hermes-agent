#!/usr/bin/env bash
# shellcheck shell=bash

# Shared first-line protocol accept-list helpers for cron adapters and guard wrappers.
# Keep these deterministic and fail-closed; callers decide fallback/route behavior.

openclaw_protocol_first_non_empty_line() {
  printf '%s\n' "${1:-}" | awk 'NF{print; exit}'
}

openclaw_protocol_sanitize_inline() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

openclaw_protocol_line_is_blocker() {
  local line="${1:-}"
  [[ "$line" == BLOCKER:* || "$line" == BLOCKER_JSON:* ]]
}

openclaw_protocol_line_is_guard_ready_or_blocker() {
  local line="${1:-}"
  [[ "$line" == READY:* || "$line" == BLOCKER:* ]]
}

openclaw_protocol_line_is_cron_quiet_success() {
  local line="${1:-}"
  case "$line" in
    READY:*|PROGRESS:*|NO_REPLY|INTERNAL_STATUS:*|INFO:*|WARN:*)
      return 0
      ;;
  esac
  return 1
}
