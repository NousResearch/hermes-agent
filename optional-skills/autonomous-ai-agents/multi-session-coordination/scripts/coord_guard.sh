#!/usr/bin/env bash
# ===========================================================================
# EDIT HISTORY (Hermes agent changes; newest first)
#   2026-08-16 | model: claude-fable-5 | provider: anthropic | settings: temp=default, reasoning=default | change: Created — shared session-coordination guard for cron wrapper scripts (multi-session coordination board, skill multi-session-coordination). Sourced by nightly_*_backup.sh wrappers as step 0.
# ===========================================================================
# Shared cron-side guard for the multi-session coordination board.
#
#   source "$HOME/.hermes/scripts/coord_guard.sh"
#   coord_guard <job-id> <policy skip|wait> <timeout-s> <ttl-min>
#   case $? in 75) exit 0;; esac   # deferred: skip this tick silently
#
# Semantics:
#   rc 0  -> either resources CLAIMED on the board (COORD_GUARD_ID set; an
#            EXIT trap releasing them is installed — if your script sets its
#            OWN `trap ... EXIT` later, include `coord_done` in it), or the
#            guard is unavailable/broken (COORD_GUARD_ID empty) and the job
#            proceeds UNGUARDED (fail-open: coordination protects work, it
#            never blocks the backup itself).
#   rc 75 -> a live session holds this job's declared resources: SKIP this
#            tick silently (holders were notified on the board; the job fires
#            again on its own schedule).
#   If the wrapper dies hard (kill -9), the claim expires via its TTL and
#   waiters get the dead-holder warning.
#
# stdout hygiene: everything goes to stderr; nothing is printed to stdout —
# safe for no_agent jobs where stdout is delivered verbatim as an iMessage.

COORD_SC="${COORD_SC:-$HOME/.hermes/scripts/session_coord.py}"
COORD_GUARD_ID=""

# Master switch, honored WITHOUT spawning Python (zero-cost when OFF). Must
# match coordination_enabled() in session_coord.py exactly: env
# HERMES_COORD_DISABLED (truthy=off, 0/false/no/off=on) overrides the sentinel
# file; absent both -> enabled.
COORD_DISABLED_FILE="${HERMES_COORD_DISABLED_FILE:-$HOME/.hermes/state/coordination_disabled}"

_coord_disabled() {
  # rc 0 = coordination is OFF; rc 1 = ON.
  local e
  if [ -n "${HERMES_COORD_DISABLED:-}" ]; then
    e=$(printf '%s' "$HERMES_COORD_DISABLED" | tr '[:upper:]' '[:lower:]')
    case "$e" in
      0|false|no|off) return 1 ;;   # explicitly ON
      *)              return 0 ;;   # any other non-empty value -> OFF
    esac
  fi
  [ -f "$COORD_DISABLED_FILE" ]      # sentinel present -> OFF
}

coord_done() {
  [ -n "${COORD_GUARD_ID:-}" ] || return 0
  python3 "$COORD_SC" "done" --id "$COORD_GUARD_ID" >/dev/null 2>&1 || true
  COORD_GUARD_ID=""
}

coord_guard() {
  local job="$1" policy="${2:-wait}" timeout="${3:-900}" ttl="${4:-90}" rc=0 out=""
  if _coord_disabled; then COORD_GUARD_ID=""; return 0; fi  # master switch OFF -> run unguarded
  [ -f "$COORD_SC" ] || { COORD_GUARD_ID=""; return 0; }   # no board -> fail open
  # stdout = guard session id (only); stderr = human chatter (let it flow to
  # the caller's stderr/log for forensics).
  out=$(python3 "$COORD_SC" cron-guard --job "$job" \
        --policy "$policy" --timeout "$timeout" --ttl "$ttl") ; rc=$?
  if [ $rc -eq 75 ]; then
    COORD_GUARD_ID=""
    return 75                       # caller: skip tick (exit 0, silent)
  fi
  if [ $rc -ne 0 ] || ! printf '%s' "$out" | grep -qE '^[0-9a-f]{12}$'; then
    COORD_GUARD_ID=""               # guard broken/odd output -> fail open
    return 0
  fi
  COORD_GUARD_ID="$out"
  trap coord_done EXIT
  return 0
}
