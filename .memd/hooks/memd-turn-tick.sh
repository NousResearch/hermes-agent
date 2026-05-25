#!/usr/bin/env bash
# memd preference-drift turn tick (F4.7 driver).
#
# Fires once per UserPromptSubmit, calling `memd preference tick` so the
# per-turn counter advances. When the counter rolls over `n_turns` (default
# 10), the CLI verb appends a `tick_fire` row to
# `.memd/logs/preference-drift.ndjson`. That NDJSON is the dogfood signal
# the V4 close deviation record forwarded to V5.
#
# Master gate: `MEMD_F4_PREF_DRIFT` must be on (1/true/on/yes). Otherwise
# this is a no-op and the CLI verb itself short-circuits.
#
# Failure mode: the tick is fire-and-forget; any error logs to stderr but
# never blocks the harness turn. The hook always exits 0 so a missing
# `memd` binary does not stall a session.
set -u

# Honor master gate cheaply before invoking memd at all.
case "${MEMD_F4_PREF_DRIFT:-}" in
  1|true|on|yes|TRUE|ON|YES) ;;
  *) exit 0 ;;
esac

# Discover bundle. Walk up from CWD if stdin carries a JSON harness payload.
INPUT="$(cat 2>/dev/null || true)"
CWD=""
SESSION_ID=""
if [ -n "$INPUT" ]; then
  parsed="$(printf '%s' "$INPUT" | python3 -c '
import json, sys
try:
    data = json.loads(sys.stdin.read() or "{}")
    print(data.get("cwd", ""))
    print(data.get("session_id", ""))
except Exception:
    print("")
    print("")
' 2>/dev/null || printf '\n\n')"
  CWD="$(printf '%s' "$parsed" | sed -n '1p')"
  SESSION_ID="$(printf '%s' "$parsed" | sed -n '2p')"
fi
[ -z "$CWD" ] && CWD="$(pwd)"

find_bundle() {
  local dir="$1"
  while [ "$dir" != "/" ] && [ -n "$dir" ]; do
    if [ -f "$dir/.memd/config.json" ]; then
      printf '%s' "$dir/.memd"
      return
    fi
    dir="$(dirname "$dir")"
  done
}

BUNDLE_ROOT="$(find_bundle "$CWD")"
[ -z "$BUNDLE_ROOT" ] && exit 0

args=(preference tick --output "$BUNDLE_ROOT")
[ -n "$SESSION_ID" ] && args+=(--session-id "$SESSION_ID")

if command -v memd >/dev/null 2>&1; then
  memd "${args[@]}" >/dev/null 2>&1 || true
fi

exit 0
