#!/bin/bash
# coord_run.sh — run a command under a session-coord claim with a GUARANTEED `done`.
#
# Guarantees: on ANY exit (normal, error, SIGINT/SIGTERM/SIGHUP, or a failed
# command) the session's claims are released and it is deregistered, so a
# one-shot / subagent / cron runner can never leave a stale claim blocking
# co-workers. This is the hard "done" complement to the board's auto-reap
# safety net (which reaps a holder only after it goes silent).
#
# Usage:
#   coord_run.sh --task "<task>" [--id <id>] [--surface <s>] [--ttl <min>] \
#                [--wait] [--timeout <s>] \
#                --res <key> [--res <key> ...] -- <cmd> [args...]
#
#   --task      REQUIRED: what the run is doing (board label).
#   --id        OPTIONAL: reuse an existing/memorized session id; else auto-register.
#   --surface   OPTIONAL: board surface label (default "cli").
#   --ttl       OPTIONAL: claim lease minutes (default 90; use 240+ for multi-hour work).
#   --wait      OPTIONAL: if the resource is held, poll politely until free/--timeout.
#               DEFAULT: fail fast with rc 75 if held (cron wrappers can `|| exit 0`).
#   --timeout   With --wait: seconds to poll (default 300).
#   --res       REQUIRED (repeatable): resource keys to claim, e.g. --res file:~/x --res skill:foo.
#   --          Separates the coord args from the command to run.
#
# Exit codes: the command's own exit code (released either way) · 75 = held and
# no --wait (not run) · 2 = usage/coord error.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SC="${HERMES_COORD_CLI:-$SCRIPT_DIR/session_coord.py}"
PY=python3
TASK=""; ID=""; SURFACE="cli"; TTL=""; WAIT=0; TIMEOUT=300; RES=()

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)     TASK="$2";       shift 2;;
    --id)       ID="$2";         shift 2;;
    --surface)  SURFACE="$2";    shift 2;;
    --ttl)      TTL="$2";        shift 2;;
    --wait)     WAIT=1;          shift;;
    --timeout)  TIMEOUT="$2";    shift 2;;
    --res)      RES+=("$2");     shift 2;;
    --)         shift; break;;
    *) echo "coord_run.sh: unexpected arg: $1" >&2; usage; exit 2;;
  esac
done

[[ -n "$TASK" ]] || { echo "coord_run.sh: --task is required" >&2; exit 2; }
[[ ${#RES[@]} -gt 0 ]] || {
  echo "coord_run.sh: at least one --res is required" >&2; usage; exit 2
}
[[ $# -gt 0 ]] || { echo "coord_run.sh: no command after --" >&2; usage; exit 2; }

# 1) (re)register; capture ONLY the clean session id (the shell-capture guard:
#    register's informational co-worker lines are '#' comments; `head -1` keeps
#    just the id so it can never become a malformed '#   e388'-style session).
if [[ -z "$ID" ]]; then
  ID="$("$PY" "$SC" register --task "$TASK" --surface "$SURFACE" | head -1)"
fi
export HERMES_COORD_ID="$ID"

# guaranteed release on ANY exit path, including a claim that fails (below).
# A `done` on a session with no acquired claims is harmless (just deregisters
# the auto-registered row); it can never leave a stale claim behind.
release() { "$PY" "$SC" done --id "$ID" >/dev/null 2>&1 || true; }
trap release EXIT INT TERM HUP

# 2) claim the full set in ONE atomic call (never acquire piecemeal).
claim=("$SC" claim --id "$ID" --task "$TASK")
for r in "${RES[@]}"; do claim+=("--res" "$r"); done
[[ -n "$TTL" ]] && claim+=("--ttl" "$TTL")
if [[ "$WAIT" -eq 1 ]]; then claim+=(--wait --timeout "$TIMEOUT"); fi
"$PY" "${claim[@]}" >/dev/null 2>&1 || {
  rc=$?
  echo "coord_run.sh: claim rc $rc (held? use --wait or treat 75 as 'skip') for:" >&2
  printf '   %s\n' "${RES[@]}" >&2
  exit "$rc"   # 75 = held without --wait; 1 = coord/DB error
}

# 3) run the command; preserve its exit code (trap still releases on failure).
"$@"
