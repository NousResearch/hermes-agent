#!/usr/bin/env bash
# prove_disk_guard_multi_board_liveness.sh
#
# Regression proof for multi-board card liveness, 2026-09-25 (card t_2a74e97a).
#
# MEASURED, on a real --reclaim run of this guard: reclaim_dead_task_worktrees
# deleted the worktrees of t_41e448bd and t_fb0730ea. Both cards were in status
# `blocked` -- non-terminal, unfinished human-blocked work -- but they live on
# the `vibebrowser` and `xsense-telegram-worker` boards, and dead_task queried
# only the DEFAULT ~/.hermes/kanban.db. Absent from that one file, they read as
# "unknown", and unknown was treated as dead.
#
# The class of bug is treating absence from ONE store as evidence of death.
# The fix asks every board and lets any non-terminal status veto. The property
# below that actually discriminates is M2: a card that is ALIVE ON A NON-DEFAULT
# BOARD AND ABSENT FROM THE DEFAULT ONE must be kept. A test that only ever
# seeds the default board passes on the broken implementation too.
#
# Asserted:
#   M1 a card done on the default board is dead
#   M2 a card alive ONLY on a non-default board is ALIVE  (the 2026-09-25 bug)
#   M3 a card done on every board that knows it is dead
#   M4 a card alive on the default board is alive
#   M5 a card no board has ever heard of is dead (steady-state scratch)
#   M6 with NO reachable DB at all, nothing is dead (fail safe, not fail open)
#
# Runs against a throwaway $HOME. Run:
#   bash ~/.hermes/scripts/tests/prove_disk_guard_multi_board_liveness.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"
command -v sqlite3 >/dev/null 2>&1 || { echo "SKIP: sqlite3 unavailable"; exit 0; }

FIX=$(mktemp -d -t dgmb)
mkdir -p "$FIX/.hermes/kanban/boards/vibebrowser" \
         "$FIX/.hermes/kanban/boards/xsense-telegram-worker"

mk() { sqlite3 "$1" "create table if not exists tasks(id text, status text);
                     insert into tasks values('$2','$3');"; }

# Default board knows m_done and m_running.
mk "$FIX/.hermes/kanban.db" t_11111111 done
mk "$FIX/.hermes/kanban.db" t_22222222 running
# It also thinks it knows t_33333333 -- as done.
mk "$FIX/.hermes/kanban.db" t_33333333 done
# A non-default board has the SAME card still open. Any veto must win.
mk "$FIX/.hermes/kanban/boards/vibebrowser/kanban.db" t_33333333 blocked
# And a card the default board has never heard of at all (the real t_fb0730ea).
mk "$FIX/.hermes/kanban/boards/xsense-telegram-worker/kanban.db" t_44444444 blocked

# Ask the guard's own dead_task, via the library seam.
verdict() {
  # HERMES_KANBAN_DB must be cleared: it pins KANBAN_DB to whatever board the
  # ambient session happens to own, so a leaked value makes the fixture $HOME
  # irrelevant and the suite measures the caller's real board instead.
  env -u HERMES_KANBAN_DB HOME="$1" bash -c '
      set -uo pipefail
      DISK_GUARD_LIB=1 source "'"$GUARD"'"
      if dead_task "'"$2"'"; then echo dead; else echo alive; fi
    ' 2>/dev/null
}
check() { # $1 label  $2 want  $3 got
  [ "$3" = "$2" ] && ok "$1 ($3)" || bad "$1 -- wanted $2, got ${3:-<nothing>}"
}

# Guard against a vacuous suite: if the seam is broken every verdict is empty
# and every check would "fail" for the wrong reason, so prove one known answer.
V=$(verdict "$FIX" t_11111111)
[ -n "$V" ] || bad "library seam broken -- dead_task produced no verdict at all"

check "M1 done on the default board is dead"            dead  "$V"
check "M2 alive ONLY on a non-default board is ALIVE"   alive "$(verdict "$FIX" t_44444444)"
check "M3 done on default but BLOCKED elsewhere is ALIVE" alive "$(verdict "$FIX" t_33333333)"
check "M4 running on the default board is alive"        alive "$(verdict "$FIX" t_22222222)"
check "M5 unknown to every board is dead"               dead  "$(verdict "$FIX" t_99999999)"

# M6 -- no DB anywhere. Losing sight of the boards must not authorise deletion.
EMPTY=$(mktemp -d -t dgmbE)
check "M6 with no reachable DB nothing is dead"         alive "$(verdict "$EMPTY" t_11111111)"
rm -rf "$EMPTY"

rm -rf "$FIX"
[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
