#!/usr/bin/env bash
# prove_disk_guard_workspace_roots.sh
#
# Regression proof for the multi-board workspace scoping fix, 2026-09-25
# (card t_2a74e97a).
#
# MEASURED: reclaim_workspaces swept only $HOME/.hermes/kanban/workspaces —
# the DEFAULT board. Every named board keeps its own
# kanban/boards/<slug>/workspaces tree with its own kanban.db, and none were
# ever reclaimed: boards/vibebrowser held 1.8GB of dead-card scratch while the
# guard logged "workspaces: removed 0 dead scratch dirs". Another "green while
# leaking" shape.
#
# The subtle danger is the db pairing. A card id only has a status in the board
# that OWNS it; dead_task treats an unknown id as dead, so resolving a board
# card against the default db would report every one of them dead and delete a
# RUNNING card's workspace. That property is asserted here explicitly.
#
# Asserted:
#   P1 a dead card's workspace on a NAMED board is reclaimed  (not too narrow)
#   P2 the default board still works                          (no regression)
#   P3 a RUNNING card on a named board is KEPT — i.e. the per-board db is
#      consulted, not the default one (this fails loudly if the pairing breaks)
#   P4 a live-held workspace is kept
#
# Runs entirely inside a throwaway $HOME. Run:
#   bash ~/.hermes/scripts/tests/prove_disk_guard_workspace_roots.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"

command -v sqlite3 >/dev/null 2>&1 || { echo "SKIP: sqlite3 unavailable"; exit 0; }

FIX=$(mktemp -d -t dgws)
DEF="$FIX/.hermes/kanban/workspaces"
BRD="$FIX/.hermes/kanban/boards/vibebrowser/workspaces"
mkdir -p "$DEF" "$BRD"

mkdb() { sqlite3 "$1" "create table tasks(id text, status text);" ; }
mkdb "$FIX/.hermes/kanban.db"
mkdb "$FIX/.hermes/kanban/boards/vibebrowser/kanban.db"

# Default board: one done card.
sqlite3 "$FIX/.hermes/kanban.db" "insert into tasks values('t_defdone','done');"
mkdir -p "$DEF/t_defdone"

# Named board: one done card, one RUNNING card, one live-held done card.
sqlite3 "$FIX/.hermes/kanban/boards/vibebrowser/kanban.db" \
  "insert into tasks values('t_brddone','done'),('t_brdrun','running'),('t_brdbusy','done');"
mkdir -p "$BRD/t_brddone" "$BRD/t_brdrun" "$BRD/t_brdbusy"

perl -e 'sleep 300' "$BRD/t_brdbusy" &
INUSE_PID=$!
sleep 1

HOME="$FIX" HERMES_KANBAN_DB="$FIX/.hermes/kanban.db" bash -c '
    set -uo pipefail
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    reclaim_workspaces
  ' >"$FIX/sweep.log" 2>&1

grep -q 'workspaces:' "$FIX/sweep.log" \
  && ok "the real reclaim_workspaces actually executed" \
  || bad "sweep did not run (library seam broken) — the keeps below are vacuous"

[ -d "$BRD/t_brddone" ] \
  && bad "TOO NARROW: a named board's dead-card workspace survived (the 1.8GB leak)" \
  || ok "reclaims a dead card's workspace on a NAMED board"

[ -d "$DEF/t_defdone" ] \
  && bad "REGRESSION: the default board's dead-card workspace survived" \
  || ok "still reclaims the default board"

[ -d "$BRD/t_brdrun" ] \
  && ok "keeps a RUNNING card on a named board (per-board db consulted)" \
  || bad "TOO BROAD: deleted a RUNNING card's workspace — the board card id was resolved against the WRONG db"

if kill -0 "$INUSE_PID" 2>/dev/null && [ -d "$BRD/t_brdbusy" ]; then
  ok "keeps a workspace held open by a live process"
else
  bad "TOO BROAD: deleted a workspace with a live process inside it"
fi

kill "$INUSE_PID" 2>/dev/null
wait "$INUSE_PID" 2>/dev/null
rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
