#!/usr/bin/env bash
# prove_disk_guard_dead_task_worktrees.sh
#
# Regression proof for the dead-card worktree class, 2026-09-25 (t_2a74e97a).
#
# MEASURED: ~/.hermes/hermes-agent-wt-t_bf5b8389-b2 held 554MB for a card in
# status `done`. Two scoping bugs hid it from the existing sweep:
#   1. repo discovery was pinned to $HOME/workspace/*/.git, so no repo outside
#      that one directory was scanned — including the agent's own checkout;
#   2. reclaim_merged_worktrees gates on `merge-base --is-ancestor <tip> main`,
#      which a squash-merging repo NEVER satisfies, so it removed 0 worktrees
#      over its entire log while reporting success.
#
# The new class keys on card terminality, not on branch topology. The property
# that must never regress is P4: a worktree whose branch is NOT merged is still
# reclaimed when its card is done — that is precisely what the ancestor gate
# got wrong, and a test that only fed it merged branches would pass on the
# broken implementation too.
#
# Asserted:
#   P1 a done card's worktree is reclaimed, in a repo OUTSIDE ~/workspace
#   P2 a running card's worktree is KEPT
#   P3 a worktree with no card id in its name is KEPT (not ours to judge)
#   P4 an UNMERGED branch whose card is done IS reclaimed (the squash-merge bug)
#   P5 a live-held worktree is KEPT
#
# Runs in a throwaway $HOME with throwaway git repos. Run:
#   bash ~/.hermes/scripts/tests/prove_disk_guard_dead_task_worktrees.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"
command -v sqlite3 >/dev/null 2>&1 || { echo "SKIP: sqlite3 unavailable"; exit 0; }
command -v git    >/dev/null 2>&1 || { echo "SKIP: git unavailable"; exit 0; }

FIX=$(mktemp -d -t dgwt)
export GIT_CONFIG_GLOBAL="$FIX/gitconfig"; : >"$GIT_CONFIG_GLOBAL"
git config --global user.email t@t; git config --global user.name t
git config --global init.defaultBranch main

mkdir -p "$FIX/.hermes"
sqlite3 "$FIX/.hermes/kanban.db" "create table tasks(id text, status text);
  insert into tasks values('t_aaaaaaaa','done'),('t_bbbbbbbb','running'),
                           ('t_cccccccc','done'),('t_dddddddd','done');"

# Repo lives under ~/.hermes — i.e. OUTSIDE ~/workspace, which the old
# discovery would never have scanned.
REPO="$FIX/.hermes/agentrepo"
git init -q "$REPO"; ( cd "$REPO" && echo x >f && git add f && git commit -qm init )

mkwt() { git -C "$REPO" worktree add -q -b "$2" "$1" >/dev/null 2>&1; }
mkwt "$FIX/.hermes/wt-t_aaaaaaaa" b-done
mkwt "$FIX/.hermes/wt-t_bbbbbbbb" b-running
mkwt "$FIX/.hermes/wt-nocard"     b-nocard
mkwt "$FIX/.hermes/wt-t_cccccccc" b-unmerged
mkwt "$FIX/.hermes/wt-t_dddddddd" b-busy
# P4: give the unmerged worktree a commit so its tip is NOT an ancestor of main.
( cd "$FIX/.hermes/wt-t_cccccccc" && echo y >g && git add g && git commit -qm diverge )

perl -e 'sleep 300' "$FIX/.hermes/wt-t_dddddddd" &
INUSE_PID=$!
sleep 1

HOME="$FIX" HERMES_KANBAN_DB="$FIX/.hermes/kanban.db" bash -c '
    set -uo pipefail
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    reclaim_dead_task_worktrees
  ' >"$FIX/sweep.log" 2>&1

grep -q 'dead-card worktrees:' "$FIX/sweep.log" \
  && ok "the real reclaim_dead_task_worktrees actually executed" \
  || { bad "sweep did not run (library seam broken) — the keeps below are vacuous"; cat "$FIX/sweep.log"; }

[ -d "$FIX/.hermes/wt-t_aaaaaaaa" ] \
  && bad "TOO NARROW: done-card worktree outside ~/workspace survived (the 554MB leak)" \
  || ok "reclaims a done card's worktree in a repo outside ~/workspace"

[ -d "$FIX/.hermes/wt-t_cccccccc" ] \
  && bad "TOO NARROW: UNMERGED branch of a done card survived — the squash-merge ancestor bug is back" \
  || ok "reclaims a done card's worktree even when its branch never merged"

[ -d "$FIX/.hermes/wt-t_bbbbbbbb" ] \
  && ok "keeps a RUNNING card's worktree" \
  || bad "TOO BROAD: deleted a running card's worktree"

[ -d "$FIX/.hermes/wt-nocard" ] \
  && ok "keeps a worktree with no card id in its name" \
  || bad "TOO BROAD: deleted a non-card worktree (a human branch)"

if kill -0 "$INUSE_PID" 2>/dev/null && [ -d "$FIX/.hermes/wt-t_dddddddd" ]; then
  ok "keeps a worktree held open by a live process"
else
  bad "TOO BROAD: deleted a worktree with a live process inside it"
fi

kill "$INUSE_PID" 2>/dev/null
wait "$INUSE_PID" 2>/dev/null
rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
