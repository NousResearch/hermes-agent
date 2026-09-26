#!/usr/bin/env bash
# prove_disk_guard_path_in_use.sh
#
# Regression proof for the SELF-MATCH bug found 2026-09-25 (card t_2a74e97a).
#
# path_in_use() gates every deletion in disk-guard.sh and the top_reclaimable
# ranking. It was implemented as:
#     ps -Ao args= | grep -Fq -- "$1"
# `grep`'s own argv contains "$1", ps lists it, and the grep matches ITSELF. So
# path_in_use returned 0 ("in use") for every path in existence. Consequences,
# all observed on this host:
#   - every reclaim class ran, logged "removed 0", and deleted nothing;
#   - the operator page said "No single reclaimable path over 100MB" while the
#     volume filled up, because the ranking filtered out all candidates.
# A guard that can never delete and can never rank is a guard that reports green
# while the host dies -- the exact failure this script exists to prevent.
#
# U1 a path no process references is NOT in use  (the bug: this returned true)
# U2 a path in a live process's argv IS in use
# U3 a live process cwd'd into the dir with a bare argv IS in use
# U4 the probe does not match its own machinery at any depth
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_path_in_use.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

FIX=$(mktemp -d -t dgpiu)
mkdir -p "$FIX/idle" "$FIX/argv" "$FIX/cwd"

probe() { # $1 = path -> prints "in-use" / "free"
  # The path goes through the ENVIRONMENT, never this helper's argv. Embedding
  # it in `bash -c` would put the needle in the probe's own command line, and
  # the harness would then measure its own machinery instead of the guard --
  # reproducing the very self-match bug under test and failing U1 for a reason
  # that has nothing to do with disk-guard.sh.
  DG_PROBE_PATH="$1" DISK_GUARD_LIB=1 bash -c '
    source "'"$GUARD"'"
    if path_in_use "$DG_PROBE_PATH"; then echo in-use; else echo free; fi
  ' 2>/dev/null | tail -1
}

check() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2' got '$3'"; }

# U1 -- the bug. Nothing references $FIX/idle.
check "U1 an unreferenced path is free" "free" "$(probe "$FIX/idle")"

# U2 -- the path appears in a live process's argv (and is NOT opened, so this
# exercises the ps/argv channel specifically, not the lsof fallback).
# `sleep 300 --dummy-arg <path>` does not work: macOS sleep rejects the extra
# operand and exits immediately, so the fixture process was already dead and U2
# measured nothing.
# `exec sleep 300` would REPLACE the argv and drop the path with it, so the
# fixture must stay in a shell whose own command line still carries it.
bash -c 'while :; do sleep 5; done' _ "$FIX/argv" >/dev/null 2>&1 &
ARGV_PID=$!
sleep 1
kill -0 "$ARGV_PID" 2>/dev/null \
  || bad "U2 fixture process died — the argv channel was not exercised"
check "U2 a path in a live argv is in-use" "in-use" "$(probe "$FIX/argv")"
kill "$ARGV_PID" 2>/dev/null; wait "$ARGV_PID" 2>/dev/null

# U3 -- bare argv, but the process cwd is inside the dir (lsof +D catches it).
( cd "$FIX/cwd" && exec sleep 300 ) &
CWD_PID=$!
sleep 1
check "U3 a live cwd inside the dir is in-use" "in-use" "$(probe "$FIX/cwd")"
kill "$CWD_PID" 2>/dev/null; wait "$CWD_PID" 2>/dev/null

# U4 -- after the helpers are gone, the same path must read free again. If the
# probe matched its own pipeline this stays "in-use" forever.
check "U4 the probe does not match its own machinery" "free" "$(probe "$FIX/cwd")"

# U5 -- SPELLING EQUIVALENCE. The caller's path and the live process's argv
# routinely name the same file differently: on macOS /var is a symlink to
# /private/var, `git worktree list` prints the resolved form, and a worker was
# launched with the unresolved one. A probe that compares only the string it
# was handed reports "free" for a directory a live process is sitting in, and
# the guard then deletes a running worker's tree. Measured 2026-09-25 against
# the real reclaim_dead_task_worktrees, which did exactly that.
mkdir -p "$FIX/spell"
bash -c 'while :; do sleep 5; done' _ "$FIX/spell" >/dev/null 2>&1 &
SPELL_PID=$!
sleep 1
kill -0 "$SPELL_PID" 2>/dev/null \
  || bad "U5 fixture process died — the spelling channel was not exercised"
# The OTHER spelling of the very same directory.
SPELL_ALT=$(cd "$FIX/spell" && pwd -P)
case "$SPELL_ALT" in
  "$FIX/spell") SPELL_ALT="" ;;   # no symlinked prefix on this host
esac
if [ -n "$SPELL_ALT" ]; then
  check "U5 a differently-spelled path to a live dir is in-use" \
        "in-use" "$(probe "$SPELL_ALT")"
else
  ok "U5 skipped: \$TMPDIR has no symlinked prefix on this host"
fi
kill "$SPELL_PID" 2>/dev/null; wait "$SPELL_PID" 2>/dev/null

rm -rf "$FIX"
[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
