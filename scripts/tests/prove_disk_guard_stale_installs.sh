#!/usr/bin/env bash
# prove_disk_guard_stale_installs.sh
#
# Regression proof for the install-generation class added 2026-09-25
# (card t_2a74e97a). MEASURED: 2.9GB sat in three ~/.hermes/installs
# generations while exactly one was live, and no reclaim class touched it —
# reclaim_safe_caches enumerates a fixed list, and ~/.hermes is a dotdir so the
# $HOME node_modules sweep skips it.
#
# The dangerous failure mode here is TOO BROAD: deleting the live generation
# removes the Hermes runtime out from under every agent on the box (which is
# how this card's first eight attempts died, from a broken dependency env).
# So liveness must be DETECTED, never pinned to a hash allowlist — an allowlist
# rots on the next reinstall and eventually names the live generation garbage.
#
# Asserted here:
#   P1 the sweep exists at all
#   P2 a superseded generation IS evicted          (not too narrow)
#   P3 the NEWEST generation is kept               (fresh spawns resolve to it)
#   P4 a generation held open by a live pid is kept, EVEN IF NOT NEWEST
#      (this is the "never delete the running runtime" property, and the one an
#      age- or allowlist-based rule gets wrong)
#
# Everything happens inside a throwaway $HOME. Run:
#   bash ~/.hermes/scripts/tests/prove_disk_guard_stale_installs.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

grep -q 'reclaim_stale_installs' "$GUARD" \
  && ok "guard contains the install-generation sweep" \
  || bad "guard has no install-generation sweep — the 2026-09-25 class fix is gone"

bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"

# An allowlist of generation hashes is the failure mode this class must not
# have: it cannot survive a reinstall.
grep -qE '[0-9a-f]{16}' "$GUARD" \
  && bad "guard hardcodes a generation hash — liveness must be detected, not pinned" \
  || ok "no hardcoded generation hash (liveness is detected)"

FIX=$(mktemp -d -t dginstalls)
ROOT="$FIX/.hermes/installs"
mkdir -p "$ROOT/gen-old" "$ROOT/gen-busy" "$ROOT/gen-newest"
# Make the ordering unambiguous regardless of mktemp timing.
touch -t 202401010000 "$ROOT/gen-old"
touch -t 202402010000 "$ROOT/gen-busy"
touch -t 202403010000 "$ROOT/gen-newest"

# P4 fixture: a live process whose argv names the NOT-newest generation.
# (perl takes the path as an ignored ARGV entry, so the path is genuinely in
# the process's argv — which is exactly what path_in_use scans.)
perl -e 'sleep 300' "$ROOT/gen-busy" &
INUSE_PID=$!
sleep 1

HOME="$FIX" bash -c '
    set -uo pipefail
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    reclaim_stale_installs
  ' >"$FIX/sweep.log" 2>&1

grep -q 'installs:' "$FIX/sweep.log" \
  && ok "the real reclaim_stale_installs actually executed" \
  || bad "sweep did not run (library seam broken) — the keeps below are vacuous"

[ -d "$ROOT/gen-old" ] \
  && bad "TOO NARROW: superseded gen-old survived the sweep" \
  || ok "evicts a superseded generation"

[ -d "$ROOT/gen-newest" ] \
  && ok "keeps the newest generation" \
  || bad "TOO BROAD: deleted the newest generation (fresh spawns would break)"

if kill -0 "$INUSE_PID" 2>/dev/null && [ -d "$ROOT/gen-busy" ]; then
  ok "keeps a live-held generation even though it is not the newest"
else
  bad "TOO BROAD: deleted the RUNNING runtime — this breaks every agent on the host"
fi

kill "$INUSE_PID" 2>/dev/null
wait "$INUSE_PID" 2>/dev/null
rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
