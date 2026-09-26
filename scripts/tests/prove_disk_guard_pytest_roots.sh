#!/usr/bin/env bash
# prove_disk_guard_pytest_roots.sh
#
# Regression proof for the 2026-09-25 regrowth class (card t_2a74e97a).
#
# MEASURED: the host fell 25Gi -> 16Gi in 25 minutes with no sanctioned scratch
# root growing. The consumer was $TMPDIR/pytest-of-<user>/pytest-N: each agent
# pytest run that builds a throwaway hermes-home provisions its own ~1.8GB
# Chromium there, pytest keeps the last 3 numbered roots regardless of size, and
# nothing reaped them. reclaim_tmp only looked at /private/tmp; the node_modules
# sweep only looked at git checkouts. So the guard could report OK while the
# host lost GB/minute -- the exact "green while dying" shape this guard exists
# to prevent.
#
# Asserted here:
#   P1 the sweep exists at all (the class fix cannot silently disappear)
#   P2 an idle pytest root IS evicted            (not too narrow)
#   P3 pytest-current is KEPT                    (not too broad: running test)
#   P4 a root held open by a live pid is KEPT    (not too broad: liveness)
#   P5 the selection is scoped to the pytest base only, never $TMPDIR at large
#
# Everything happens inside a private $TMPDIR fixture; the real temp root is
# never touched. Run: bash ~/.hermes/scripts/tests/prove_disk_guard_pytest_roots.sh
set -uo pipefail

GUARD="${DISK_GUARD:-$HOME/.hermes/scripts/disk-guard.sh}"
FAIL=0
ok()  { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; FAIL=1; }

grep -q 'reclaim_pytest_roots' "$GUARD" \
  && ok "guard contains the pytest-root sweep" \
  || bad "guard has no pytest-root sweep — the 2026-09-25 class fix is gone"

bash -n "$GUARD" && ok "guard parses" || bad "guard has a syntax error"

FIX=$(mktemp -d -t dgpytest)
BASE="$FIX/pytest-of-$(id -un)"
mkdir -p "$BASE"/pytest-1 "$BASE"/pytest-2 "$BASE"/pytest-3 "$FIX/unrelated-scratch"
ln -s "$BASE/pytest-3" "$BASE/pytest-current"

# P4 fixture: a live process holding pytest-2 open.
( cd "$BASE/pytest-2" && exec sleep 300 ) &
INUSE_PID=$!
sleep 1

# Drive the guard's REAL function, with reclaim enabled but every other class
# neutralised, so this harness cannot drift from the script it protects.
TMPDIR="$FIX" bash -c '
    set -uo pipefail
    DISK_GUARD_LIB=1 source "'"$GUARD"'"
    reclaim_pytest_roots
  ' >"$FIX/sweep.log" 2>&1
# Anti-vacuity: if the seam stopped working the sweep never ran, and every
# "kept" assertion below would pass for the wrong reason.
grep -q 'pytest roots:' "$FIX/sweep.log" \
  && ok "the real reclaim_pytest_roots actually executed" \
  || bad "sweep did not run (library seam broken) — the keeps below are vacuous"

[ -d "$BASE/pytest-1" ] \
  && bad "TOO NARROW: idle pytest-1 survived the sweep" \
  || ok "evicts an idle pytest root"

[ -d "$BASE/pytest-3" ] \
  && ok "keeps pytest-current's target" \
  || bad "TOO BROAD: deleted the in-progress run (pytest-current)"

if kill -0 "$INUSE_PID" 2>/dev/null && [ -d "$BASE/pytest-2" ]; then
  ok "keeps a root held open by a live process"
else
  bad "TOO BROAD: deleted a root with a live process inside it"
fi

[ -d "$FIX/unrelated-scratch" ] \
  && ok "scoped to the pytest base, leaves the rest of \$TMPDIR alone" \
  || bad "TOO BROAD: touched \$TMPDIR outside pytest-of-<user>"

kill "$INUSE_PID" 2>/dev/null
wait "$INUSE_PID" 2>/dev/null
rm -rf "$FIX"

[ "$FAIL" = 0 ] && { echo "=== RESULT: PASS ==="; exit 0; }
echo "=== RESULT: FAIL ==="; exit 1
