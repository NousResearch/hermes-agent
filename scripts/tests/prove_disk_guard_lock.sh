#!/usr/bin/env bash
# Proof harness for disk-guard-cron.sh single-flight locking.
#
# CLASS proven: an age-only stale-lock rule let a tick killed by
# `launchctl kickstart -k` wedge the guard into a silent 60-minute no-op —
# the guard reporting exit 0 while doing nothing. Liveness-derived staleness
# must reclaim that lock on the very next tick.
#
# EVIDENCE CHANNEL (review B2, 2026-09-21): this harness used to count
# "tick rc=" lines in ~/.hermes/logs/disk-guard.log. The code under test
# truncates that log to its last 2000 lines on every tick, so when the evicted
# lines contained an older tick line the count stayed flat and a tick that
# really ran was scored "silent no-op" — a false FAIL indistinguishable from
# the mutation RED. Evidence now comes from DISK_GUARD_RECEIPT, an append-only
# file owned by this harness that the code under test never rotates or writes
# outside the tick path.
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_lock.sh
set -uo pipefail

# Overridable so a mutation harness can run against a COPY under $TMPDIR
# instead of the live launchd-scheduled file (review B3, 2026-09-21: a harness
# that mutated production in place left the guard neutered when it was killed).
CRON="${DISK_GUARD_CRON:-$HOME/.hermes/profiles/software-engineer/scripts/disk-guard-cron.sh}"
# ISOLATED lockdir. Using the live one raced the launchd-scheduled tick
# (StartInterval=900) and produced non-deterministic FAILs — including a failed
# R0 anti-vacuity check, i.e. a real scheduled tick holding the lock while the
# harness's own tick tried to run.
LOCKDIR=$(mktemp -d -t dglock)/lock
export DISK_GUARD_LOCKDIR="$LOCKDIR"
RECEIPT=$(mktemp -t dgproof)
export DISK_GUARD_RECEIPT="$RECEIPT"
export DISK_GUARD_DRY_RUN=1   # never page a human from a proof run
fail=0

check() { # $1 name  $2 expected  $3 actual
  if [ "$2" = "$3" ]; then
    echo "PASS $1 ($3)"
  else
    echo "FAIL $1: expected '$2' got '$3'"; fail=1
  fi
}

# NB: `grep -c` PRINTS "0" and exits 1 on no-match, so `grep -c ... || echo 0`
# emits TWO lines ("0\n0") and every numeric comparison against it breaks.
receipts() { # grep -c prints 0 AND exits 1 on no-match, so `grep -c ... || echo 0`
             # emits TWO lines ("0\n0") and every numeric test against it breaks.
             # wc -l always prints exactly one integer.
  grep "tick rc=" "$RECEIPT" 2>/dev/null | wc -l | tr -d ' '
}

restore=""
rm -rf "$LOCKDIR"

# R0 anti-vacuity: an unobstructed tick must produce exactly one receipt. If the
# receipt channel itself is dead, every later check would read "silent no-op"
# and the harness would pass under mutation for the wrong reason.
: > "$RECEIPT"
DISK_GUARD_INVOKER=proof bash "$CRON" >/dev/null 2>&1
check "R0 receipt channel is live (unobstructed tick records itself)" "1" "$(receipts)"

# 1. A lock held by a DEAD pid must be reclaimed: the tick must actually RUN.
: > "$RECEIPT"
mkdir -p "$LOCKDIR"
sleep 0 & dead=$!; wait "$dead" 2>/dev/null   # a pid guaranteed to be gone
echo "$dead" > "$LOCKDIR/pid"
DISK_GUARD_INVOKER=proof bash "$CRON" >/dev/null 2>&1
if [ "$(receipts)" -ge 1 ]; then
  check "dead-holder lock is reclaimed (tick actually ran)" "ran" "ran"
else
  check "dead-holder lock is reclaimed (tick actually ran)" "ran" "silent no-op"
fi

# 2. A lock held by a LIVE pid must be respected: the tick must NOT run.
# Evidence is again the receipt, not empty stdout — a crashed tick is also silent.
: > "$RECEIPT"
mkdir -p "$LOCKDIR"
sleep 120 & live=$!
echo "$live" > "$LOCKDIR/pid"
out2=$(DISK_GUARD_INVOKER=proof bash "$CRON" 2>&1)
check "live-holder lock is respected (tick did not run)" "0" "$(receipts)"
check "live-holder tick is silent" "" "$out2"
kill "$live" 2>/dev/null; wait "$live" 2>/dev/null
rm -rf "$LOCKDIR"

rm -f "$RECEIPT"; rm -rf "$LOCKDIR"
exit "$fail"
