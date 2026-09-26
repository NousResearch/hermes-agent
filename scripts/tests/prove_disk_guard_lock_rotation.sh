#!/usr/bin/env bash
# Regression proof for review finding B2 (2026-09-21).
#
# CLASS proven: prove_disk_guard_lock.sh used to take its evidence from
#   grep -c "tick rc=" ~/.hermes/logs/disk-guard.log
# before/after a tick. disk-guard-cron.sh truncates that same log with
# `tail -n 2000` on every tick. Once the log is saturated at 2000 lines, a tick
# whose evicted lines contain an older "tick rc=" line leaves the count FLAT
# even though the tick ran — scoring a real run as "silent no-op". That false
# FAIL is byte-identical to the mutation RED for the stale-lock guard, so the
# guard's own mutation proof could not distinguish a caught regression from a
# log-rotation artifact.
#
# This harness DELIBERATELY manufactures that collision and asserts:
#   C1 the collision is real   — the count-based channel goes flat on a real tick
#   C2 the receipt channel is immune — it records the same tick anyway
#
# C1 is the anti-vacuity check: if the collision cannot be staged (log not
# saturated, wrapper stopped rotating), C1 FAILS rather than letting C2 pass for
# a reason that has nothing to do with the fix.
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_lock_rotation.sh
set -uo pipefail

CRON="$HOME/.hermes/profiles/software-engineer/scripts/disk-guard-cron.sh"
LOG="$HOME/.hermes/logs/disk-guard.log"
# Isolated lockdir: the live one is contended by the launchd-scheduled tick
# (StartInterval 900), which was the OTHER source of non-determinism in the lock
# proofs. A proof must not race production.
LOCKDIR=$(mktemp -d -t dgrotlock)/lock
export DISK_GUARD_LOCKDIR="$LOCKDIR"
RECEIPT=$(mktemp -t dgrot)
BACKUP=$(mktemp -t dgrotbak)
export DISK_GUARD_RECEIPT="$RECEIPT"
export DISK_GUARD_DRY_RUN=1   # never page a human from a proof run
fail=0

check() { # $1 name  $2 expected  $3 actual
  if [ "$2" = "$3" ]; then echo "PASS $1 ($3)"; else echo "FAIL $1: expected '$2' got '$3'"; fail=1; fi
}
# grep -c prints "0" and exits 1 on no match; never chain `|| echo 0` onto it.
count_in() { local c; c=$(grep -c "tick rc=" "$1" 2>/dev/null); echo "${c:-0}"; }

cleanup() {
  [ -s "$BACKUP" ] && cp "$BACKUP" "$LOG"
  rm -f "$RECEIPT" "$BACKUP"
}
trap cleanup EXIT

cp "$LOG" "$BACKUP"
rm -rf "$LOCKDIR" 2>/dev/null

# Stage the collision: a saturated 2000-line log whose OLDEST lines (the ones
# `tail -n 2000` will evict) are tick lines. Derived from the wrapper's own
# rotation constant, not from a guessed number.
#
# SATURATION IS STAGED, NOT INHERITED. This harness used to build the fixture as
# `20 tick lines + tail -(KEEP-20) of the live log`, which only reaches KEEP
# lines when the live log is ALREADY saturated. On 2026-09-25 the live log was
# 1754 lines, the fixture came out short, `tail -n KEEP` evicted nothing, the
# count channel rose normally and C1 reported "not-staged" — an anti-vacuity
# check that fails for a reason unrelated to the property under test is just as
# useless as one that passes for the wrong reason. Pad deterministically.
KEEP=$(grep -o 'tail -n [0-9]*' "$CRON" | head -1 | awk '{print $3}')
KEEP=${KEEP:-2000}
{
  for i in $(seq 1 20); do
    printf '%s tick rc=0 invoker=rotationfixture%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$i"
  done
  tail -n "$((KEEP - 20))" "$BACKUP"
  # Pad with non-tick filler until the fixture is exactly KEEP lines, so the
  # wrapper's own rotation is guaranteed to evict the tick lines at the head.
  have=$(wc -l < "$BACKUP" | tr -d ' ')
  [ "$have" -gt "$((KEEP - 20))" ] && have=$((KEEP - 20))
  need=$((KEEP - 20 - have))
  i=0
  while [ "$i" -lt "$need" ]; do
    printf 'rotation fixture filler line %s\n' "$i"
    i=$((i + 1))
  done
} > "$LOG"
staged=$(wc -l < "$LOG" | tr -d ' ')
head1=$(head -1 "$LOG")
echo "staged log lines=$staged (keep=$KEEP) head='${head1:0:60}'"

before_count=$(count_in "$LOG")
: > "$RECEIPT"
DISK_GUARD_INVOKER=rotationproof bash "$CRON" >/dev/null 2>&1
after_count=$(count_in "$LOG")
receipt_count=$(count_in "$RECEIPT")

echo "count-channel: before=$before_count after=$after_count   receipt-channel: $receipt_count"

# C1: the tick definitely ran (receipt proves it), yet the count did not rise.
if [ "$receipt_count" -ge 1 ] && [ "$after_count" -le "$before_count" ]; then
  check "C1 rotation collision staged (count channel goes flat on a real tick)" "collision" "collision"
else
  check "C1 rotation collision staged (count channel goes flat on a real tick)" "collision" "not-staged"
fi

# C2: the receipt channel recorded that same tick regardless.
check "C2 receipt channel survives the rotation that erases the count" "1" "$receipt_count"

exit "$fail"
