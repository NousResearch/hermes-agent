#!/usr/bin/env bash
# Proof harness for the pinned disk floor.
#
# CLASS proven: on 2026-09-21 the floor moved 10Gi -> 25Gi between two alerts
# because it was a plain env var. A floor is a decision; silent drift in either
# direction (down = neutered guard reporting green, up = manufactured RED)
# must be impossible without an explicit, labelled override.
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_floor.sh
set -uo pipefail

CRON="$HOME/.hermes/profiles/software-engineer/scripts/disk-guard-cron.sh"
fail=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1 ($3)"; else echo "FAIL $1: expected '$2' got '$3'"; fail=1; fi; }

# Read back the floor the wrapper actually resolves, without running a scan:
# source the header up to the GUARD assignment in a subshell.
resolve_floor() { # env already set by caller
  ( set +u
    eval "$(sed -n '1,/^GUARD=/p' "$CRON" | grep -v '^GUARD=')" >/dev/null 2>&1
    echo "$DISK_GUARD_FLOOR_GI" )
}

# The pinned value is DERIVED from the file, not restated here: a harness that
# hardcodes the number goes stale silently the next time the owner moves it.
# DECIMAL-aware since 2026-09-22 (pin 1 -> 0.5): an integer-only pattern does
# not read a 0.5 pin as wrong, it reads it as ABSENT, and this harness then
# exits 1 with "cannot parse" — a parse failure dressed up as a floor failure.
PIN=$(sed -n 's/^FLOOR_GI_PINNED=\([0-9][0-9]*\(\.[0-9][0-9]*\)*\)$/\1/p' "$CRON")
[ -n "$PIN" ] || { echo "FAIL cannot parse FLOOR_GI_PINNED from $CRON"; exit 1; }
echo "pinned floor read from $CRON: ${PIN}Gi"
check "default floor is the pinned value" "$PIN" "$(resolve_floor)"
check "downward override is refused" "$PIN" "$(DISK_GUARD_FLOOR_GI=0 resolve_floor)"
check "unlabelled upward override is refused" "$PIN" "$(DISK_GUARD_FLOOR_GI=25 resolve_floor)"
check "labelled upward override is allowed" "25" \
      "$(DISK_GUARD_FLOOR_GI=25 DISK_GUARD_FLOOR_OVERRIDE_REASON='harness: force RED' resolve_floor)"

# The pin must be justified in the file, not just asserted — a bare number with
# no derivation is how the next person "tunes" it again.
for token in scratch-cost.sh max_in_progress FLOOR_GI_PINNED; do
  grep -q "$token" "$CRON" && echo "PASS derivation mentions $token" \
    || { echo "FAIL derivation missing $token"; fail=1; }
done

exit "$fail"
