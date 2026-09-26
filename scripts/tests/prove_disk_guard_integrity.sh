#!/usr/bin/env bash
# Proof harness for disk-guard-integrity.sh (review B3, 2026-09-21).
#
# CLASS proven: round3_lock_evidence.sh mutated the LIVE launchd-scheduled
# disk-guard-cron.sh in place, with its restore on the happy path only. A
# timeout between mutate and restore left production monitoring neutered
# (`if [ -n "$holder" ] && false;`) while launchd kept executing it every 900s.
# Nothing noticed. That is the same failure class the guard exists to prevent:
# a monitor that is silently not monitoring.
#
# The integrity check must therefore:
#   I1  DERIVE the set of scheduled scripts from the launchd plists (the source
#       of truth), never from a hand-maintained list in the checker;
#   I2  go RED when a scheduled script's content differs from its blessed hash;
#   I3  go RED when a scheduled script has NO blessed hash at all (an undeclared
#       scheduled script is unguarded — P-GUARD: a guard built on a list must
#       fail on items present in the source of truth but absent from the list);
#   I4  go GREEN again once the content is restored, with no re-bless needed.
#
# Run: bash ~/.hermes/scripts/tests/prove_disk_guard_integrity.sh
set -uo pipefail

INTEG="$HOME/.hermes/scripts/disk-guard-integrity.sh"
fail=0

check() { # $1 name  $2 expected  $3 actual
  if [ "$2" = "$3" ]; then
    echo "PASS $1 ($3)"
  else
    echo "FAIL $1: expected '$2' got '$3'"; fail=1
  fi
}

if [ ! -x "$INTEG" ]; then
  echo "FAIL integrity checker does not exist at $INTEG"
  exit 1
fi

# Isolated fixture: a fake LaunchAgents dir with a fake scheduled script, so the
# proof never touches the real production file or the real plists.
FIX=$(mktemp -d -t dgintegrity)
AGENTS="$FIX/LaunchAgents"; mkdir -p "$AGENTS"
SCRIPT="$FIX/scheduled-fixture.sh"
BASELINES="$FIX/baselines"

printf '#!/usr/bin/env bash\necho fixture v1\n' > "$SCRIPT"
cat > "$AGENTS/com.agentlabs.fixture.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.agentlabs.fixture</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string>
    <string>$SCRIPT</string>
  </array>
  <key>StartInterval</key><integer>900</integer>
</dict></plist>
PLIST

export DISK_GUARD_AGENTS_DIR="$AGENTS"
export DISK_GUARD_BASELINES="$BASELINES"
export DISK_GUARD_SCRIPT_ROOT="$FIX"

run() { bash "$INTEG" "$@" >/tmp/dgintegrity.out 2>&1; echo $?; }

# I1 — the checker must DERIVE the scheduled script from the plist. Before any
# bless exists it must name that exact path in its output; a checker that
# hardcodes its targets cannot possibly name a fixture path it never heard of.
rc=$(run --check)
if grep -q "$SCRIPT" /tmp/dgintegrity.out; then
  check "I1 scheduled script is DERIVED from the launchd plist" "derived" "derived"
else
  check "I1 scheduled script is DERIVED from the launchd plist" "derived" "not-derived"
  echo "     ---- checker output ----"; sed 's/^/     /' /tmp/dgintegrity.out
fi

# I3 — unblessed scheduled script must be RED, not silently skipped.
check "I3 undeclared scheduled script is RED (no blessed hash)" "1" "$rc"

# Bless it, then it must be GREEN.
run --bless >/dev/null
check "I0 blessed baseline verifies clean" "0" "$(run --check)"

# I2 — mutate the scheduled script exactly as the lock harness did.
cp "$SCRIPT" "$FIX/orig"
printf '#!/usr/bin/env bash\necho fixture MUTATED\n' > "$SCRIPT"
check "I2 mutated scheduled script is RED" "1" "$(run --check)"
grep -q "$SCRIPT" /tmp/dgintegrity.out \
  && echo "PASS I2b RED names the drifted path" \
  || { echo "FAIL I2b RED does not name the drifted path"; fail=1; }

# I4 — restoring the content must go GREEN with no re-bless.
cp "$FIX/orig" "$SCRIPT"
check "I4 restored scheduled script is GREEN again (no re-bless)" "0" "$(run --check)"

rm -rf "$FIX"
exit "$fail"
