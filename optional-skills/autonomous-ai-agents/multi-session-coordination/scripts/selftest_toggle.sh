#!/bin/bash
# Master-switch verification for session_coord.py + coord_guard.sh.
# Proves: OFF is a fail-open no-op (never blocks a caller), ON restores full
# conflict detection byte-for-byte, and the env override beats the sentinel
# file in BOTH directions. Everything runs against scratch state in /tmp.
# macOS bash 3.2 compatible; safe to run anytime.
# GUARD is an intentionally overridable path (CI sets it), and COORD_SC is
# exported for a guard that is sourced INSIDE the same subshell, so shellcheck's
# "modification is local to the subshell" note is a false positive here.
# shellcheck disable=SC1090,SC2030,SC2031
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SC="${SC:-$HERE/session_coord.py}"
GUARD="${GUARD:-$HERE/coord_guard.sh}"
PY=python3

D=$(mktemp -d /tmp/toggle.XXXXXX)
export HERMES_COORD_DB="$D/board.db"
export HERMES_COORD_DISABLED_FILE="$D/coordination_disabled"
export HERMES_COORD_CRON_MANIFEST="$D/manifest.json"
export HERMES_COORD_CRON_JOBS="$D/jobs.json"
# Never let a real switch leak in from the environment running the suite.
unset HERMES_COORD_DISABLED
pass=0; fail=0
ck() { # ck <desc> <expected_rc> <actual_rc> [grep_str] [output]
  local d="$1" want="$2" got="$3" g="${4:-}" out="${5:-}"
  if [ "$want" != "$got" ]; then echo "FAIL: $d (rc want=$want got=$got)"; fail=$((fail+1)); return; fi
  if [ -n "$g" ] && ! grep -q "$g" <<<"$out"; then echo "FAIL: $d (missing '$g' in: $out)"; fail=$((fail+1)); return; fi
  echo "ok:   $d"; pass=$((pass+1))
}

# ================= baseline: ENABLED by default =================
# A holds a resource; a rival must see it as HELD (rc 75).
A=$($PY "$SC" register --task holder --surface test | head -1)
$PY "$SC" claim --id "$A" --res demo --task hold >/dev/null
B=$($PY "$SC" register --task rival --surface test | head -1)
OUT=$($PY "$SC" check --id "$B" --res demo); RC=$?
ck "default ENABLED: rival check -> 75 HELD" 75 $RC "" "$OUT"
OUT=$($PY "$SC" switch); RC=$?
ck "switch reports ENABLED by default" 0 $RC "ENABLED" "$OUT"

# ================= disable -> sentinel written =================
OUT=$($PY "$SC" disable); RC=$?
ck "disable succeeds" 0 $RC "DISABLED" "$OUT"
sent=0; [ -f "$HERMES_COORD_DISABLED_FILE" ] || sent=1
ck "disable wrote the sentinel file" 0 "$sent"
OUT=$($PY "$SC" switch); RC=$?
ck "switch now reports DISABLED" 0 $RC "DISABLED" "$OUT"

# ================= OFF = fail-open no-op on every verb =================
# check must report FREE with rc 0 even though A 'held' demo while ON.
OUT=$($PY "$SC" check --id "$B" --res demo); RC=$?
ck "OFF: check -> 0 FREE (no enforcement)" 0 $RC "FREE" "$OUT"
# register still yields a usable 12-hex id on stdout.
C=$($PY "$SC" register --task x --surface test 2>/dev/null | head -1)
hex=0; [[ $C =~ ^[0-9a-f]{12}$ ]] || hex=1
ck "OFF: register still prints a 12-hex id" 0 "$hex"
# claim never blocks.
$PY "$SC" claim --id "$C" --res demo --task hold >/dev/null 2>&1; RC=$?
ck "OFF: claim -> 0 (never blocks)" 0 $RC
# housekeeping verbs succeed quietly.
$PY "$SC" "done" --id "$C" >/dev/null 2>&1; ck "OFF: done -> 0" 0 $?
$PY "$SC" release --id "$C" --all >/dev/null 2>&1; ck "OFF: release -> 0" 0 $?
$PY "$SC" inbox --id "$C" >/dev/null 2>&1; ck "OFF: inbox -> 0" 0 $?
OUT=$($PY "$SC" status); RC=$?
ck "OFF: status announces DISABLED" 0 $RC "DISABLED" "$OUT"

# ---- the critical fail-open property: cron-guard stdout MUST be empty ----
# (the shell guard reads stdout as the guard id; empty/non-hex => run unguarded)
OUT=$($PY "$SC" cron-guard --job job1 --res demo --policy wait --timeout 2 2>/dev/null); RC=$?
ck "OFF: cron-guard rc 0" 0 $RC
empty=0; [ -z "$OUT" ] || empty=1
ck "OFF: cron-guard stdout is EMPTY (fail-open)" 0 "$empty"

# JSON no-op stays valid JSON and carries the disabled marker.
OUT=$($PY "$SC" check --id "$B" --res demo --json); RC=$?
ck "OFF: --json check carries \"disabled\":true" 0 $RC '"disabled": true' "$OUT"

# ================= enable -> enforcement restored =================
OUT=$($PY "$SC" enable); RC=$?
ck "enable succeeds" 0 $RC "ENABLED" "$OUT"
gone=0; [ -f "$HERMES_COORD_DISABLED_FILE" ] || gone=1
ck "enable removed the sentinel file" 1 "$gone"
OUT=$($PY "$SC" check --id "$B" --res demo); RC=$?
ck "ENABLED again: rival check -> 75 HELD (enforcement restored)" 75 $RC "" "$OUT"

# ================= idempotency + toggle =================
OUT=$($PY "$SC" enable); ck "enable when already ON is idempotent" 0 $? "already in that state" "$OUT"
$PY "$SC" switch toggle >/dev/null; ck "toggle ON->OFF" 0 "$($PY "$SC" switch --json | grep -q '"enabled": false'; echo $?)"
$PY "$SC" switch toggle >/dev/null; ck "toggle OFF->ON" 0 "$($PY "$SC" switch --json | grep -q '"enabled": true'; echo $?)"

# ================= env override beats the sentinel, both ways =================
# Sentinel says OFF, but env=0 forces ON -> enforcement returns.
$PY "$SC" disable >/dev/null
OUT=$(HERMES_COORD_DISABLED=0 $PY "$SC" check --id "$B" --res demo); RC=$?
ck "env=0 forces ON despite sentinel -> 75 HELD" 75 $RC "" "$OUT"
# No sentinel (ON), but env=1 forces OFF -> no-op FREE.
$PY "$SC" enable >/dev/null
OUT=$(HERMES_COORD_DISABLED=1 $PY "$SC" check --id "$B" --res demo); RC=$?
ck "env=1 forces OFF despite no sentinel -> 0 FREE" 0 $RC "FREE" "$OUT"
OUT=$(HERMES_COORD_DISABLED=off $PY "$SC" switch)
ck "env=off is treated as ON" 0 $? "ENABLED" "$OUT"

# ================= shell guard honors the switch (sourced, like a cron wrapper) =================
if [ -f "$GUARD" ]; then
  # A live session holds res:guarded; manifest maps job1 -> res:guarded.
  printf '{"job1":{"resources":["res:guarded"],"policy":"wait","critical":false}}' > "$HERMES_COORD_CRON_MANIFEST"
  H=$($PY "$SC" register --task holder2 --surface test | head -1)
  $PY "$SC" claim --id "$H" --res "res:guarded" --task hold >/dev/null
  $PY "$SC" enable >/dev/null
  ( set -u; export COORD_SC="$SC"; . "$GUARD"; coord_guard job1 wait 2 5; exit $? )
  ck "guard ON: defers (75) while session holds the resource" 75 $?
  $PY "$SC" disable >/dev/null
  ( set -u; export COORD_SC="$SC"; . "$GUARD"; coord_guard job1 wait 2 5; rc=$?; \
    ck_id="$COORD_GUARD_ID"; [ -z "$ck_id" ] && exit $rc || exit 99 )
  ck "guard OFF (sentinel): proceeds unguarded (0), no claim id" 0 $?
  $PY "$SC" enable >/dev/null
  ( set -u; export COORD_SC="$SC"; export HERMES_COORD_DISABLED=1; . "$GUARD"; coord_guard job1 wait 2 5; exit $? )
  ck "guard OFF (env): proceeds unguarded (0)" 0 $?
  ( set -u; export COORD_SC="$SC"; . "$GUARD"; coord_guard job1 wait 2 5; exit $? )
  ck "guard back ON: defers (75) again" 75 $?
  $PY "$SC" release --id "$H" --all >/dev/null 2>&1
else
  echo "skip: coord_guard.sh not found next to session_coord.py ($GUARD)"
fi

rm -rf "$D"
echo "-----------------------------------------------"
echo "toggle suite: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
