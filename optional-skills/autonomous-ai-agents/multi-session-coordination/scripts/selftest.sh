#!/bin/bash
# Live test suite for session_coord.py against a SCRATCH db.
set -u
SC="${SC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_coord.py}"
export HERMES_COORD_DB=/tmp/coord_test_$$.db
PY=python3
pass=0; fail=0
ck() { # ck <desc> <expected_rc> <actual_rc> [grep_str] [output]
  local d="$1" want="$2" got="$3" g="${4:-}" out="${5:-}"
  if [ "$want" != "$got" ]; then echo "FAIL: $d (rc want=$want got=$got)"; fail=$((fail+1)); return; fi
  if [ -n "$g" ] && ! grep -q "$g" <<<"$out"; then echo "FAIL: $d (missing '$g' in: $out)"; fail=$((fail+1)); return; fi
  echo "ok:   $d"; pass=$((pass+1))
}

# --- 1. register two sessions
A=$($PY $SC register --task "memory hygiene sweep" --surface desktop | head -1)
OUT=$($PY $SC register --task "CMMC report tweak" --surface cli)
B=$(head -1 <<<"$OUT")
ck "register B sees A as co-worker" 0 $? "memory hygiene sweep" "$OUT"

# --- 2. A claims memory + skills dir (exclusive)
OUT=$($PY $SC claim --id $A --res memory --res "file:~/.hermes/skills" --task "memory hygiene"); RC=$?
ck "A claims memory+skills" 0 $RC "CLAIMED" "$OUT"

# --- 3. B check -> held, holder info present
OUT=$($PY $SC check --id $B --res memory); RC=$?
ck "B check memory -> 75 + holder task shown" 75 $RC "memory hygiene" "$OUT"

# --- 4. dir-prefix conflict: child path blocked, sibling-with-prefix-name free
OUT=$($PY $SC check --id $B --res "file:~/.hermes/skills/imessage/SKILL.md"); RC=$?
ck "child of claimed dir -> held" 75 $RC "" "$OUT"
OUT=$($PY $SC check --id $B --res "file:~/.hermes/skills-retired"); RC=$?
ck "path-boundary: skills-retired NOT blocked by skills claim" 0 $RC "FREE" "$OUT"

# --- 5. no-wait claim returns 75 fast
OUT=$($PY $SC claim --id $B --res memory --task "wants memory too"); RC=$?
ck "B non-wait claim -> 75 HELD" 75 $RC "co-worker" "$OUT"

# --- 6. B claims with --wait in bg; A releases 3s later; B must acquire + then see notification
( $PY $SC claim --id $B --res memory --task "wants memory too" --wait --timeout 60 > /tmp/bwait_$$.out 2>&1; echo $? > /tmp/bwait_$$.rc ) &
sleep 3
$PY $SC release --id $A --res memory >/dev/null
wait
RC=$(cat /tmp/bwait_$$.rc); OUT=$(cat /tmp/bwait_$$.out)
ck "B wait-claim acquires after A releases" 0 "$RC" "CLAIMED" "$OUT"
OUT=$($PY $SC inbox --id $B)
ck "B inbox has release notification from A" 0 $? "released by co-worker" "$OUT"

# --- 7. A still holds skills; waiters_on visibility: A claims again elsewhere while B waits on skills
( $PY $SC wait --id $B --res "file:~/.hermes/skills" --timeout 20 >/dev/null 2>&1 ) &
sleep 2
OUT=$($PY $SC status)
ck "status shows B waiting on skills" 0 $? "waiting on" "$OUT"
$PY $SC release --id $A --all >/dev/null
wait

# --- 8. RACE: 6 concurrent claims on same key -> exactly 1 winner
for i in 1 2 3 4 5 6; do
  ( S=$($PY $SC register --task "racer $i" --surface test | head -1)
    $PY $SC claim --id $S --res race-key --task "racer $i" >/dev/null 2>&1
    echo "$i:$?" >> /tmp/race_$$.out ) &
done
wait
WINNERS=$(grep -c ':0$' /tmp/race_$$.out)
ck "race: exactly 1 of 6 concurrent claims wins" 1 "$WINNERS" "" ""
LOSERS=$(grep -c ':75$' /tmp/race_$$.out)
ck "race: 5 losers get rc 75" 5 "$LOSERS" "" ""

# --- 9. shared mode: two shared coexist; exclusive blocked by shared
S1=$($PY $SC register --task s1 --surface test | head -1)
S2=$($PY $SC register --task s2 --surface test | head -1)
$PY $SC claim --id $S1 --res "box:gpu1" --mode shared --task "reading logs" >/dev/null
OUT=$($PY $SC claim --id $S2 --res "box:gpu1" --mode shared --task "also reading"); RC=$?
ck "two shared claims coexist" 0 $RC "CLAIMED" "$OUT"
OUT=$($PY $SC claim --id $B --res "box:gpu1" --task "mutating"); RC=$?
ck "exclusive blocked by shared holders" 75 $RC "" "$OUT"

# --- 10. TTL expiry: claim ttl ~2.4s; waiter gets EXPIRED warning
S3=$($PY $SC register --task "doomed" --surface test | head -1)
$PY $SC claim --id $S3 --res doomed-res --ttl 0.04 --task "will die" >/dev/null
$PY $SC wait --id $B --res doomed-res --timeout 1 >/dev/null 2>&1  # registers B as waiter
sleep 3
OUT=$($PY $SC check --id $B --res doomed-res); RC=$?
ck "expired claim no longer blocks" 0 $RC "FREE" "$OUT"
OUT=$($PY $SC inbox --id $B)
ck "waiter warned about EXPIRED holder" 0 $? "EXPIRED" "$OUT"

# --- 11. done: releases all + notifies waiters + status clean
$PY $SC claim --id $S1 --res final-res --mode exclusive --task "final task" >/dev/null
( $PY $SC wait --id $S2 --res final-res --timeout 30 >/dev/null 2>&1 ) &
sleep 2
OUT=$($PY $SC done --id $S1)
ck "done releases everything" 0 $? "DONE" "$OUT"
wait
OUT=$($PY $SC inbox --id $S2)
ck "done notified the waiter" 0 $? "finished cleanly" "$OUT"

# --- 12. steal with reason -> victim notified
$PY $SC claim --id $S2 --res steal-res --task "holding" >/dev/null
OUT=$($PY $SC steal --id $B --res steal-res --reason "holder session confirmed dead by user"); RC=$?
ck "steal force-releases" 0 $RC "FORCE-RELEASED" "$OUT"
OUT=$($PY $SC inbox --id $S2)
ck "victim notified of steal + reason" 0 $? "confirmed dead" "$OUT"

# --- 13. id resolution: 8-char prefix works; no-match & ambiguous fail loudly (v2.3.1)
P1=$($PY $SC register --task "prefix target" --surface test | head -1)
$PY $SC claim --id $P1 --res "res:prefixfix" --task "holding for prefix test" >/dev/null
SHORT=${P1:0:8}
OUT=$($PY $SC release --id "$SHORT" --all); RC=$?
ck "8-char prefix release actually releases" 0 $RC "RELEASED" "$OUT"
OUT=$($PY $SC check --id $P1 --res "res:prefixfix"); RC=$?
ck "prefix release really freed the resource (not a fake success)" 0 $RC "FREE" "$OUT"
OUT=$($PY $SC "done" --id "zzzznope" 2>&1); RC=$?
ck "no-match id errors loudly (rc 2)" 2 $RC "no session matches" "$OUT"
OUT=$($PY $SC "done" --id "$SHORT"); RC=$?
ck "done via 8-char prefix deregisters the real session" 0 $RC "DONE" "$OUT"

# --- 14. explicit ids (v2.3.2): register --id, idempotent re-register,
#         claim-auto-register orphan-proofing, invalid id rejection
OUT=$($PY $SC register --id explicit-id-test-0826 --task "explicit id task" --surface test); RC=$?
ck "register --id uses the caller's id" 0 $RC "explicit-id-test-0826" "$OUT"
$PY $SC claim --id explicit-id-test-0826 --res "res:explid" --task "hold" >/dev/null
OUT=$($PY $SC release --id explicit-id-test-0826 --all); RC=$?
ck "release resolves the explicit id (orphan bug fixed)" 0 $RC "RELEASED: res:explid" "$OUT"
OUT=$($PY $SC register --id explicit-id-test-0826 --task "re-registered task"); RC=$?
ck "re-register same explicit id is idempotent" 0 $RC "explicit-id-test-0826" "$OUT"
OUT=$($PY $SC register --id "bad id!" --task "nope" 2>&1); RC=$?
ck "invalid explicit id rejected rc 2" 2 $RC "invalid" "$OUT"
OUT=$($PY $SC claim --id never-registered-xyz --res "res:orphanproof" --task "orphan probe" >/dev/null; $PY $SC release --id never-registered-xyz --all); RC=$?
ck "claim auto-registers unknown id -> releasable" 0 $RC "RELEASED: res:orphanproof" "$OUT"
$PY $SC "done" --id explicit-id-test-0826 >/dev/null 2>&1
$PY $SC "done" --id never-registered-xyz >/dev/null 2>&1

echo; echo "RESULT: $pass passed, $fail failed"
rm -f /tmp/bwait_$$.out /tmp/bwait_$$.rc /tmp/race_$$.out "$HERMES_COORD_DB"* 2>/dev/null || true
exit $fail
