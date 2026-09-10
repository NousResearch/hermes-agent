#!/bin/bash
# selftest_v2_priority.sh — Layer 2 verification of session_coord.py v2
# (priority/preempt/pause/resume/lineage/fencing/migration features).
# Safe: runs entirely against scratch DBs in /tmp. macOS bash 3.2 compatible.

SC="${SC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_coord.py}"
DB="/tmp/coordv2_$$.db"
V1DB="/tmp/v1_mig_$$.db"
LOG="/tmp/coordv2_log_$$.txt"
export HERMES_COORD_DB="$DB"
rm -f "$DB" "$DB"-wal "$DB"-shm "$V1DB" "$LOG"
: > "$LOG"

co() { python3 "$SC" "$@"; }

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "ok:   $1"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL: $1"; echo "      -> $2"; }
log() { printf '%s\n' "$1" >> "$LOG"; }

first_line() { printf '%s\n' "$1" | head -1; }

# ---------------------------------------------------------------- setup
OUT=$(co register --task "task A (unranked)" 2>&1); log "$OUT"
A=$(first_line "$OUT"); A8=${A:0:8}
OUT=$(co claim --id "$A" --res res:R --task "A work" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "setup: A claims res:R" "rc=$RC out=$OUT"
OUT=$(co register --task "task B" 2>&1); log "$OUT"; B=$(first_line "$OUT"); B8=${B:0:8}
OUT=$(co register --task "task C" 2>&1); log "$OUT"; C=$(first_line "$OUT"); C8=${C:0:8}

# ---------------------------------------------------------------- check 1
OUT=$(co prioritize --order "${B8}=1,${C8}=2" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ]; then ok "1a. prioritize --order B=1,C=2 rc 0"
else bad "1a. prioritize --order B=1,C=2 rc 0" "rc=$RC out=$OUT"; fi

OUT=$(co claim --id "$C" --res res:R --task "C wants R" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 75 ] && printf '%s' "$OUT" | grep -q "HELD"; then
  ok "1b. C claim res:R (A holds) -> rc 75 HELD"
else bad "1b. C claim res:R -> rc 75 HELD" "rc=$RC out=$OUT"; fi

OUT=$(co preempt --id "$B" --res res:R --reason "user says B first" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ]; then ok "1c. B (rank 1) preempt res:R -> rc 0"
else bad "1c. B preempt res:R -> rc 0" "rc=$RC out=$OUT"; fi

OUT=$(co inbox --id "$A" 2>&1); RC=$?; log "$OUT"
if printf '%s' "$OUT" | grep -q "USER-PRIORITY REQUEST"; then
  ok "1d. A inbox contains USER-PRIORITY REQUEST"
else bad "1d. A inbox contains USER-PRIORITY REQUEST" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 2
OUT=$(co register --task "task D (unranked)" 2>&1); log "$OUT"; D=$(first_line "$OUT")
OUT=$(co preempt --id "$D" --res res:R --reason "no rank" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 75 ] && printf '%s' "$OUT" | grep -q "REFUSED"; then
  ok "2.  unranked D preempt -> rc 75 REFUSED"
else bad "2.  unranked D preempt -> rc 75 REFUSED" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 3
OUT=$(co preempt --id "$B" --res res:R --reason "again 1" 2>&1); log "$OUT"
OUT=$(co preempt --id "$B" --res res:R --reason "again 2" 2>&1); log "$OUT"
OUT=$(co inbox --id "$A" 2>&1); log "$OUT"
N=$(printf '%s\n' "$OUT" | grep -c "USER-PRIORITY REQUEST")
if [ "$N" -eq 1 ]; then ok "3.  preempt dedupe: exactly ONE preempt_request in A inbox"
else bad "3.  preempt dedupe: exactly ONE preempt_request" "count=$N out=$OUT"; fi

# ---------------------------------------------------------------- check 4
OUT=$(co pause --id "$A" --note "ck-note-xyz" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "4a-pre. A pause rc 0" "rc=$RC out=$OUT"
OUT=$(co claim --id "$B" --res res:R --task "B work" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ]; then ok "4a. after A pause, B claim res:R -> rc 0"
else bad "4a. after A pause, B claim res:R -> rc 0" "rc=$RC out=$OUT"; fi
OUT=$(co claim --id "$C" --res res:R --task "C try" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 75 ]; then ok "4b. C claim res:R while B holds -> rc 75"
else bad "4b. C claim res:R while B holds -> rc 75" "rc=$RC out=$OUT"; fi
OUT=$(co status 2>&1); RC=$?; log "$OUT"
if printf '%s' "$OUT" | grep -qi "paused" && printf '%s' "$OUT" | grep -q "ck-note-xyz"; then
  ok "4c. status shows paused A + checkpoint note ck-note-xyz"
else bad "4c. status shows paused + ck-note-xyz" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 5
OUT=$(co done --id "$B" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "5-pre. B done rc 0" "rc=$RC out=$OUT"

OUT=$(co register --task "helper H" 2>&1); log "$OUT"; H=$(first_line "$OUT")
OUT=$(co claim --id "$H" --res res:Q --task "H work" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "5-pre. H claims res:Q" "rc=$RC out=$OUT"
OUT=$(co register --task "task E" 2>&1); log "$OUT"; E=$(first_line "$OUT"); E8=${E:0:8}
OUT=$(co prioritize --session "$E8" --rank 1 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "5-pre. prioritize E=1" "rc=$RC out=$OUT"

EOUT="/tmp/coordv2_e_$$.txt"; COUT="/tmp/coordv2_c_$$.txt"
co claim --id "$E" --res res:Q --task "E wait" --wait --timeout 60 >"$EOUT" 2>&1 &
EPID=$!
sleep 2
co claim --id "$C" --res res:Q --task "C wait" --wait --timeout 60 >"$COUT" 2>&1 &
CPID=$!
sleep 1
OUT=$(co release --id "$H" --res res:Q 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "5-pre. H release res:Q" "rc=$RC out=$OUT"
sleep 6
OUT=$(co status 2>&1); log "$OUT"
if printf '%s' "$OUT" | grep -q "res:Q  <- ${E8}"; then
  ok "5a. after H release, holder of res:Q is E (rank 1)"
else bad "5a. holder of res:Q is E" "status=$OUT"; fi
CN=$(grep -c "CLAIMED" "$COUT")
if [ "$CN" -eq 0 ]; then ok "5b. C (rank 2) still waiting while E holds (fenced)"
else bad "5b. C still waiting while E holds" "c_out=$(cat "$COUT")"; fi
OUT=$(co done --id "$E" 2>&1); log "$OUT"
wait $CPID; CRC=$?
log "$(cat "$COUT")"
if [ $CRC -eq 0 ] && grep -q "CLAIMED" "$COUT"; then
  ok "5c. after E done, C's wait-claim completes rc 0 CLAIMED"
else bad "5c. C wait-claim completes rc 0 CLAIMED" "rc=$CRC c_out=$(cat "$COUT")"; fi
wait $EPID 2>/dev/null
log "$(cat "$EOUT")"

# ---------------------------------------------------------------- check 6
OUT=$(co register --task "helper H2" 2>&1); log "$OUT"; H2=$(first_line "$OUT")
OUT=$(co claim --id "$H2" --res res:S --task "H2 work" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "6-pre. H2 claims res:S" "rc=$RC out=$OUT"
OUT=$(co register --task "task F" 2>&1); log "$OUT"; F=$(first_line "$OUT"); F8=${F:0:8}
OUT=$(co register --task "task G" 2>&1); log "$OUT"; G=$(first_line "$OUT"); G8=${G:0:8}
OUT=$(co prioritize --order "${F8}=3,${G8}=3" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "6-pre. prioritize F=3,G=3" "rc=$RC out=$OUT"

FOUT="/tmp/coordv2_f_$$.txt"; GOUT="/tmp/coordv2_g_$$.txt"
co claim --id "$F" --res res:S --task "F wait" --wait --timeout 60 >"$FOUT" 2>&1 &
FPID=$!
sleep 1
co claim --id "$G" --res res:S --task "G wait" --wait --timeout 60 >"$GOUT" 2>&1 &
GPID=$!
sleep 1
OUT=$(co release --id "$H2" --res res:S 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "6-pre. H2 release res:S" "rc=$RC out=$OUT"
sleep 6
OUT=$(co status 2>&1); log "$OUT"
GN=$(grep -c "CLAIMED" "$GOUT")
if printf '%s' "$OUT" | grep -q "res:S  <- ${F8}" && [ "$GN" -eq 0 ]; then
  ok "6a. FIFO tie-break: F (earlier waiter, equal rank) acquired; G still waiting"
else bad "6a. FIFO tie-break F before G" "status=$OUT g_out=$(cat "$GOUT")"; fi
OUT=$(co done --id "$F" 2>&1); log "$OUT"
wait $GPID; GRC=$?
log "$(cat "$GOUT")"
if [ $GRC -eq 0 ] && grep -q "CLAIMED" "$GOUT"; then
  ok "6b. after F done, G completes rc 0 CLAIMED"
else bad "6b. G completes rc 0 CLAIMED" "rc=$GRC g_out=$(cat "$GOUT")"; fi
wait $FPID 2>/dev/null
log "$(cat "$FOUT")"

# ---------------------------------------------------------------- check 7
# res:R is free (B done in check 5 released it); A still paused with resume spot.
OUT=$(co claim --id "$C" --res res:R --task "C ranked beats paused unranked A" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ]; then ok "7a. C (rank 2) claims free res:R over A's unranked paused spot -> rc 0"
else bad "7a. C claims res:R over paused A -> rc 0" "rc=$RC out=$OUT"; fi
OUT=$(co done --id "$C" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "7-pre. C done rc 0" "rc=$RC out=$OUT"
OUT=$(co resume --id "$A" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ] && printf '%s' "$OUT" | grep -q "RESUMED" \
   && printf '%s' "$OUT" | grep -q "ck-note-xyz"; then
  ok "7b. A resume -> rc 0, RESUMED + checkpoint note ck-note-xyz"
else bad "7b. A resume -> rc 0 RESUMED + ck-note-xyz" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 8
OUT=$(co register --task "task P" 2>&1); log "$OUT"; P=$(first_line "$OUT")
OUT=$(co claim --id "$P" --res res:T --ttl 0.04 --task "P short" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "8-pre. P claims res:T ttl 0.04" "rc=$RC out=$OUT"
OUT=$(co pause --id "$P" --note "x" --ttl 0.04 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "8-pre. P pause ttl 0.04" "rc=$RC out=$OUT"
sleep 4
OUT=$(co status 2>&1); log "$OUT"   # triggers reap
OUT=$(co inbox --id "$P" 2>&1); RC=$?; log "$OUT"
if printf '%s' "$OUT" | grep -qi "expired\|lapsed"; then
  ok "8a. P inbox has expired/lapsed notification for paused-claim TTL"
else bad "8a. P inbox expired/lapsed" "rc=$RC out=$OUT"; fi
OUT=$(co resume --id "$P" 2>&1); RC=$?; log "$OUT"
if printf '%s' "$OUT" | grep -q "nothing paused"; then
  ok "8b. P resume after expiry -> 'nothing paused'"
else bad "8b. P resume -> 'nothing paused'" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 9
OUT=$(co register --task "parent" --rank 1 2>&1); log "$OUT"; PAR=$(first_line "$OUT"); PAR8=${PAR:0:8}
OUT=$(co register --task "kid1" --parent "$PAR" --slot a 2>&1); log "$OUT"; K1=$(first_line "$OUT"); K18=${K1:0:8}
OUT=$(co register --task "kid2" --parent "$PAR" --slot b 2>&1); log "$OUT"; K2=$(first_line "$OUT"); K28=${K2:0:8}
OUT=$(co claim --id "$PAR" --res res:U --task "parent work" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "9-pre. PARENT claims res:U" "rc=$RC out=$OUT"
OUT=$(co claim --id "$K2" --res res:U --task "K2 try" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 75 ]; then ok "9a. K2 claim res:U while PARENT holds -> rc 75 (parent beats child)"
else bad "9a. K2 claim while PARENT holds -> rc 75" "rc=$RC out=$OUT"; fi

K1OUT="/tmp/coordv2_k1_$$.txt"; K2OUT="/tmp/coordv2_k2_$$.txt"
co claim --id "$K2" --res res:U --task "K2 wait" --wait --timeout 90 >"$K2OUT" 2>&1 &
K2PID=$!
sleep 1
co claim --id "$K1" --res res:U --task "K1 wait" --wait --timeout 90 >"$K1OUT" 2>&1 &
K1PID=$!
sleep 1
OUT=$(co release --id "$PAR" --res res:U 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "9-pre. PARENT release res:U" "rc=$RC out=$OUT"
sleep 6
OUT=$(co status 2>&1); log "$OUT"
K2N=$(grep -c "CLAIMED" "$K2OUT")
if printf '%s' "$OUT" | grep -q "res:U  <- ${K18}" && [ "$K2N" -eq 0 ]; then
  ok "9b. slot rank beats FIFO: K1 (slot a) acquired res:U though K2 waited longer"
else bad "9b. K1 (slot a) beats K2 (slot b, earlier waiter)" "status=$OUT k2_out=$(cat "$K2OUT")"; fi
wait $K1PID 2>/dev/null
log "$(cat "$K1OUT")"

# ---------------------------------------------------------------- check 10
OUT=$(co register --task "stranger" 2>&1); log "$OUT"; ST=$(first_line "$OUT"); ST8=${ST:0:8}
OUT=$(co prioritize --order "${ST8}=1,${PAR8}=2" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "10-pre. prioritize STRANGER=1,PARENT=2" "rc=$RC out=$OUT"
JOUT=$(co status --json 2>/dev/null); log "$JOUT"
K1RANK=$(printf '%s' "$JOUT" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for s in d["active_sessions"]:
    if s["session"]=="'"$K18"'":
        print(s["rank"]); break
')
if [ "$K1RANK" = "2a" ]; then ok "10a. K1 effective rank re-computed live to 2a"
else bad "10a. K1 rank == 2a after PARENT re-rank" "got rank='$K1RANK'"; fi
OUT=$(co inbox --id "$K1" 2>&1); RC=$?; log "$OUT"
if printf '%s' "$OUT" | grep -q "PRIORITY UPDATE"; then
  ok "10b. K1 got PRIORITY UPDATE notification"
else bad "10b. K1 inbox has PRIORITY UPDATE" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- check 12 (independent holder so it can't cascade from 9b)
OUT=$(co register --task "holder2" 2>&1); log "$OUT"; HOLD2=$(first_line "$OUT")
OUT=$(co claim --id "$HOLD2" --res res:V --task "holder2 work" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "12-pre. HOLD2 claims res:V" "rc=$RC out=$OUT"
OUT=$(co preempt --id "$ST" --res res:V --reason "user wants stranger first" 2>&1); RC=$?; log "$OUT"
[ $RC -eq 0 ] || bad "12-pre. STRANGER (rank 1) preempt res:V rc 0" "rc=$RC out=$OUT"
ERR=$(co status --id "$HOLD2" 2>&1 >/dev/null); log "$ERR"
if printf '%s' "$ERR" | grep -q "urgent notification"; then
  ok "12. holder status --id shows urgent-alert on STDERR"
else bad "12. urgent notification on stderr" "stderr='$ERR'"; fi
kill $K2PID 2>/dev/null; wait $K2PID 2>/dev/null
log "$(cat "$K2OUT")"

# ---------------------------------------------------------------- check 11 (v1-schema migration)
# Fixture built via python3 sqlite3 module — the sqlite3 CLI does not exist
# on all CI runners (e.g. GitHub windows-latest).
python3 - "$V1DB" <<'PYSQL'
import sqlite3, sys, time
con = sqlite3.connect(sys.argv[1])
con.executescript("""
CREATE TABLE sessions(
    id TEXT PRIMARY KEY, task TEXT, surface TEXT,
    started_at REAL, last_seen REAL, status TEXT DEFAULT 'active');
CREATE TABLE claims(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, resource TEXT,
    mode TEXT DEFAULT 'exclusive', task TEXT, claimed_at REAL, ttl_min REAL,
    released_at REAL, status TEXT DEFAULT 'held');
CREATE TABLE waiters(
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, resource TEXT,
    since REAL, note TEXT, active INTEGER DEFAULT 1);
CREATE TABLE notifications(
    id INTEGER PRIMARY KEY AUTOINCREMENT, to_session TEXT, from_session TEXT,
    resource TEXT, kind TEXT, body TEXT, created_at REAL, read_at REAL);
""")
now = time.time()
con.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)",
            ("v1sessionabcdef", "legacy v1 task", "cli", now, now, "active"))
con.execute("INSERT INTO claims(session_id,resource,mode,task,claimed_at,ttl_min,status)"
            " VALUES(?,?,?,?,?,?,?)",
            ("v1sessionabcdef", "res:LEGACY", "exclusive", "legacy claim", now, 90, "held"))
con.commit(); con.close()
PYSQL
OUT=$(HERMES_COORD_DB="$V1DB" co status 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ] && printf '%s' "$OUT" | grep -q "v1sessio" \
   && printf '%s' "$OUT" | grep -q "res:LEGACY" \
   && ! printf '%s' "$OUT" | grep -q "Traceback"; then
  ok "11a. v1-schema DB: status rc 0, session + held claim visible, no traceback"
else bad "11a. v1 migration: status" "rc=$RC out=$OUT"; fi
OUT=$(HERMES_COORD_DB="$V1DB" co register --task "v2 joins v1 db" 2>&1); RC=$?; log "$OUT"
if [ $RC -eq 0 ] && ! printf '%s' "$OUT" | grep -q "Traceback"; then
  ok "11b. v1-schema DB: register rc 0, no traceback"
else bad "11b. v1 migration: register rc 0" "rc=$RC out=$OUT"; fi

# ---------------------------------------------------------------- global traceback sweep
TB=$(grep -c "Traceback" "$LOG")
if [ "$TB" -eq 0 ]; then ok "13. zero Python tracebacks across all captured output"
else bad "13. zero tracebacks" "found $TB — see $LOG"; grep -n -A3 "Traceback" "$LOG" | head -20; fi

# ---------------------------------------------------------------- cleanup
# 2>/dev/null: on Windows runners SQLite WAL handles can linger briefly and
# rm reports "Device or resource busy" — harmless; never fail the suite on it.
rm -f "$DB" "$DB"-wal "$DB"-shm "$V1DB" "$V1DB"-wal "$V1DB"-shm \
      "$EOUT" "$COUT" "$FOUT" "$GOUT" "$K1OUT" "$K2OUT" "$LOG" 2>/dev/null || true

echo
echo "RESULT: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
