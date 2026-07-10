#!/usr/bin/env bash
# PRD-8: Aegis LCM store snapshot + fresh-reset for a clean Phase-2 benchmark.
# Implements docs/PRD-8-aegis-lcm-store-reset.md (3-pass Opus reviewed).
#
# Safety model: snapshot+verify -> move-aside (never delete) -> restart -> prove
# clean. Reversible at every step. Touches ONLY ~/.hermes/profiles/aegis/.
#
# Usage:
#   lcm_reset_aegis_store.sh --dry-run   # plan only, mutate nothing
#   lcm_reset_aegis_store.sh --go        # execute (privileged: bounces Aegis gw)
#   lcm_reset_aegis_store.sh --rollback <TS>   # restore *.polluted-<TS>
set -u

PROFILE_DIR="$HOME/.hermes/profiles/aegis"
DB="$PROFILE_DIR/lcm.db"
LOCK="$PROFILE_DIR/.lcm-reset.lock"
TS="$(date +%Y%m%d_%H%M%S)"
UID_NUM="$(id -u)"
SENTINEL_NS_RE='LCM-LIVE-RECOVERY-|LCM-ARMB-|recover-'
PROBE="RESET-PROBE-$TS"
MODE="${1:-}"

log(){ printf '[reset %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die(){ printf '[reset FATAL] %s\n' "$*" >&2; release_lock; exit 1; }

# ---- advisory lock (atomic mkdir = O_EXCL) ------------------------------------
acquire_lock(){
  if mkdir "$LOCK" 2>/dev/null; then
    printf '%s %s\n' "$$" "$TS" > "$LOCK/owner"; return 0
  fi
  local opid; opid="$(awk '{print $1}' "$LOCK/owner" 2>/dev/null)"
  if [ -n "${opid:-}" ] && ! kill -0 "$opid" 2>/dev/null; then
    log "WARN stale lock (pid $opid dead) -> overriding"; rm -rf "$LOCK"
    mkdir "$LOCK" && printf '%s %s\n' "$$" "$TS" > "$LOCK/owner" && return 0
  fi
  die "another reset/campaign holds the lock (owner: $(cat "$LOCK/owner" 2>/dev/null)); refusing"
}
release_lock(){ [ -d "$LOCK" ] && [ "$(awk '{print $1}' "$LOCK/owner" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"; }
trap release_lock EXIT

# ---- launchd label/plist/keepalive resolution (§4.0) --------------------------
resolve_job(){
  LABEL="$(launchctl list 2>/dev/null | awk 'tolower($3) ~ /hermes.*aegis/ {print $3}')"
  local n; n="$(printf '%s\n' "$LABEL" | grep -c .)"
  [ "$n" -eq 1 ] || die "expected exactly 1 aegis launchd label, got $n: [$LABEL]"
  local print_out; print_out="$(launchctl print "gui/$UID_NUM/$LABEL" 2>/dev/null)"
  PLIST="$(printf '%s\n' "$print_out" | awk -F'= ' '/path =/{print $2; exit}' | tr -d ' ')"
  [ -n "${PLIST:-}" ] || PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  [ -f "$PLIST" ] || die "pinned plist does not exist on disk: $PLIST"
  KEEPALIVE="false"
  printf '%s\n' "$print_out" | grep -iq 'keepalive.*=.*true' && KEEPALIVE="true"
  log "LABEL=$LABEL"; log "PLIST=$PLIST"; log "KEEPALIVE=$KEEPALIVE"
}

# ---- sqlite helpers -----------------------------------------------------------
sq(){ sqlite3 "$1" "$2" 2>/dev/null; }
table_count(){ # db, table -> count or "absent"
  local exists; exists="$(sq "$1" "SELECT name FROM sqlite_master WHERE type='table' AND name='$2';")"
  [ "$exists" = "$2" ] || { echo "absent"; return; }
  sq "$1" "SELECT COUNT(*) FROM $2;"
}

db_open(){ lsof "$DB" 2>/dev/null | grep -q . ; }  # true if any proc holds the db

stop_gateway(){
  if [ "$KEEPALIVE" = "true" ]; then
    log "bootout (KeepAlive=true)"; launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
  else
    log "kill TERM (KeepAlive=false)"; launchctl kill TERM "gui/$UID_NUM/$LABEL" 2>/dev/null || true
  fi
  local i; for i in $(seq 1 60); do
    if ! db_open; then log "db fd released after ${i}s"; return 0; fi
    sleep 1
  done
  if [ "$KEEPALIVE" = "true" ]; then
    die "TIMEOUT: db still held after 60s AND Aegis is BOOTED-OUT/DOWN — run: $0 --rollback $TS (or bootstrap $PLIST)"
  fi
  die "TIMEOUT: db fd still held after 60s; aborted, nothing moved"
}

start_gateway(){
  if [ "$KEEPALIVE" = "true" ]; then
    log "bootstrap $PLIST"; launchctl bootstrap "gui/$UID_NUM" "$PLIST" 2>/dev/null || true
  else
    log "kickstart -k $LABEL"; launchctl kickstart -k "gui/$UID_NUM/$LABEL" 2>/dev/null || true
  fi
}

readiness_wait(){ # bounded no-write ping
  local i; for i in $(seq 1 12); do
    if hermes -p aegis chat -Q -q "ping" >/dev/null 2>&1; then log "gateway ready (${i}x5s)"; return 0; fi
    sleep 5
  done
  die "gateway did not accept a turn within 60s after restart"
}

# ============================ MAIN MODES =======================================
do_dry_run(){
  log "DRY RUN — resolving environment, mutating nothing."
  resolve_job
  log "DB=$DB ($(du -h "$DB" 2>/dev/null | awk '{print $1}'))"
  log "messages=$(table_count "$DB" messages) summary_nodes=$(table_count "$DB" summary_nodes)"
  local free_kb need_kb; free_kb="$(df -k "$PROFILE_DIR" | awk 'NR==2{print $4}')"
  need_kb=$(( $(du -k "$DB" | awk '{print $1}') * 2 ))
  log "disk free=${free_kb}KB need(2x db)=${need_kb}KB $([ "$free_kb" -ge "$need_kb" ] && echo OK || echo SHORT)"
  log "campaign running: $(pgrep -f 'lcm_live_recovery|lcm_arm_b' | grep -qc . && echo YES || echo no)"
  log "PLAN: stop($([ "$KEEPALIVE" = true ] && echo bootout || echo kill)) -> checkpoint -> backup+verify -> move .polluted-$TS -> restart -> probe -> prove clean"
  log "DRY RUN complete. Re-run with --go to execute."
}

do_go(){
  acquire_lock
  resolve_job
  log "pre-flight checks..."
  pgrep -f 'lcm_live_recovery|lcm_arm_b' | grep -q . && die "a campaign is running; refusing"
  local free_kb need_kb; free_kb="$(df -k "$PROFILE_DIR" | awk 'NR==2{print $4}')"
  need_kb=$(( $(du -k "$DB" | awk '{print $1}') * 2 ))
  [ "$free_kb" -ge "$need_kb" ] || die "insufficient disk: free=${free_kb}KB need=${need_kb}KB"

  log "quiescing Aegis..."
  stop_gateway

  log "checkpoint + snapshot..."
  local ckpt; ckpt="$(sq "$DB" "PRAGMA wal_checkpoint(TRUNCATE);")"
  local busy; busy="$(printf '%s' "$ckpt" | cut -d'|' -f1)"
  [ "${busy:-1}" = "0" ] || die "wal_checkpoint busy=$busy (reader still attached); aborting"
  [ ! -s "$DB-wal" ] || die "$DB-wal non-empty after TRUNCATE checkpoint; gateway not fully down"
  local auth_count; auth_count="$(sq "$DB" "SELECT COUNT(*) FROM messages;")"
  log "authoritative message count = $auth_count"
  cp "$DB" "$DB.backup-$TS" || die "backup copy failed"
  shasum -a 256 "$DB.backup-$TS" > "$DB.backup-$TS.sha256"
  [ "$(sq "$DB.backup-$TS" "PRAGMA integrity_check;")" = "ok" ] || die "backup integrity_check != ok"
  [ "$(sq "$DB.backup-$TS" "SELECT COUNT(*) FROM messages;")" = "$auth_count" ] || die "backup count mismatch"
  log "backup verified: $DB.backup-$TS (integrity ok, count=$auth_count)"

  log "moving polluted store aside (NOT deleting)..."
  [ -e "$DB" ] && mv "$DB" "$DB.polluted-$TS"
  [ -e "$DB-wal" ] && mv "$DB-wal" "$DB.polluted-$TS-wal"
  [ -e "$DB-shm" ] && mv "$DB-shm" "$DB.polluted-$TS-shm"

  log "restarting Aegis on fresh store..."
  start_gateway
  readiness_wait

  log "AC-3 functional probe (single write+recall, outside sentinel namespace)..."
  hermes -p aegis chat -q "Remember this exact fact verbatim: $PROBE. Reply OK." >/dev/null 2>&1 || true
  hermes -p aegis chat -q "What was the exact fact I just gave you?" >/dev/null 2>&1 || true

  log "proving clean..."
  local mc nc; mc="$(table_count "$DB" messages)"; nc="$(table_count "$DB" summary_nodes)"
  log "POST-RESET: messages=$mc summary_nodes=$nc (expect messages in {absent,1..2}, nodes in {absent,0})"
  case "$nc" in absent|0) ;; *) die "summary_nodes=$nc not clean";; esac
  case "$mc" in absent|0|1|2) ;; *) die "messages=$mc not clean (probe should be the only rows)";; esac

  cat <<EOF

---------------- PRD-8 RESET COMPLETE () ----------------
LABEL=$LABEL  PLIST=$PLIST  KEEPALIVE=$KEEPALIVE
backup (verified): $DB.backup-$TS  (integrity ok, count=$auth_count)
polluted (kept):   $DB.polluted-$TS (+ -wal/-shm)
fresh store:       messages=$mc summary_nodes=$nc
probe row:         $PROBE (benign, outside sentinel namespace)
ROLLBACK:          $0 --rollback $TS
CLEANUP (after PASS + accept):
  rm -f "$DB.backup-$TS"* "$DB.polluted-$TS"*
------------------------------------------------------------
EOF
  release_lock
}

do_rollback(){
  local rts="${2:-}"; [ -n "$rts" ] || die "usage: $0 --rollback <TS>"
  [ -e "$DB.polluted-$rts" ] || die "no polluted snapshot for TS=$rts"
  acquire_lock; resolve_job
  log "rollback to .polluted-$rts ..."
  stop_gateway
  rm -f "$DB" "$DB-wal" "$DB-shm"
  mv "$DB.polluted-$rts" "$DB"
  [ -e "$DB.polluted-$rts-wal" ] && mv "$DB.polluted-$rts-wal" "$DB-wal"
  [ -e "$DB.polluted-$rts-shm" ] && mv "$DB.polluted-$rts-shm" "$DB-shm"
  start_gateway; readiness_wait
  log "rollback complete; restored count=$(table_count "$DB" messages)"
  release_lock
}

case "$MODE" in
  --dry-run) do_dry_run ;;
  --go)      do_go ;;
  --rollback) do_rollback "$@" ;;
  *) echo "usage: $0 --dry-run | --go | --rollback <TS>"; exit 2 ;;
esac
