# marker.sh -- the update marker for posix.sh, which sources it. Bash 3.2
# (macOS /bin/bash) compatible: no associative arrays, no {fd} redirections.
#
# The caller defines MARKER (the marker path), INSTALL_ROOT, DESKTOP_PID,
# HANDOFF_RUN, STARTED_AT and log(). Tests source this file on its own.
#
# Body (every reader parses it the same way; the shared corpus
# tests/fixtures/update_marker_corpus.json is the contract):
#
#     <pid>\n<started_at>\nct:<creation time>\n[delegate:<pid> ct:<ct>\n][run:<id>\n]
#
# Each line drops one leading BOM (line 1), one trailing CR and surrounding
# spaces/tabs. Line 1 (a u32 pid) and line 2 (digits that fit u64) are required, else
# the marker is MALFORMED. A line 3 that is not `ct:<n>` makes it v1. Lines 4+
# are tagged: the first well-formed `delegate:<pid> ct:<n>` and the first
# well-formed `run:<id>` count; anything else is ignored.
#
# Identity: (pid, creation time). Our own pid is ours only at our exact
# creation time (5 ms: the marker rounds to 3 decimals); a no-ct claim naming
# our pid is a previous incarnation. Any other pid matches within 2 s; with no
# recorded or readable creation time it is live only within 20 minutes of
# line 2.
#
# Mutation (A7): every read -> judge -> write/delete of the marker happens
# while holding an exclusive kernel lock on the sidecar "$MARKER.lock" (flock
# on the open file description of fd 9, through flock(1) or perl's Fcntl,
# which ships with macOS). The sidecar is never deleted. Busy for 10 s, or no
# lock tool at all, fails closed.

MARKER_CT_RE='^ct:([0-9]+(\.[0-9]+)?)$'
MARKER_DELEGATE_RE='^delegate:([0-9]+) ct:([0-9]+(\.[0-9]+)?)$'
MARKER_RUN_RE='^run:([A-Za-z0-9._-]{1,128})$'
MARKER_LOCK_TOOL="${MARKER_LOCK_TOOL:-}"
MARKER_LOCK_TOOL_FORCED="$MARKER_LOCK_TOOL"  # tests pin a tool; the checkout probe honours it
MY_PID="${MY_PID:-$$}"
MY_CT="${MY_CT:-}"

# ── process identity ─────────────────────────────────────────────────────────
proc_ct() { # pid -> creation time (unix seconds, 3 decimals), or nothing
  local pid="$1" stat rest start btime hz lstart secs
  [ "$pid" -gt 0 ] 2>/dev/null || return 0
  if [ -r "/proc/$pid/stat" ]; then
    stat="$(cat "/proc/$pid/stat" 2>/dev/null)" || return 0
    rest="${stat##*) }"  # comm may hold spaces/parens; fields resume after the last ") "
    start="$(printf '%s\n' "$rest" | awk '{print $20}')"  # field 22 overall
    btime="$(awk '/^btime /{print $2}' /proc/stat 2>/dev/null)"
    hz="$(getconf CLK_TCK 2>/dev/null)"
    [ -n "$start" ] && [ -n "$btime" ] && [ -n "$hz" ] || return 0
    awk -v b="$btime" -v s="$start" -v h="$hz" 'BEGIN{printf "%.3f\n", b + s / h}'
  elif [ "$(uname)" = "Darwin" ]; then
    # ps prints lstart in local time: render AND parse it in UTC so a DST
    # fall-back hour cannot shift the identity by 3600 s.
    lstart="$(TZ=UTC0 LC_ALL=C ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//;s/ *$//')"
    [ -n "$lstart" ] || return 0
    secs="$(TZ=UTC0 LC_ALL=C date -j -f '%a %b %e %T %Y' "$lstart" +%s 2>/dev/null)" || return 0
    [ -n "$secs" ] && printf '%s.000\n' "$secs"
  fi
}

pid_alive() { # pid exists and is not a zombie
  local pid="$1" st
  [ "$pid" -gt 0 ] 2>/dev/null || return 1
  st="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
  [ -n "$st" ] && [ "${st#Z}" = "$st" ]
}

pid_parent() { # pid -> parent pid, or nothing
  ps -o ppid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

ct_close() { # a b tolerance -> 0 iff |a - b| <= tolerance
  awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{d=a-b; if (d<0) d=-d; exit !(d<=t)}'
}

marker_now() { date +%s; }

ident_state() { # pid recorded-ct -> 0 same process, 1 gone/reused, 2 alive but a ct is unknown
  local pid="$1" want="${2:-}" have
  pid_alive "$pid" || return 1
  [ -n "$want" ] || return 2
  have="$(proc_ct "$pid")"
  [ -n "$have" ] || return 2
  ct_close "$want" "$have" 2.0
}

ident_alive() { # pid recorded-ct -> 0 iff that process may still be running (unknown counts)
  ident_state "$1" "${2:-}"; [ $? -ne 1 ]
}

marker_identity() { # pid recorded-ct age -> 0 OURS, 1 DEAD, 2 LIVE (another process)
  local pid="$1" want="${2:-}" age="$3" have
  [ "$pid" -gt 0 ] 2>/dev/null || return 1
  if [ "$pid" = "$MY_PID" ]; then
    [ -n "$MY_CT" ] || MY_CT="$(proc_ct "$MY_PID")"
    [ -n "$want" ] && [ -n "$MY_CT" ] && ct_close "$want" "$MY_CT" 0.005 && return 0
    return 1
  fi
  pid_alive "$pid" || return 1
  have=""
  [ -z "$want" ] || have="$(proc_ct "$pid")"
  if [ -z "$want" ] || [ -z "$have" ]; then
    [ "$age" -le 1200 ] && return 2
    return 1
  fi
  ct_close "$want" "$have" 2.0 && return 2
  return 1
}

# ── parsing and judging ──────────────────────────────────────────────────────
marker_u32() { # digits -> U32 (leading zeros dropped) iff it fits an unsigned 32-bit pid
  local n="${1#"${1%%[!0]*}"}"
  n="${n:-0}"
  if [ "${#n}" -gt 10 ] || { [ "${#n}" -eq 10 ] && [[ "$n" > 4294967295 ]]; }; then return 1; fi
  U32="$n"
}

marker_line() { # raw line -> LINE without one trailing CR and surrounding spaces/tabs
  LINE="${1%$'\r'}"
  LINE="${LINE#"${LINE%%[![:blank:]]*}"}"
  LINE="${LINE%"${LINE##*[![:blank:]]}"}"
}

M_PID="" M_STARTED="" M_STARTED_DIGITS="" M_CT="" M_DPID="" M_DCT="" M_RUN="" M_RUNS=""
marker_parse() { # marker text -> M_* fields. M_PID="" = MALFORMED.
  local raw n=0 l1="" l2="" dpid dct
  M_PID="" M_STARTED="" M_STARTED_DIGITS="" M_CT="" M_DPID="" M_DCT="" M_RUN="" M_RUNS=""
  while IFS= read -r raw || [ -n "$raw" ]; do
    n=$((n + 1))
    [ "$n" -ne 1 ] || raw="${raw#$'\357\273\277'}"
    marker_line "$raw"
    case "$n" in
      1) l1="$LINE" ;;
      2) l2="$LINE" ;;
      3) if [[ "$LINE" =~ $MARKER_CT_RE ]]; then M_CT="${BASH_REMATCH[1]}"; fi ;;
      *)
        if [ -z "$M_DPID" ] && [[ "$LINE" =~ $MARKER_DELEGATE_RE ]]; then
          dpid="${BASH_REMATCH[1]}" dct="${BASH_REMATCH[2]}"
          if marker_u32 "$dpid"; then M_DPID="$U32" M_DCT="$dct"; fi
        elif [[ "$LINE" =~ $MARKER_RUN_RE ]]; then
          [ -n "$M_RUN" ] || M_RUN="${BASH_REMATCH[1]}"
          M_RUNS="$M_RUNS$LINE"$'\n'
        fi ;;
    esac
  done <<EOF_MARKER
$1
EOF_MARKER
  case "$l1" in ''|*[!0-9]*) return 0 ;; esac
  case "$l2" in ''|*[!0-9]*) return 0 ;; esac
  marker_u32 "$l1" || return 0
  M_STARTED="${l2#"${l2%%[!0]*}"}"
  M_STARTED="${M_STARTED:-0}"
  M_STARTED_DIGITS="$M_STARTED"  # exact value (no leading zeros) for rewrites; M_STARTED is clamped
  # line 2 fits u64 (any digit count), compared as two 10-digit halves -- never in
  # shell arithmetic, which is signed 64-bit
  if [ "${#M_STARTED}" -gt 20 ]; then
    M_STARTED="" M_STARTED_DIGITS=""; return 0
  elif [ "${#M_STARTED}" -eq 20 ]; then
    local hi=$(( 10#${M_STARTED:0:10} )) lo=$(( 10#${M_STARTED:10} ))
    if [ "$hi" -gt 1844674407 ] || { [ "$hi" -eq 1844674407 ] && [ "$lo" -gt 3709551615 ]; }; then
      M_STARTED="" M_STARTED_DIGITS=""; return 0
    fi
  fi
  [ "${#M_STARTED}" -le 18 ] || M_STARTED=999999999999999999  # far future: "young"
  M_PID="$U32"
}

# marker text -> J_VERDICT (malformed|dead|ours|live), J_OWNER (the live owner,
# else the live delegate, else ""), J_OWNER_STATE / J_DELEGATE_STATE (0 ours,
# 1 dead/absent, 2 live), plus the M_* fields.
marker_judge() {
  local age
  marker_parse "$1"
  J_OWNER="" J_OWNER_STATE=1 J_DELEGATE_STATE=1
  if [ -z "$M_PID" ]; then J_VERDICT=malformed; return 0; fi
  age=$(( $(marker_now) - M_STARTED ))
  marker_identity "$M_PID" "$M_CT" "$age"; J_OWNER_STATE=$?
  if [ -n "$M_DPID" ]; then marker_identity "$M_DPID" "$M_DCT" "$age"; J_DELEGATE_STATE=$?; fi
  if [ "$J_OWNER_STATE" -ne 1 ]; then J_OWNER="$M_PID"
  elif [ "$J_DELEGATE_STATE" -ne 1 ]; then J_OWNER="$M_DPID"
  fi
  if [ "$J_OWNER_STATE" -eq 0 ] || [ "$J_DELEGATE_STATE" -eq 0 ]; then J_VERDICT=ours
  elif [ -n "$J_OWNER" ]; then J_VERDICT=live
  else J_VERDICT=dead
  fi
}

marker_live() { # marker text -> 0 iff someone (us included) still owns it
  marker_judge "$1"
  [ "$J_VERDICT" = live ] || [ "$J_VERDICT" = ours ]
}

marker_canonical() { # pid started ct-text [delegate-pid delegate-ct] -> body, run lines kept
  local body="$1"$'\n'"$2"$'\n'
  [ -z "$3" ] || body="${body}ct:$3"$'\n'
  [ -z "${4:-}" ] || body="${body}delegate:$4 ct:$5"$'\n'
  printf '%s%s' "$body" "$M_RUNS"
}

# ── A7: the sidecar kernel lock ──────────────────────────────────────────────
marker_lock_tool() {
  if [ -z "$MARKER_LOCK_TOOL" ]; then
    if command -v flock >/dev/null 2>&1; then MARKER_LOCK_TOOL=flock
    elif command -v perl >/dev/null 2>&1; then MARKER_LOCK_TOOL=perl
    else MARKER_LOCK_TOOL=none
    fi
  fi
}

fd_flock() { # fd wait-seconds -> 0 iff an exclusive flock is now held on that fd's open file
  # description (it outlives the helper process: the shell still holds the fd).
  marker_lock_tool
  case "$MARKER_LOCK_TOOL" in
    flock) if [ "$2" -gt 0 ]; then flock -x -w "$2" "$1"; else flock -x -n "$1"; fi ;;
    perl)
      perl -MFcntl=:flock -e '
        open(my $f, "+<&=", $ARGV[0]) or open($f, "<&=", $ARGV[0]) or exit 2;
        my $tries = $ARGV[1] * 10;
        for (my $i = 0; ; $i++) {
          exit 0 if flock($f, LOCK_EX | LOCK_NB);
          exit 1 if $i >= $tries;
          select(undef, undef, undef, 0.1);
        }' "$1" "$2" ;;
    *) return 1 ;;
  esac
}

marker_locked() { # fn args... -> runs fn holding the A7 lock; 75 when the lock is busy
  local rc
  if ! { exec 9>>"$MARKER.lock"; } 2>/dev/null; then
    log "WARNING: cannot open the update marker lock $MARKER.lock"; return 75
  fi
  if ! fd_flock 9 10; then
    exec 9>&-
    log "update marker lock is busy (or no flock/perl to take it): treating the marker as owned"
    return 75
  fi
  "$@"; rc=$?
  exec 9>&-
  return $rc
}

# ── mutations (call ONLY through marker_locked) ──────────────────────────────
marker_read() { # -> SEEN, return 1 when there is no marker
  [ -e "$MARKER" ] || return 1
  SEEN="$(cat "$MARKER" 2>/dev/null)"
}

marker_replace() { # body -> atomic replace (we hold the lock, so nobody judged in between)
  local tmp="$MARKER.$$.tmp"
  printf '%s' "$1" > "$tmp" 2>/dev/null && mv -f "$tmp" "$MARKER" 2>/dev/null && return 0
  rm -f "$tmp" 2>/dev/null
  return 1
}

marker_publish_new() { # body -> 0 iff we created the marker (never replaces one)
  local tmp="$MARKER.$$.tmp" rc
  printf '%s' "$1" > "$tmp" 2>/dev/null || { rm -f "$tmp" 2>/dev/null; return 2; }
  ln "$tmp" "$MARKER" 2>/dev/null; rc=$?  # link(2) is an atomic no-replace create
  rm -f "$tmp" 2>/dev/null
  return $rc
}

marker_young_empty() { # A3 fallback writers create then write: a fresh empty file is a claim in flight
  local mtime
  [ -f "$MARKER" ] && [ ! -s "$MARKER" ] || return 1
  mtime="$(stat -c %Y "$MARKER" 2>/dev/null || stat -f %m "$MARKER" 2>/dev/null)" || return 1
  [ -n "$mtime" ] && [ $(( $(marker_now) - mtime )) -lt 5 ]
}

marker_own_body() { # started -> our canonical claim (stable result / hand-off identity)
  [ -n "$MY_CT" ] || MY_CT="$(proc_ct "$MY_PID")"
  M_RUNS=""
  local run="${RESULT_RUN_ID:-$HANDOFF_RUN}"
  [ -z "$run" ] || M_RUNS="run:$run"$'\n'
  MARKER_BODY="$(marker_canonical "$MY_PID" "$1" "$MY_CT")"$'\n'
}

marker_ancestor() { # pid -> 0 iff it is one of our ancestors
  local p="$MY_PID" i
  for i in 1 2 3 4 5 6 7 8; do
    p="$(pid_parent "$p")"
    case "$p" in ''|0|1) return 1 ;; esac
    [ "$p" = "$1" ] && return 0
  done
  return 1
}

# The launcher-lineage rule -- ONE rule, the same in marker-claim.ps1
# (Test-MarkerLauncherRule); the shared table is
# tests/scripts/desktop_update/lineage_rule_cases.py. An OLD Desktop's bridge
# "<launcher pid>\n<started_at>\n" (v1: no ct, no delegate) names the launcher
# it spawned (be3fd671d70 checkout.ts) instead of itself. It is adopted iff it
# does not name the Desktop and
#   the named pid is alive: it is our parent AND (its parent is the Desktop OR
#                           line 2 == HERMES_UPDATE_STARTED_AT -- a Desktop
#                           that already quit leaves its launcher re-parented);
#   the named pid is gone:  line 2 == HERMES_UPDATE_STARTED_AT (the old Desktop
#                           hands the launcher the same value it writes).
# 1/0 facts: v1 names_desktop named_alive named_is_our_parent
# named_parent_is_desktop env_started_matches -> 0 adopt
marker_launcher_rule() {
  [ "$1" = 1 ] && [ "$2" = 0 ] || return 1
  if [ "$3" = 1 ]; then
    [ "$4" = 1 ] && { [ "$5" = 1 ] || [ "$6" = 1 ]; }
  else
    [ "$6" = 1 ]
  fi
}

marker_env_started_matches() { # line-2 digits -> 0 iff HERMES_UPDATE_STARTED_AT is plain ASCII
  # digits (no sign, no spaces) of the same value
  local env="${HERMES_UPDATE_STARTED_AT:-}" want="${1#"${1%%[!0]*}"}"
  case "$env" in ''|*[!0-9]*) return 1 ;; esac
  env="${env#"${env%%[!0]*}"}"
  [ "${env:-0}" = "${want:-0}" ]
}

marker_old_desktop_bridge() { # the parsed marker is an OLD Desktop's launcher bridge (rule above)
  local x="$M_PID" v1=0 names_desktop=0 alive=0 is_parent=0 parent_desktop=0 env_match=0
  [ -n "$M_CT" ] || [ -n "$M_DPID" ] || v1=1
  [ "$x" != "$DESKTOP_PID" ] || names_desktop=1
  marker_env_started_matches "$M_STARTED_DIGITS" && env_match=1
  if pid_alive "$x"; then
    alive=1
    [ "$x" != "$(pid_parent "$MY_PID")" ] || is_parent=1
    [ "$(pid_parent "$x")" != "$DESKTOP_PID" ] || parent_desktop=1
  fi
  marker_launcher_rule "$v1" "$names_desktop" "$alive" "$is_parent" "$parent_desktop" "$env_match"
}

marker_take() { # how started -> replace or create our claim; 0 ok, 2 unwritable
  marker_own_body "$2"
  if [ "$1" = new ]; then
    marker_publish_new "$MARKER_BODY"
    case $? in 0) ;; 2) return 2 ;; *) return 1 ;; esac
  else
    marker_replace "$MARKER_BODY" || return 2
  fi
  MARKER_CLAIMED=1
}

# Claim protocol (hand-off protocol 2, see SPEC in the Round 5 report):
#   --handoff-run R   adopt ONLY the Desktop's live bridge carrying run:R;
#   --desktop-pid P   (no run: an older Desktop) adopt its bridge by lineage, or
#                     claim afresh when the Desktop is our ancestor;
#   neither           claim: reclaim a dead marker, refuse a live one.
# 0 claimed/adopted, 1 refused (MARKER_REFUSED_PID), 2 claim unwritable.
marker_claim_locked() {
  local verdict=absent
  if marker_read; then
    if [ -z "$SEEN" ] && marker_young_empty; then MARKER_REFUSED_PID=unknown; return 1; fi
    marker_judge "$SEEN"; verdict="$J_VERDICT"
  fi
  if [ -n "$HANDOFF_RUN" ]; then
    if [ "$verdict" = live ] && [ "$M_PID" = "$DESKTOP_PID" ] && [ "$J_OWNER_STATE" -eq 2 ] \
        && [ -n "$M_CT" ] && [ "$M_RUN" = "$HANDOFF_RUN" ] && [ "$J_DELEGATE_STATE" -eq 1 ]; then
      STARTED_AT="$M_STARTED"  # one acquisition time for the whole chain
      marker_take replace "$M_STARTED" || return 2
      log "adopted the Desktop's update marker (desktop pid $DESKTOP_PID -> $MY_PID, run $HANDOFF_RUN)"
      return 0
    fi
    log "update marker is not the live bridge of desktop pid $DESKTOP_PID run $HANDOFF_RUN ($verdict): exiting without claiming"
    MARKER_REFUSED_PID="${J_OWNER:-${M_PID:-$DESKTOP_PID}}"
    return 1
  fi
  if [ "$DESKTOP_PID" -gt 0 ] 2>/dev/null; then
    if [ "$verdict" = live ] && [ "$J_OWNER" = "$M_PID" ] && [ "$J_DELEGATE_STATE" -eq 1 ] \
        && { [ "$M_PID" = "$DESKTOP_PID" ] || marker_old_desktop_bridge; }; then
      STARTED_AT="$M_STARTED"
      marker_take replace "$M_STARTED" || return 2
      log "adopted the Desktop's update marker (desktop pid $DESKTOP_PID, bridge pid $M_PID -> $MY_PID)"
      return 0
    fi
    case "$verdict" in
      dead|malformed|absent)
        if [ "$verdict" = dead ] && marker_old_desktop_bridge; then
          STARTED_AT="$M_STARTED"
          marker_take replace "$M_STARTED" || return 2
          log "adopted the Desktop's update marker (exited bridge pid $M_PID -> $MY_PID)"
          return 0
        fi
        if ! marker_ancestor "$DESKTOP_PID"; then
          log "no update marker from desktop pid $DESKTOP_PID and it did not start us: exiting without claiming"
          MARKER_REFUSED_PID="${M_PID:-$DESKTOP_PID}"
          return 1
        fi ;;
    esac
  fi
  case "$verdict" in
    live) MARKER_REFUSED_PID="$J_OWNER"; return 1 ;;
    ours) MARKER_BODY="$SEEN"$'\n'; MARKER_CLAIMED=1; return 0 ;;
    dead|malformed)
      rm -f "$MARKER" 2>/dev/null
      log "reclaimed a dead update marker (pid ${M_PID:-?})" ;;
  esac
  marker_take new "$STARTED_AT"
  case $? in
    0) log "claimed update marker (pid $MY_PID${MY_CT:+ ct $MY_CT})"; return 0 ;;
    2) return 2 ;;
  esac
  MARKER_REFUSED_PID=unknown
  return 1
}

marker_claim() { # FIRST action of the daemon. Sets MARKER_BODY/CLAIMED.
  marker_locked marker_claim_locked
  case $? in
    0) return 0 ;;
    2) log "WARNING: could not write update marker"; return 0 ;;
    75) MARKER_REFUSED_PID="${MARKER_REFUSED_PID:-unknown}"; return 1 ;;
    *) return 1 ;;
  esac
}

marker_add_delegate_locked() { # pid ct -> zero ONLY when the delegate was published
  marker_read || return 1
  marker_judge "$SEEN"
  if [ "$M_PID" != "$MY_PID" ] || [ "$J_OWNER_STATE" -ne 0 ]; then
    log "update marker is no longer ours; no delegate written"; return 1
  fi
  if [ -n "$M_DPID" ] && [ "$M_DPID" != "$1" ] && [ "$J_DELEGATE_STATE" -eq 2 ]; then return 1; fi
  marker_replace "$(marker_canonical "$M_PID" "$M_STARTED_DIGITS" "$M_CT" "$1" "$2")"$'\n' \
    && log "update marker names update pid $1 (ct $2) as its delegate"
}

marker_add_delegate() { # pid -> line 4 `delegate:<pid> ct:<ct>` (C1 rule 6)
  local ct
  [ "$MARKER_CLAIMED" -eq 1 ] || return 1
  ct="$(proc_ct "$1")"
  [ -n "$ct" ] || { log "WARNING: no creation time for update pid $1; marker names no delegate"; return 1; }
  marker_locked marker_add_delegate_locked "$1" "$ct"
}

marker_refresh_locked() { # old packaged Desktops age a marker on line 2 (20 min) and then
  # DELETE it, live owner or not. While we still own it, keep line 2 young.
  marker_read || return 0
  marker_judge "$SEEN"
  [ "$M_PID" = "$MY_PID" ] && [ "$J_OWNER_STATE" -eq 0 ] || return 0
  if [ "$J_DELEGATE_STATE" -eq 1 ]; then M_DPID="" M_DCT=""; fi
  marker_replace "$(marker_canonical "$M_PID" "$(marker_now)" "$M_CT" "$M_DPID" "$M_DCT")"$'\n'
}

marker_release_locked() { # A7 rule 5 / corpus "release"
  marker_read || return 0
  marker_judge "$SEEN"
  [ -n "$M_PID" ] || return 0
  if [ "$M_PID" = "$MY_PID" ] && [ "$J_OWNER_STATE" -eq 0 ]; then
    if [ -n "$M_DPID" ] && [ "$M_DPID" != "$MY_PID" ] && [ "$J_DELEGATE_STATE" -eq 2 ]; then
      marker_replace "$(marker_canonical "$M_DPID" "$M_STARTED_DIGITS" "$M_DCT")"$'\n'
      log "handed the update marker to its live delegate pid $M_DPID (hermes update)"
    else
      rm -f "$MARKER" 2>/dev/null
    fi
    return 0
  fi
  if [ "$M_DPID" = "$MY_PID" ] && [ "$J_DELEGATE_STATE" -eq 0 ]; then
    if [ "$J_OWNER_STATE" -ne 1 ]; then
      marker_replace "$(marker_canonical "$M_PID" "$M_STARTED_DIGITS" "$M_CT")"$'\n'
    else
      rm -f "$MARKER" 2>/dev/null
    fi
    return 0
  fi
  log "leaving update marker: no longer ours"
}

# ── the checkout lock (hermes_cli/update_lock.py::checkout_lock_path) ────────
checkout_lock_path() {
  local root="$INSTALL_ROOT" dot="$INSTALL_ROOT/.git" gitdir line common
  if [ -d "$dot" ]; then
    gitdir="$dot"
  elif [ -f "$dot" ]; then
    IFS= read -r line < "$dot" || [ -n "$line" ] || { printf '%s\n' "$root/.hermes-update.lock"; return; }
    line="${line#$'\357\273\277'}"; marker_line "$line"; line="$LINE"
    case "$line" in gitdir:*) ;; *) printf '%s\n' "$root/.hermes-update.lock"; return ;; esac
    marker_line "${line#gitdir:}"; gitdir="$LINE"
    case "$gitdir" in /*) ;; *) gitdir="$root/$gitdir" ;; esac
  else
    printf '%s\n' "$root/.hermes-update.lock"; return
  fi
  if [ -f "$gitdir/commondir" ]; then
    IFS= read -r common < "$gitdir/commondir" || [ -n "$common" ]
    common="${common#$'\357\273\277'}"; marker_line "$common"; common="$LINE"
    case "$common" in /*) gitdir="$common" ;; '') ;; *) gitdir="$gitdir/$common" ;; esac
  fi
  printf '%s\n' "$gitdir/hermes-update.lock"
}

checkout_lock_held() { # 0 iff some process holds the checkout kernel lock right now
  # The probe takes the REAL lock, so it must never make a concurrent `hermes
  # update` see it busy: one non-blocking try, given straight back. With perl
  # (macOS, nearly every Linux) or python3 (the daemon already needs it) the
  # take and the drop are two consecutive syscalls in one process that opened
  # the file itself -- no process exit, wait or shell work in between. flock(1)
  # can only take it on a shell fd, so there the hold spans flock(1)'s exit and
  # our close. No tool: held (fail closed, the release waits).
  local path rc tool="$MARKER_LOCK_TOOL_FORCED"
  path="$(checkout_lock_path)"
  [ -f "$path" ] || return 1
  if [ -z "$tool" ]; then
    if command -v perl >/dev/null 2>&1; then tool=perl
    elif command -v python3 >/dev/null 2>&1; then tool=python3
    elif command -v flock >/dev/null 2>&1; then tool=flock
    else tool=none
    fi
  fi
  case "$tool" in
    perl)
      perl -MFcntl=:flock -e '
        open(my $f, "<", $ARGV[0]) or exit 2;
        flock($f, LOCK_EX | LOCK_NB) or exit 1;
        flock($f, LOCK_UN); exit 0' "$path"; rc=$? ;;
    python3)
      python3 -c '
import fcntl, os, sys
try:
    fd = os.open(sys.argv[1], os.O_RDONLY)
except OSError:
    sys.exit(2)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
fcntl.flock(fd, fcntl.LOCK_UN)' "$path"; rc=$? ;;
    flock)
      { exec 8<"$path"; } 2>/dev/null || return 1
      flock -x -n 8; rc=$?
      exec 8<&- ;;  # closing our probe fd releases it when we did get it
    *) rc=1 ;;
  esac
  [ "$rc" -eq 1 ]
}

# ── helper ops for the Desktop (it never mutates the marker itself) ──────────
marker_op_reclaim_locked() {
  marker_read || { echo absent; return 0; }
  if [ -z "$SEEN" ] && marker_young_empty; then echo busy; return 0; fi
  marker_judge "$SEEN"
  case "$J_VERDICT" in
    live|ours) echo "live $J_OWNER" ;;
    *)
      if checkout_lock_held; then echo held; return 0; fi
      rm -f "$MARKER" 2>/dev/null
      log "reclaimed a dead update marker for the Desktop (pid ${M_PID:-?})"
      echo reclaimed ;;
  esac
}

marker_op_withdraw_locked() {
  marker_read || { echo absent; return 0; }
  marker_judge "$SEEN"
  if [ -z "$M_PID" ] || [ "$M_RUN" != "$HANDOFF_RUN" ]; then echo foreign; return 0; fi
  if [ "$M_PID" != "$DESKTOP_PID" ] && [ "$J_OWNER_STATE" -eq 2 ]; then echo "taken $M_PID"; return 0; fi
  if [ "$M_PID" = "$DESKTOP_PID" ] && [ "$J_OWNER_STATE" -eq 2 ] && [ -n "$M_CT" ]; then
    rm -f "$MARKER" 2>/dev/null
    log "withdrew the Desktop's hand-off marker (run $HANDOFF_RUN)"
    echo withdrawn; return 0
  fi
  echo foreign
}

marker_op() { # reclaim | withdraw -> one verdict line on stdout (same words as marker.ps1):
  # reclaim  absent | live <pid> | busy | held (dead, but the checkout lock is held) | reclaimed
  # withdraw absent | taken <pid> (a live adopter carries the run) | withdrawn (this Desktop's
  #          own bridge, removed) | foreign (anything else, kept -- a dead adopter included)
  local rc
  case "$1" in
    reclaim) marker_locked marker_op_reclaim_locked ;;
    withdraw)
      [ -n "$HANDOFF_RUN" ] && [ "$DESKTOP_PID" -gt 0 ] 2>/dev/null || { echo "withdraw needs --handoff-run and --desktop-pid" >&2; return 64; }
      marker_locked marker_op_withdraw_locked ;;
    *) echo "unknown --marker-op: $1" >&2; return 64 ;;
  esac
  rc=$?
  [ "$rc" -ne 75 ] || { echo busy; rc=0; }
  return $rc
}
