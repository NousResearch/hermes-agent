#!/bin/sh
# Hermes process supervisor: one per Hermes process, reaping the process groups it owns once it dies.
#
# Usage: sh process_supervisor.sh <parent-pgid> [<spool-dir>], spawned with start_new_session=True and stdin a
# pipe whose only write end Hermes holds (hermes_cli/process_identity.py, "owner-death supervisor").
# stdin carries one command per line:
#     register <pgid>     reap this process group when Hermes dies
#     unregister <pgid>   stop reaping it (Hermes stopped that child itself)
# EOF means Hermes is gone, by any cause including SIGKILL and the OOM killer: SIGTERM every group
# still registered, wait up to 3 s, SIGKILL the survivors, exit. A clean exit is the same EOF, so a
# child Hermes never unregistered dies with it either way.
#
# POSIX sh, not Python: Hermes keeps this process resident for its whole life, and an idle sh costs
# about 0.5 MB USS where an idle interpreter costs about 8 MB. It must also never import Hermes:
# it runs after Hermes may already be dead.
#
# When <spool-dir> exists (a gateway exports telemetry for this home), a reap leaves one content-free
# record there (agent/monitoring/managed_processes.py): how many groups still had a member to signal, and
# whether any needed SIGKILL.
#
# Malformed lines are skipped: a corrupt line must not cost the reaping guarantee for every other
# group. Groups that are never a valid target (pgid <= 1, our own, Hermes's own) are refused at
# registration, so a caller bug costs one unreaped child, never the Hermes process tree.
# Residual risk: groups are reaped by number, so a registered group that exits on its own can have
# its pgid recycled before Hermes prunes it; Hermes prunes empty groups on every registration change.

trap '' INT HUP  # a dying parent's SIGINT/SIGHUP must not preempt the reap: EOF is the only signal
parent=$1
spool=${2:-}
# Every register line compares against it: a bad value would silently refuse every registration.
case $parent in ''|*[!0-9]*) echo "hermes process supervisor: bad parent pgid '$parent'; refusing to run" >&2; exit 2 ;; esac
# Run inside the parent's group, a killpg of the parent would take us down before we reap.
kill -0 -$$ 2>/dev/null || {
  echo "hermes process supervisor: not a process-group leader; refusing to run" >&2
  exit 2
}
reg=' '
while IFS=' ' read -r verb pg rest; do
  case $pg in ''|*[!0-9]*) continue ;; esac
  [ -z "$rest" ] && [ ${#pg} -le 10 ] || continue
  case $verb in
    register)
      [ "$pg" -gt 1 ] && [ "$pg" -ne $$ ] && [ "$pg" -ne "$parent" ] || continue
      case $reg in *" $pg "*) ;; *) reg="$reg$pg " ;; esac ;;
    unregister)
      keep=' '
      for g in $reg; do [ "$g" = "$pg" ] || keep="$keep$g "; done
      reg=$keep ;;
  esac
done
# Groups that still had a member to signal: a registered child that already exited and was collected is not a
# reap. (A zombie still answers a signal: a child that exited but that its owner never waited for, until init
# collects it after the owner's death, can be counted.)
reaped=0
# After the reap, never before it: a record costs a fork and a write, the reap must not wait on them.
report() {
  [ "$reaped" -gt 0 ] && [ -n "$spool" ] && [ -d "$spool" ] || return 0
  n=0
  for f in "$spool"/*.json "$spool"/.*.tmp; do [ -e "$f" ] && n=$((n + 1)); done  # temp files a dying writer left count
  [ "$n" -lt 1000 ] || return 0  # full: nothing drains it (same cap as managed_processes._MAX_SPOOLED)
  ts=$(command -p date +%s 2>/dev/null) || ts=
  case $ts in ''|0|*[!0-9]*) ts=0 ;; *) ts=${ts}000000000 ;; esac  # 0: the reader stamps it
  tmp="$spool/.supervisor-$$.tmp"
  printf '{"event":"managed_process","subsystem":"owned_children","action":"orphan_reap","outcome":"%s","actor":"supervisor","count":%s,"ts_ns":%s}\n' \
    "$1" "$reaped" "$ts" >"$tmp" 2>/dev/null && mv -f "$tmp" "$spool/$ts-$$.json" 2>/dev/null || rm -f "$tmp" 2>/dev/null
  return 0
}
for g in $reg; do kill -TERM -"$g" 2>/dev/null && reaped=$((reaped + 1)); done
for _ in 1 2 3; do
  live=''
  for g in $reg; do kill -0 -"$g" 2>/dev/null && live="$live $g"; done
  reg=$live
  [ -n "$reg" ] || { report terminated; exit 0; }
  # The default PATH: a broken PATH must not turn the TERM grace into an instant KILL.
  command -p sleep 1 2>/dev/null || sleep 1
done
for g in $reg; do kill -KILL -"$g" 2>/dev/null; done
report killed
exit 0
