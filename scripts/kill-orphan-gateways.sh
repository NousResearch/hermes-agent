#!/usr/bin/env bash
# kill-orphan-gateways.sh - Kill orphaned `gateway run --replace` processes
#
# Gateway crash/restart loops can spawn `gateway run --replace` processes
# that don't properly die - they get reparented to PID 1 (init) and accumulate
# as orphans, wasting ~30MB each.
#
# v4 - 03/06/26: fix false-positive kills of systemd-managed specialist gateways.
# v3 introduced delta detection but still classified ALL PPID=1 processes as orphans,
# which killed 11 legitimate per-profile systemd gateway services every 15 minutes.
# 
# v4 fix: any PID that belongs to a hermes-gateway*.service systemd unit is NEVER
# classified as an orphan, regardless of PPID. Only truly orphaned processes
# (PPID=1 and NOT tracked by any systemd service) are killed.
#
# Silent-by-default contract (unchanged from v3):
#   - 0 orphans            -> completely silent (no stdout, no Discord)
#   - 1+ orphans, count    ->  silent (cleanup is working as expected)
#     not worse than last
#   - count INCREASED vs   -> WARN: orphans growing, investigate
#     last run
#   - count > THRESHOLD    -> WARN regardless (structural problem)
#   - keeper-pid mismatch  -> WARN: gateway identity changed
# Threshold default 5. Override with KILL_ORPHAN_THRESHOLD env.
#
# Usage:
#   ./kill-orphan-gateways.sh          # dry-run (print what would be killed)
#   ./kill-orphan-gateways.sh --apply  # actually kill (default when non-tty)
set -euo pipefail

# P13 isolation: GATEWAY_ORPHAN_DRY_RUN=1 forces dry-run regardless of TTY.
# This overrides the non-TTY default that would kill processes when cron
# runs without a terminal.
DRY_RUN=true
if [[ "${GATEWAY_ORPHAN_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=true
elif [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
elif [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ ! -t 1 ]]; then
    DRY_RUN=false
fi

THRESHOLD=${KILL_ORPHAN_THRESHOLD:-5}
STATE_DIR="${HOME}/.hermes/state"
STATE_FILE="${STATE_DIR}/gateway-orphan-watchdog.json"

mkdir -p "$STATE_DIR"

# Build a set of PIDs tracked by ANY hermes-gateway systemd service.
# These are legitimate processes, NOT orphans, regardless of their PPID.
#
# ONLY trust a unit if it is actually active. ``systemctl list-units
# --state=running`` can leak cached/stale unit definitions in non-systemd
# contexts (cron env, missing $XDG_RUNTIME_DIR, etc.) — when no live
# ``hermes-gateway*.service`` unit exists, ``systemctl show MainPID`` may
# still return a non-zero PID that is in fact the ghost of a unit that
# never loaded. The watchdog then "protects" PIDs that are not actually
# managed by systemd, and the orphan herd grows unbounded (observed
# 2026-06-26: 13 gateway --replace procs, 0 active units, watchdog
# reported 0 orphans, 0 kills, KEEP_LIST included every PID). Guard
# against that by verifying ActiveState==active before trusting the
# MainPID. Ghost units are skipped — their PIDs go through normal
# orphan classification and get killed if PPID=1.
declare -A SYSTEMD_PIDS
for svc in $(systemctl list-units --type=service --state=running --no-legend 2>/dev/null | grep -oP 'hermes-gateway\S*\.service' || true); do
    active_state=$(systemctl show -p ActiveState --value "$svc" 2>/dev/null || true)
    if [[ "$active_state" != "active" ]]; then
        # Ghost unit — not actually running. Skip so its PID gets
        # classified as an orphan instead of protected.
        continue
    fi
    main_pid=$(systemctl show -p MainPID --value "$svc" 2>/dev/null || true)
    if [[ -n "$main_pid" && "$main_pid" != "0" ]]; then
        SYSTEMD_PIDS[$main_pid]=1
    fi
done

# Find all gateway --replace processes
mapfile -t GATEWAY_PIDS < <(ps -eo pid,ppid,lstart,args --no-headers \
    | grep -E 'python.*hermes_cli\.main gateway run --replace' \
    | grep -v grep \
    | awk '{print $1}' || true)
# Always-defined count variable; mapfile leaves array empty if no match.
GATEWAY_TOTAL=${#GATEWAY_PIDS[@]}

# Read prior state
PREV_COUNT=0
PREV_PIDS=""
if [[ -f "$STATE_FILE" ]]; then
    PREV_COUNT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('last_count',0))" "$STATE_FILE" 2>/dev/null || echo 0)
    PREV_PIDS=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(str(x) for x in d.get('last_pids',[])))" "$STATE_FILE" 2>/dev/null || echo "")
fi

# Find the live keeper (prefer systemd MainPID, fall back to newest PID)
SYSTEMD_MAIN_PID=$(systemctl show -p MainPID --value hermes-gateway.service 2>/dev/null | tr -d ' ' || true)
PRIMARY=""
if [[ -n "$SYSTEMD_MAIN_PID" && "$SYSTEMD_MAIN_PID" != "0" ]]; then
    for pid in "${GATEWAY_PIDS[@]}"; do
        if [[ "$pid" == "$SYSTEMD_MAIN_PID" ]]; then
            PRIMARY="$pid"
            break
        fi
    done
fi
if [[ -z "$PRIMARY" ]] && (( GATEWAY_TOTAL > 0 )); then
    PRIMARY=$(printf '%s\n' "${GATEWAY_PIDS[@]}" | sort -n | tail -1)
fi
PRIMARY_PPID=$(ps -o ppid= -p "$PRIMARY" 2>/dev/null | tr -d ' ' || echo "?")

# Classify: orphans are processes that are PPID=1 AND NOT tracked by any
# systemd hermes-gateway service. Systemd-managed services always have PPID=1
# (because systemd IS PID 1) but they are legitimate, NOT orphans.
KILL_LIST=()
KEPT_LIST=()
for pid in "${GATEWAY_PIDS[@]}"; do
    # Always protect the PRIMARY keeper
    if [[ -n "$PRIMARY" && "$pid" == "$PRIMARY" ]]; then
        KEPT_LIST+=("$pid")
        continue
    fi
    # Protect ANY PID tracked by a hermes-gateway systemd service
    if [[ -n "${SYSTEMD_PIDS[$pid]+_}" ]]; then
        KEPT_LIST+=("$pid")
        continue
    fi
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "?")
    if [[ "$ppid" == "1" ]]; then
        KILL_LIST+=("$pid")
    else
        KEPT_LIST+=("$pid")
    fi
done

KILLED=0
KILLED_RAM_MB=0
if ! $DRY_RUN; then
    for pid in "${KILL_LIST[@]}"; do
        # Triple-check: never kill the keeper, even if classification slipped.
        if [[ -n "$PRIMARY" && "$pid" == "$PRIMARY" ]]; then
            echo "  [SKIP] $pid is PRIMARY - refusing to kill" >&2
            continue
        fi
        # Quadruple-check: never kill a systemd-tracked PID.
        if [[ -n "${SYSTEMD_PIDS[$pid]+_}" ]]; then
            echo "  [SKIP] $pid is tracked by a systemd service - refusing to kill" >&2
            continue
        fi
        rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)
        subtree=$(ps --ppid "$pid" -o pid= 2>/dev/null || true)
        for child in $subtree; do kill -TERM "$child" 2>/dev/null || true; done
        kill -TERM "$pid" 2>/dev/null || true
        KILLED=$((KILLED + 1))
        KILLED_RAM_MB=$((KILLED_RAM_MB + rss_kb / 1024))
    done
fi

# Always persist state (so next run knows prior count)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIDS_JSON=$(printf '%s\n' "${GATEWAY_PIDS[@]}" | python3 -c "import sys,json; print(json.dumps([int(x) for x in sys.stdin if x.strip()]))")
KILL_JSON=$(printf '%s\n' "${KILL_LIST[@]}" | python3 -c "import sys,json; print(json.dumps([int(x) for x in sys.stdin if x.strip()]))")
KEEPER_JSON=$(printf '%s\n' "${KEPT_LIST[@]}" | python3 -c "import sys,json; print(json.dumps([int(x) for x in sys.stdin if x.strip()]))")
python3 - "$STATE_FILE" "$NOW" "$PRIMARY" "$PRIMARY_PPID" \
    "$PREV_COUNT" "$GATEWAY_TOTAL" "$KILLED" "$KILLED_RAM_MB" \
    "$PIDS_JSON" "$KILL_JSON" "$KEEPER_JSON" <<'PY'
import json, sys
state_file, now, primary, primary_ppid, prev_count, total, killed, ram_mb, pids, kill_list, kept_list = sys.argv[1:]
data = {
    "last_run_at": now,
    "keeper_pid": int(primary) if primary.isdigit() else None,
    "keeper_ppid": int(primary_ppid) if primary_ppid.isdigit() else None,
    "last_count": int(total),
    "last_killed": int(killed),
    "last_killed_ram_mb": int(ram_mb),
    "last_pids": json.loads(pids),
    "last_kill_list": json.loads(kill_list),
    "last_kept_list": json.loads(kept_list),
    "prev_count": int(prev_count),
}
with open(state_file, "w") as f:
    json.dump(data, f, indent=2)
PY

# Decision: should we emit a Discord alert?
TOTAL="$GATEWAY_TOTAL"
ORPHAN_COUNT=${#KILL_LIST[@]}
KEPT_COUNT=${#KEPT_LIST[@]}
SHOULD_ALERT=false
ALERT_REASON=""

# Rule 1: count grew vs last run -> growing
if (( TOTAL > PREV_COUNT && PREV_COUNT > 0 )); then
    SHOULD_ALERT=true
    ALERT_REASON="orphans grew ${PREV_COUNT}→${TOTAL} (new ones being created)"
fi

# Rule 2: count above structural threshold
if (( ORPHAN_COUNT > THRESHOLD )); then
    SHOULD_ALERT=true
    ALERT_REASON="${ORPHAN_COUNT} orphans > threshold ${THRESHOLD}"
fi

# Rule 3: dry-run with structural issues (for human review without killing)
if $DRY_RUN && (( ORPHAN_COUNT > 0 && ORPHAN_COUNT > THRESHOLD )); then
    SHOULD_ALERT=true
    ALERT_REASON="DRY-RUN: ${ORPHAN_COUNT} orphans would be killed (>threshold ${THRESHOLD})"
fi

# Output (silent when healthy, summary on alert)
if $SHOULD_ALERT; then
    ts=$(date +%d/%m/%y\ %H:%M)
    echo "🟡 Gateway Orphan Watchdog - ${ts}"
    echo "Reason: ${ALERT_REASON}"
    echo "Primary: PID ${PRIMARY:-?} (PPID=${PRIMARY_PPID:-?})"
    echo "Total gateway --replace: ${TOTAL}  |  Orphans (PPID=1, not systemd): ${ORPHAN_COUNT}  |  Systemd-kept: ${KEPT_COUNT}"
    if (( KILLED > 0 )); then
        echo "Killed this run: ${KILLED}  (~${KILLED_RAM_MB}MB RAM reclaimed)"
    fi
    echo "Prev run count: ${PREV_COUNT}  |  Threshold: ${THRESHOLD}"
    if (( ORPHAN_COUNT > 0 )); then
        echo "Orphan PIDs:"
        printf '  - %s\n' "${KILL_LIST[@]}"
    fi
    echo ""
    echo "Investigate: true orphan gateway processes with PPID=1 not tracked by systemd."
fi
# Silent when healthy. Exit 0 always.
exit 0