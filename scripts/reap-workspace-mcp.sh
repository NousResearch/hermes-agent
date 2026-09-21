#!/bin/bash
# Reap orphaned workspace-mcp processes (kensei, 2026-06-03)
# See /home/kensei/reports/security_sweep_2026-06-03.md §2 (HIGH finding)
#
# workspace-mcp is spawned on-demand by hermes chat sessions and other MCP
# consumers, but never cleaned up when the consumer exits. Over time this
# accumulates ~480MB of orphan processes.
#
# This script kills any workspace-mcp process whose parent has exited
# (PPID=1) OR that is older than 2 hours.
#
# Env overrides (P13 isolation / local disposable runs):
#   REAP_DRY_RUN=1 — list candidates only, do not kill anything or write log
#   REAP_LOG_PATH — override log file location
#   REAP_MAX_AGE_SECONDS — override kill age threshold (default 7200)

set -e
LOG="${REAP_LOG_PATH:-/home/kensei/.hermes/logs/reap-workspace-mcp.log}"
MAX_AGE_SECONDS=${REAP_MAX_AGE_SECONDS:-7200}  # 2 hours
NOW=$(date +%s)
DRY_RUN=0
if [ "${REAP_DRY_RUN:-0}" = "1" ]; then DRY_RUN=1; fi

# Find candidates: any workspace-mcp --single-user process
mapfile -t PIDS < <(pgrep -f 'workspace-mcp --single-user' || true)
KILLED=0
KEPT=0

for PID in "${PIDS[@]}"; do
    if [ -z "$PID" ]; then continue; fi

    # Check parent: if PPID=1, it's orphaned
    PPID_VAL=$(ps -o ppid= -p "$PID" 2>/dev/null | tr -d ' ' || echo "")

    # Check age
    if [ -r "/proc/$PID" ]; then
        START_JIFFIES=$(awk '{print $22}' /proc/$PID/stat 2>/dev/null || echo 0)
        CLK_TCK=$(getconf CLK_TCK)
        if [ "$START_JIFFIES" -gt 0 ]; then
            START_SEC=$((START_JIFFIES / CLK_TCK))
            UPTIME=$(awk '{print $1}' /proc/uptime)
            PROC_AGE=$(echo "$UPTIME - $START_SEC" | bc -l 2>/dev/null || echo 0)
        else
            PROC_AGE=0
        fi
    else
        PROC_AGE=0
    fi

    # Kill if orphan OR older than threshold
    if [ "$PPID_VAL" = "1" ] || [ "${PROC_AGE%.*}" -gt "$MAX_AGE_SECONDS" ]; then
        REASON="orphan"
        [ "${PROC_AGE%.*}" -gt "$MAX_AGE_SECONDS" ] && REASON="age>${MAX_AGE_SECONDS}s (${PROC_AGE%.*}s)"
        if [ "$DRY_RUN" = "1" ]; then
            echo "dry-run: would kill pid=$PID reason=$REASON"
        else
            echo "$(date -Iseconds) killing pid=$PID reason=$REASON" >> "$LOG"
            kill -TERM "$PID" 2>/dev/null || true
        fi
        KILLED=$((KILLED + 1))
    else
        KEPT=$((KEPT + 1))
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    echo "dry-run reap complete: would_kill=$KILLED kept=$KEPT"
else
    echo "$(date -Iseconds) reap complete: killed=$KILLED kept=$KEPT" >> "$LOG"
fi
exit 0
