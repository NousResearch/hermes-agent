#!/usr/bin/env bash
# kill-orphan-mcps.sh — Kill MCP orphan processes from dead interactive sessions
#
# v2 — Fixed: matches on process NAME (comm), not full args. Filters earlyoom false positives.
#
# Interactive Hermes chat / Claude Code sessions spawn their own MCP processes
# (workspace-mcp, ms-365-mcp-server, nanobanana-mcp) that orphan when the
# interactive session exits. The production gateway's MCPs must be preserved.
#
# Usage:
#   ./kill-orphan-mcps.sh          # dry-run (print what would be killed)
#   ./kill-orphan-mcps.sh --apply  # actually kill
#
# Silent when no orphans found (exit 0, no output). Ideal for cron.

set -euo pipefail

# P13 isolation: MCP_ORPHAN_DRY_RUN=1 forces dry-run regardless of TTY.
# This overrides the non-TTY default that would kill processes when cron
# runs without a terminal.
DRY_RUN=true
if [[ "${MCP_ORPHAN_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=true
elif [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
elif [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ ! -t 1 ]]; then
    DRY_RUN=false
fi

# Identify ALL production gateway PIDs (long-running, not on a pts)
GATEWAY_PIDS=$(ps -eo pid,ppid,tty,comm,args --no-headers \
    | grep -E 'python.*gateway run' \
    | grep '?' \
    | awk '{print $1}' \
    | tr '\n' ' ')

# Find all MCP processes — match on comm/name, NOT full args (avoid earlyoom match)
# Covers: workspace-mcp, ms-365-mcp, gitnexus, github-mcp, mailbox_cleaner, adaptive_rate_limiter
MCP_PIDS=$(ps -eo pid,comm --no-headers \
    | grep -iE 'workspace-mcp|ms-365-mcp|gitnexus|github-mcp|mailbox_cleaner|adaptive_rate_limiter' \
    | grep -v grep \
    | awk '{print $1}' || true)

# nanobanana runs under npm exec or sh wrappers, so also catch child nodes.
# Grep no-match is a healthy state here, so never let it trip set -e/pipefail.
NANO_PIDS=$(ps -eo pid,comm,args --no-headers \
    | grep -i 'nanobanana' \
    | grep -v grep \
    | awk '{print $1}' || true)

# Also catch the uv launchers for workspace-mcp and npm exec launchers.
# No matches is healthy; keep it non-fatal.
LAUNCHER_PIDS=$(ps -eo pid,args --no-headers \
    | grep -E 'uv tool uvx workspace-mcp' \
    | grep -v grep \
    | awk '{print $1}' || true)

ALL_PIDS=$(echo "$MCP_PIDS $NANO_PIDS $LAUNCHER_PIDS" | tr ' ' '\n' | sort -nu)

trace_root() {
    local pid=$1
    local visited=""
    while true; do
        if echo "$visited" | grep -q ":$pid:"; then
            echo "$pid"
            return
        fi
        visited=":$pid:"
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ') || {
            echo "DEAD"
            return
        }
        if [[ -z "$ppid" || "$ppid" -le 1 ]]; then
            echo "$pid"
            return
        fi
        pid=$ppid
    done
}

KILL_LIST=()
KILLED=0
SAVED=0

for pid in $ALL_PIDS; do
    root=$(trace_root "$pid")
    comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")

    if [[ "$root" == "DEAD" ]]; then
        continue
    fi

    # Check if root is a gateway PID
    is_gateway=false
    for gpid in $GATEWAY_PIDS; do
        if [[ "$root" == "$gpid" ]]; then
            is_gateway=true
            break
        fi
    done
    if $is_gateway; then
        SAVED=$((SAVED + 1))
        continue
    fi

    # Check if root is in our own process tree (this kanban worker or its ancestors)
    OUR_PIDS=""
    p=$$
    while [[ "$p" -gt 1 ]]; do
        OUR_PIDS="$OUR_PIDS:$p:"
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ' || echo "1")
    done
    if echo "$OUR_PIDS" | grep -q ":$root:"; then
        SAVED=$((SAVED + 1))
        continue
    fi

    # Check if any ancestor belongs to a live interactive SSH session (pts with logged-in user)
    LIVE_PTS=false
    apid=$pid
    while [[ "$apid" -gt 1 ]]; do
        atty=$(ps -o tty= -p "$apid" 2>/dev/null | tr -d ' ' || echo "?")
        if [[ "$atty" =~ ^pts/ ]]; then
            apts_num=${atty#pts/}
            if who | grep -q "pts/$apts_num"; then
                LIVE_PTS=true
                break
            fi
        fi
        apid=$(ps -o ppid= -p "$apid" 2>/dev/null | tr -d ' ' || echo "1")
        if [[ -z "$apid" || "$apid" -le 1 ]]; then
            break
        fi
    done
    if $LIVE_PTS; then
        SAVED=$((SAVED + 1))
        continue
    fi

    KILL_LIST+=("$pid")
done

if [[ ${#KILL_LIST[@]} -eq 0 ]]; then
    exit 0
fi

if $DRY_RUN; then
    echo "=== MCP Orphan Killer (DRY RUN) ==="
    for pid in "${KILL_LIST[@]}"; do
        comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")
        echo "  would kill: PID $pid ($comm)"
    done
    echo "Pass --apply to execute."
    exit 0
fi

# Kill mode: silent on normal operations (≤5 orphans = routine cron cleanup)
# Only output when kill count is high (possible runaway) or kills fail
THRESHOLD=5

for pid in "${KILL_LIST[@]}"; do
    comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")
    subtree=$(ps --ppid "$pid" -o pid= 2>/dev/null || true)
    for child in $subtree; do kill -TERM "$child" 2>/dev/null || true; done
    kill -TERM "$pid" 2>/dev/null && KILLED=$((KILLED + 1)) || {
        echo "WARN: failed to kill PID $pid ($comm)"
    }
done

# Silent exit for routine cleanup (≤ threshold orphans)
if [[ $KILLED -le $THRESHOLD ]]; then
    exit 0
fi

# Only report large kill batches (possible runaway spawning)
echo "⚠️ Killed $KILLED orphan MCP processes (threshold: $THRESHOLD)"
free -m | awk '/^Mem:/{printf "RAM: %s used / %s total\n", $3, $2}'
