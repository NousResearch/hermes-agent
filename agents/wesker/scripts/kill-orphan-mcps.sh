#!/usr/bin/env bash
# kill-orphan-mcps.sh — Kill MCP orphan processes from dead interactive sessions
#
# Interactive Hermes chat / Claude Code sessions spawn their own MCP processes
# (workspace-mcp, ms-365-mcp, gitnexus, mailbox_cleaner, adaptive_rate_limiter,
# github-mcp) that orphan when the interactive session exits. The production
# gateway's MCPs must be preserved.
#
# This script identifies orphans by checking each MCP process against
# three safe-keep conditions (in order):
#   (a) owned by the production gateway process, or
#   (b) owned by the current kanban worker session, or
#   (c) traces to a live interactive SSH session (pts with logged-in user).
#
# Any MCP process failing all three is an orphan from a dead session
# (hermes chat, claude on pts/* that disconnected) and gets killed.
#
# Usage:
#   ./kill-orphan-mcps.sh          # dry-run (print what would be killed)
#   ./kill-orphan-mcps.sh --apply  # actually kill
#
# Bug: MCP processes orphan when Hermes sessions exit (no lifecycle cleanup)
# Full RCA: ~/.hermes/kanban/boards/ops/workspaces/t_27c06998/bug-15275-root-cause-analysis.md
# Upstream: #57228 (reconnect leak), #40343 (bg processes survive exit)

set -euo pipefail

# Default: apply mode when called from cron (no tty), dry-run when called interactively
DRY_RUN=true
if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=false
elif [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
elif [[ ! -t 1 ]]; then
    # Non-interactive (cron etc.) — default to apply
    DRY_RUN=false
fi

# Identify the production gateway PID (the long-running one, not on a pts)
GATEWAY_PID=$(ps -eo pid,ppid,tty,comm,args --no-headers \
    | grep -E 'python.*gateway run' \
    | grep '?' \
    | awk '{print $1}' \
    | head -1)

# Identify this kanban worker's session PID (our uv launcher)
# We look for uv tool uvx workspace-mcp whose parent is our own process tree
KANBAN_PID=$$
# Actually, we need to find the top-level process of our run
# The most reliable approach: find all MCP trees and trace their roots

echo "=== MCP Orphan Killer ==="
echo "Gateway PID: ${GATEWAY_PID:-<not found>}"
echo "My PID: $KANBAN_PID"
echo ""

# Find all MCP processes
MCP_PIDS=$(ps -eo pid,comm,args --no-headers \
    | grep -iE 'workspace-mcp|ms-365-mcp|gitnexus.*mcp|mailbox_cleaner_mcp|adaptive_rate_limiter|github-mcp' \
    | grep -v grep \
    | awk '{print $1}')

# Also include the uv launchers and node exec wrappers
LAUNCHER_PIDS=$(ps -eo pid,args --no-headers \
    | grep -E 'uv tool uvx (workspace-mcp|gitnexus)|node.*(gitnexus mcp|ms-365-mcp)' \
    | grep -v grep \
    | awk '{print $1}')

ALL_PIDS=$(echo "$MCP_PIDS $LAUNCHER_PIDS" | tr ' ' '\n' | sort -nu)

KILL_LIST=()
KILLED=0
SAVED=0

trace_root() {
    local pid=$1
    local visited=""
    while true; do
        # Detect cycles
        if echo "$visited" | grep -q ":$pid:"; then
            echo "$pid"
            return
        fi
        visited="$visited:$pid:"
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

for pid in $ALL_PIDS; do
    root=$(trace_root "$pid")
    comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")
    args=$(ps -o args= -p "$pid" 2>/dev/null | head -c 100 || echo "<gone>")

    if [[ "$root" == "DEAD" ]]; then
        echo "DEAD    PID $pid ($comm) — process already gone, skipping"
        continue
    fi

    # Check if root is gateway
    if [[ -n "${GATEWAY_PID:-}" && "$root" == "$GATEWAY_PID" ]]; then
        echo "KEEP    PID $pid PPID=$root ($comm) — gateway-owned"
        SAVED=$((SAVED + 1))
        continue
    fi

    # Check if root is in our own process tree (kanban worker and its ancestors)
    # Walk up from our PID to find our root
    OUR_PIDS=""
    p=$KANBAN_PID
    while [[ "$p" -gt 1 ]]; do
        OUR_PIDS="$OUR_PIDS:$p:"
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ' || echo "1")
    done

    if echo "$OUR_PIDS" | grep -q ":$root:"; then
        echo "KEEP    PID $pid PPID=$root ($comm) — own session"
        SAVED=$((SAVED + 1))
        continue
    fi

    # Check if any ancestor belongs to a live interactive SSH session (pts with logged-in user)
    # Walk up from the MCP PID and look for a pts ancestor
    LIVE_PTS=false
    apid=$pid
    while [[ "$apid" -gt 1 ]]; do
        atty=$(ps -o tty= -p "$apid" 2>/dev/null | tr -d ' ' || echo "?")
        if [[ "$atty" =~ ^pts/ ]]; then
            # Found a pts ancestor — check if that pts has a logged-in user
            acmd=$(ps -o comm= -p "$apid" 2>/dev/null | tr -d ' ' || echo "")
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
        echo "KEEP    PID $pid ($comm) — live pts session"
        SAVED=$((SAVED + 1))
        continue
    fi

    root_comm=$(ps -o comm= -p "$root" 2>/dev/null | tr -d ' ' || echo "<gone>")
    root_args=$(ps -o args= -p "$root" 2>/dev/null | head -c 80 || echo "<gone>")
    root_tt=$(ps -o tty= -p "$root" 2>/dev/null | tr -d ' ' || echo "?")
    root_elapsed=$(ps -o etime= -p "$root" 2>/dev/null | tr -d ' ' || echo "?")

    echo "ORPHAN  PID $pid ($comm) — root=PPID $root ($root_comm, tty=$root_tt, age=$root_elapsed, args=$root_args)"
    KILL_LIST+=("$pid")
done

echo ""
echo "=== Summary ==="
echo "Gateway PID: ${GATEWAY_PID:-<unknown>}"
echo "Processes kept: $SAVED"
echo "Orphans to kill: ${#KILL_LIST[@]}"
echo ""

if [[ ${#KILL_LIST[@]} -eq 0 ]]; then
    echo "No orphans to clean. All quiet."
    exit 0
fi

if $DRY_RUN; then
    echo "DRY RUN — run with --apply to kill."
    echo ""
    for pid in "${KILL_LIST[@]}"; do
        comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")
        echo "  would kill: PID $pid ($comm)"
    done
    echo ""
    echo "Pass --apply to execute."
else
    echo "KILLING orphans..."
    TOTAL_RSS=0
    for pid in "${KILL_LIST[@]}"; do
        comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo "<gone>")
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
        # Kill the whole subtree
        subtree=$(ps --ppid "$pid" -o pid= 2>/dev/null || true)
        for child in $subtree; do
            child_rss=$(ps -o rss= -p "$child" 2>/dev/null | tr -d ' ' || echo "0")
            TOTAL_RSS=$((TOTAL_RSS + child_rss))
            kill -TERM "$child" 2>/dev/null || true
        done
        kill -TERM "$pid" 2>/dev/null && echo "  killed: PID $pid ($comm, RSS=${rss}KB)" && TOTAL_RSS=$((TOTAL_RSS + rss)) || echo "  already gone: PID $pid ($comm)"
        KILLED=$((KILLED + 1))
    done
    TOTAL_MB=$((TOTAL_RSS / 1024))
    echo ""
    echo "Killed $KILLED orphan processes."
    echo "Total RSS reclaimed: ~${TOTAL_MB}MB"
    echo ""
    free -m | head -2
fi
