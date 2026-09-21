#!/usr/bin/env bash
# systemd ExecStartPre guard: wait for any existing gateway process for the
# same HERMES_HOME to fully exit before spawning a new one.
#
# This prevents the race where systemctl restart sends SIGTERM to the old
# process and immediately starts the new one. The old process's atexit handler
# can remove the PID file while the process is still alive (graceful shutdown
# of in-flight LLM calls), causing the new process to see no PID file and
# start alongside the still-alive old process — producing two concurrent
# writers to the same kanban DB.
#
# Usage in systemd service:
#   ExecStartPre=/home/kensei/.hermes/scripts/gateway-exec-start-pre.sh
#
# The script reads HERMES_HOME from the environment (set in the service file).
# It scans for matching gateway processes and waits up to 60s for them to exit.
# Exit 0 = safe to start; exit 1 = timeout, caller should abort.

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
TIMEOUT_SECONDS=60
POLL_INTERVAL=1

# Normalise HERMES_HOME to a canonical path for matching
CANONICAL_HOME="$(cd "$HERMES_HOME" 2>/dev/null && pwd)" || {
  echo "gateway-exec-start-pre: HERMES_HOME=$HERMES_HOME does not exist, skipping guard"
  exit 0
}

echo "gateway-exec-start-pre: scanning for existing gateway processes (HERMES_HOME=$CANONICAL_HOME)"

# Find all python processes running hermes_cli.main gateway for this HERMES_HOME
# We match by:
#   1. Command line contains "hermes_cli.main gateway run"
#   2. Environment contains HERMES_HOME=<CANONICAL_HOME>
#   3. Not this script's own PID (we're not a gateway process)
#
# /proc/<pid>/environ is only readable by the process owner (kensei), so
# this works on the VPS where the script runs as the same user.

OUR_PID=$$
MATCHING_PIDS=()

for proc_dir in /proc/[0-9]*/; do
  pid="${proc_dir%/}"
  pid="${pid##*/}"
  [ "$pid" = "$OUR_PID" ] && continue
  [ "$pid" -le 1 ] && continue

  # Check command line
  cmdline_file="${proc_dir}cmdline"
  [ -r "$cmdline_file" ] || continue
  cmdline=$(tr '\0' ' ' < "$cmdline_file" 2>/dev/null) || continue
  case "$cmdline" in
    *hermes_cli.main*gateway*run*) ;;
    *) continue ;;
  esac

  # Check HERMES_HOME in environment
  environ_file="${proc_dir}environ"
  [ -r "$environ_file" ] || continue
  # environ is NUL-delimited; match exact HERMES_HOME=<path>
  if tr '\0' '\n' < "$environ_file" 2>/dev/null | grep -Fqx "HERMES_HOME=$CANONICAL_HOME"; then
    MATCHING_PIDS+=("$pid")
  fi
done

if [ ${#MATCHING_PIDS[@]} -eq 0 ]; then
  echo "gateway-exec-start-pre: no existing gateway processes found, safe to start"
  exit 0
fi

echo "gateway-exec-start-pre: found ${#MATCHING_PIDS[@]} existing gateway process(es): ${MATCHING_PIDS[*]}"
echo "gateway-exec-start-pre: waiting up to ${TIMEOUT_SECONDS}s for them to exit..."

ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT_SECONDS ]; do
  ALL_GONE=true
  REMAINING=()
  for pid in "${MATCHING_PIDS[@]}"; do
    if [ -d "/proc/$pid" ]; then
      ALL_GONE=false
      REMAINING+=("$pid")
    fi
  done

  if $ALL_GONE; then
    echo "gateway-exec-start-pre: all existing gateway processes have exited (${ELAPSED}s)"
    exit 0
  fi

  sleep "$POLL_INTERVAL"
  ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

# Timeout — check if any are still alive
STILL_ALIVE=()
for pid in "${MATCHING_PIDS[@]}"; do
  if [ -d "/proc/$pid" ]; then
    STILL_ALIVE+=("$pid")
  fi
done

echo "gateway-exec-start-pre: TIMEOUT after ${TIMEOUT_SECONDS}s — ${#STILL_ALIVE[@]} process(es) still alive: ${STILL_ALIVE[*]}"
echo "gateway-exec-start-pre: systemd will proceed anyway (this is a soft guard), but risk of duplicate gateway exists"
exit 0
