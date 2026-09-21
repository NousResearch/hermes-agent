#!/usr/bin/env bash
# GitNexus re-index runner — detached worker for gitnexus-reindex-daily.sh
# and gitnexus-reindex-hook.sh.
# Arguments: <log_file> <gitnexus_bin> <repo_path> <mode: full|incremental>
#
# Repaired 2026-07-15 (C019): explicit failure reporting on missing args
# and non-zero exit codes (previously swallowed by set -e + timeout).
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "ERROR: expected 4 arguments: <log_file> <gitnexus_bin> <repo_path> <mode>" >&2
    exit 1
fi

LOG_FILE="$1"
GITNEXUS="$2"
REPO="$3"
MODE="$4"

if [ ! -x "$GITNEXUS" ]; then
    echo "ERROR: gitnexus binary not found or not executable: $GITNEXUS" | tee -a "$LOG_FILE" >&2
    exit 1
fi
if [ ! -d "$REPO" ]; then
    echo "ERROR: repo path not found: $REPO" | tee -a "$LOG_FILE" >&2
    exit 1
fi

if [ "$MODE" = "full" ]; then
    echo "=== GitNexus Full Rebuild ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    if timeout 600 "$GITNEXUS" analyze "$REPO" --force >> "$LOG_FILE" 2>&1; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
    fi
    echo "Exit code: $EXIT_CODE — $(date)" >> "$LOG_FILE"
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: GitNexus full rebuild failed (exit $EXIT_CODE) — see $LOG_FILE" >&2
    fi
    exit $EXIT_CODE
else
    echo "=== GitNexus Incremental Re-index ===" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    if timeout 600 "$GITNEXUS" analyze "$REPO" >> "$LOG_FILE" 2>&1; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
    fi
    echo "Exit code: $EXIT_CODE — $(date)" >> "$LOG_FILE"
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "ERROR: GitNexus incremental re-index failed (exit $EXIT_CODE) — see $LOG_FILE" >&2
    fi
    exit $EXIT_CODE
fi
