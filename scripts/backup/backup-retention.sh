#!/usr/bin/env bash
# Compatibility wrapper for the P13 backup-retention-policy job.
# The actual implementation is backup-retention.py.
set -euo pipefail

# P13: disabled-staging guard
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] backup-retention (would verify archives, delete none)"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/backup-retention.py" "$@"
