#!/usr/bin/env bash
set -euo pipefail
HERMES_HOME="${HERMES_HOME:-/home/kensei/.hermes}"

# P13: disabled-staging guard — exit early when cron is disabled
if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN] $(basename "$0")"; exit 0; fi

cd ${HERMES_HOME}/scripts
# Fetch calendar events
python3 calendar_brief_combined.py >/dev/null 2>&1
# Format and output concise Discord summary + HTML attachment
python3 calendar_brief_format.py
