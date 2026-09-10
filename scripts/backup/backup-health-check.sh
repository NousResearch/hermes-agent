#!/usr/bin/env bash
# Compatibility entrypoint for the P13 backup-health-watchdog job.
set -euo pipefail

exec python3 "$(dirname "$0")/backup-health-check.py" "$@"
