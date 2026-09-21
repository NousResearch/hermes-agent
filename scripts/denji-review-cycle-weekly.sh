#!/usr/bin/env bash
set -euo pipefail
# P13: disabled-staging guard — exit early when cron is disabled
if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN] $(basename "$0")"; exit 0; fi

# Wrapper: weekly denji profile review cycle.
# Separated so the cron scheduler can call it without arguments.
#
# W1-R (Batch 1): exec canonical repository-relative scripts/denji-review-cycle.py
# with --cycle weekly, rather than absent ~/.hermes/scripts/denji-review-cycle.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/denji-review-cycle.py" --cycle weekly "$@"
