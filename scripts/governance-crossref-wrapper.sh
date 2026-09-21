#!/usr/bin/env bash
set -euo pipefail
# P13: disabled-staging guard — exit early when cron is disabled
if [ "${DRY_RUN:-0}" = "1" ]; then echo "[DRY_RUN] $(basename "$0")"; exit 0; fi

# governance-crossref-wrapper.sh
# Wrapper for governance-crossref.py - checks if Denji review exists, runs cross-reference.
# Silent when no review file found (cron-output-contract).
#
# W1-S (Batch 1): resolves the crossref script relative to this wrapper own
# location (repository-relative) rather than the absent
# ~/.hermes/scripts/governance-crossref.py. The active implementation was
# restored from scripts/archive/governance-crossref.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Active repository-relative target: scripts/governance-crossref.py.
CROSSREF="${SCRIPT_DIR}/governance-crossref.py"

# The projector is ledger-driven.  Do not use the legacy Denji profile-review
# JSON or self-evaluation files as an authority for current findings.
LEDGER_DB="${HERMES_HOME:-/home/kensei/.hermes}/governance/profile-activity-ledger.sqlite"
if [ ! -f "$LEDGER_DB" ]; then
    exit 0
fi

OUTPUT=$(python3 "$CROSSREF" 2>&1)
if [ "$OUTPUT" != "[SILENT]" ]; then
    printf '%s\n' "$OUTPUT"
fi
