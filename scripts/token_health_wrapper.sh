#!/bin/bash
# P13: TOKEN_HEALTH_DRY_RUN=1 short-circuits before any Python invocation.
# HERMES_HOME is respected for runbook output paths.
# 2026-09-15 fix: render the report even when the check exits non-zero (rc=1 =
# tokens expired/unhealthy). Previously a non-zero rc skipped rendering but still
# emitted MEDIA:<report> for a file that never existed -> delivery errors and no
# alert content. Now: report always generated when JSON output exists; MEDIA only
# emitted when the file is actually present.
set -euo pipefail

if [[ "${TOKEN_HEALTH_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run: would run token_health.py + render report"
  exit 0
fi

HERMES_HOME_DIR="${HERMES_HOME:-$HOME/.hermes}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
HOME_DIR="${HOME:-/home/kensei}"
cd "$HOME_DIR"

mkdir -p "${HERMES_HOME_DIR}/runbooks/token-health/$(date +%Y-%m-%d)"
report_file="${HERMES_HOME_DIR}/runbooks/token-health/$(date +%Y-%m-%d)/report.html"

output_file=$(mktemp)
rc=0
/home/kensei/repos/KenseiAgent/.venv/bin/python "$SCRIPT_DIR/token_health.py" > "$output_file" || rc=$?
if [ ! -s "$output_file" ]; then
    echo "Token health check produced no output (rc=$rc)"
    rm -f "$output_file"
    exit 0
fi

# Render regardless of rc — the report IS the alert payload.
export TOKEN_HEALTH_JSON="$(cat "$output_file")"
/home/kensei/repos/KenseiAgent/.venv/bin/python "$SCRIPT_DIR/token_health_render.py" "$report_file" || true

# Extract overall status from JSON (fallback keeps the job alive on malformed output)
overall_status=$(echo "$TOKEN_HEALTH_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('overall','unknown'))" 2>/dev/null || echo unknown)

if [ "$overall_status" = "healthy" ] && [ "$rc" -eq 0 ]; then
    echo "Token health all OK"
else
    echo "Token health check output (overall=$overall_status, rc=$rc)"
fi

# token_health_render.py already prints its own MEDIA directive on success.
if [ ! -f "$report_file" ]; then
    echo "(token-health report not generated; skipping attachment)"
fi
rm -f "$output_file"
exit 0
