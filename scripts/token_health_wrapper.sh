#!/bin/bash
# P13: TOKEN_HEALTH_DRY_RUN=1 short-circuits before any Python invocation.
# HERMES_HOME is respected for runbook output paths.
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
if [ "$rc" -ne 0 ]; then
    echo "Token health check failed (rc=$rc)"
    echo "MEDIA:$report_file"
    rm -f "$output_file"
    exit 0
fi

export TOKEN_HEALTH_JSON="$(cat "$output_file")"
/home/kensei/repos/KenseiAgent/.venv/bin/python "$SCRIPT_DIR/token_health_render.py" "$report_file"

# Extract overall status from JSON
overall_status=$(echo "$TOKEN_HEALTH_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['overall'])")

if [ "$overall_status" = "healthy" ]; then
    echo "Token health all OK"
else
    echo "Token health check output"
fi
echo "MEDIA:$report_file"
rm -f "$output_file"
exit 0
