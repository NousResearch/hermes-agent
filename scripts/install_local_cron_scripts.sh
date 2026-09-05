#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Taipei
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$HERMES_HOME/scripts"
mkdir -p "$TARGET_DIR"

for script in   hermes_upstream_update_guard.sh   hermes_upstream_preflight.py   hermes_upstream_review.py   hermes_upstream_apply.py   hermes_upstream_report.py   taiex_0050_report.sh   taiex_0050_report_open.sh   taiex_0050_report_normal.sh   taiex_0050_report_close.sh   taiwan_weather_report.sh; do
  install -m 0755 "$SCRIPT_DIR/$script" "$TARGET_DIR/$script"
done
