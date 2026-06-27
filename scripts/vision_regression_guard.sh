#!/usr/bin/env bash
# Watchdog: silent on pass, alert on fail
set -euo pipefail

REPO="${HERMES_AGENT_REPO:-$HOME/AppData/Local/hermes/hermes-agent}"
cd "$REPO"

if [[ -x "venv/Scripts/python.exe" ]]; then
  PY="venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY="python"
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
output=$("$PY" -m pytest tests/agent/test_auxiliary_vision_config_base_url.py -q --tb=short -n 0 -o addopts='' 2>&1) || true

if echo "$output" | grep -iqE 'FAILED|ERROR|failed'; then
  echo "$output"
fi
