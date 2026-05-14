#!/usr/bin/env bash
set -euo pipefail
PY=${HERMES_NOTION_SYNC_PYTHON:-/home/solo/.hermes/hermes-agent/venv/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=python3
fi
DEFAULT_REPO="$HOME/.hermes/hermes-agent"
if [[ -n "${HERMES_HOME:-}" && "$HERMES_HOME" == */profiles/* ]]; then
  DEFAULT_REPO="${HERMES_HOME%%/profiles/*}/hermes-agent"
fi
REPO=${HERMES_NOTION_SYNC_REPO:-$DEFAULT_REPO}
MAX_CREATES=${HERMES_NOTION_SYNC_MAX_CREATES:-25}
ARGS=(--apply --quiet --max-creates "$MAX_CREATES")
if [[ -n "${HERMES_NOTION_SYNC_REPORT_DIR:-}" ]]; then
  ARGS+=(--report-dir "$HERMES_NOTION_SYNC_REPORT_DIR")
fi
cd "$REPO"
if [[ -f "$REPO/hermes_cli/notion_kanban_sync.py" ]]; then
  exec "$PY" -m hermes_cli.notion_kanban_sync "${ARGS[@]}"
fi
SCRIPT=${HERMES_NOTION_SYNC_SCRIPT:-$HOME/.hermes/profiles/dev/scripts/notion_kanban_sync.py}
exec "$PY" "$SCRIPT" "${ARGS[@]}"
