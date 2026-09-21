#!/usr/bin/env bash
# Publish approved personal drafts to Postiz for scheduling and posting.
# Runs as no_agent cron. Silent when nothing to publish.
#
# HERMES_AGENT_ROOT is supplied by the cron scheduler from the job workdir.
# POSTIZ_DRY_RUN=1 performs read-only checks but never claims or publishes.
set -euo pipefail

RUNTIME_ROOT="${HERMES_AGENT_ROOT:-$PWD}"
CE="$RUNTIME_ROOT/content_engine"
[[ -d "$CE" ]] || { echo "ERROR: content engine not found under runtime root: $RUNTIME_ROOT" >&2; exit 1; }
PYTHON="$RUNTIME_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: runtime Python not executable: $PYTHON" >&2; exit 1; }
export CONTENT_ENGINE_ROOT="$CE"
export CONTENT_ENGINE_DB_PATH="${CONTENT_ENGINE_DB_PATH:-$CE/db/content_engine.db}"
[[ -f "$CONTENT_ENGINE_DB_PATH" ]] || { echo "ERROR: content engine database missing: $CONTENT_ENGINE_DB_PATH" >&2; exit 1; }
cd "$CE"

# Source env for DB credentials
set -a
source /home/kensei/.hermes/.env 2>/dev/null || true
set +a

export PYTHONPATH="${CE}:${PYTHONPATH:-}"

args=()
[[ "${POSTIZ_DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
timeout 60 "$PYTHON" publish_to_postiz.py "${args[@]}"
