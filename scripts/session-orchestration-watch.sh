#!/usr/bin/env bash
# session-orchestration-watch.sh
#
# Hermes --no-agent cron script: run one session-orchestration watcher tick.
#
# Usage (cron job, no-agent):
#   hermes cron create --no-agent --schedule "every 1 minute" \
#     --name "session-orchestration-watch" \
#     --script "scripts/session-orchestration-watch.sh"
#
# The script exits silently (empty stdout, no delivery) when
# session_orchestration.enabled is false — the Hermes no_agent delivery
# rule treats empty stdout as a silent run.
#
# Environment
# -----------
# HERMES_HOME    -- Hermes data dir (default: ~/.hermes)
# HERMES_AGENT   -- Path to hermes-agent repo root (auto-detected)
# PYTHONPATH     -- Extended to include the hermes-agent root

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate hermes-agent root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_AGENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Python environment — prefer the repo venv
# ---------------------------------------------------------------------------
PYTHON="${HERMES_AGENT_ROOT}/venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="$(command -v python3 2>/dev/null || true)"
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "ERROR: python3 not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run one watcher tick
# ---------------------------------------------------------------------------
export PYTHONPATH="${HERMES_AGENT_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON}" -m session_orchestration.watcher
