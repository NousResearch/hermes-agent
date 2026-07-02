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

# Load secrets from ~/.hermes/.env so the feed digest can post to Discord via
# the REST API. The gateway loads .env into its own process, but this cron
# subprocess does not inherit it — without DISCORD_BOT_TOKEN the watcher detects
# attention items but silently skips every feed post ("discord_failed").
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
if [[ -f "${HERMES_HOME}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090,SC1091
    source "${HERMES_HOME}/.env" 2>/dev/null || true
    set +a
fi

# Tee the watcher's own logs (INFO to stderr via basicConfig) to a rotating-ish
# logfile. stdout stays empty so the no_agent delivery rule still treats the run
# as silent — but transitions, feed posts, and post failures become observable
# instead of vanishing into a discarded stderr. Without this, a silently-skipped
# feed ping (missing token, Discord 4xx, no transition) leaves zero trace.
WATCH_LOG="${HERMES_HOME}/logs/session-orchestration-watch.log"
mkdir -p "${HERMES_HOME}/logs" 2>/dev/null || true

exec "${PYTHON}" -m session_orchestration.watcher 2>> "${WATCH_LOG}"
