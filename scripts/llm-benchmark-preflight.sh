#!/usr/bin/env bash
# Executable preflight wrapper for llm-benchmark-weekly cron.
#
# Runs the bounded measurements collector + deterministic preflight builder,
# writes both state files atomically under HERMES_HOME/cron_states/
# llm-benchmark-weekly/, and emits the preflight JSON to stdout for LLM context.
#
# Fails closed: non-zero exit, short diagnostic on stderr, nothing on stdout.
# Credentials are never read from files — the Python module reads keys solely
# from os.environ (which the cron scheduler sanitises), so absent keys yield
# credential_unavailable rows rather than leaking secrets.
#
# Attach as the no_agent cron script for llm-benchmark-weekly:
#   script: llm-benchmark-preflight.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

PYTHON="${HERMES_CRON_PYTHON:-python3}"
[[ -x "$PYTHON" ]] || PYTHON="python3"

# Bootstrap PYTHONPATH so scripts.* imports resolve regardless of invocation cwd.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# HERMES_HOME is respected for state paths; default to ~/.hermes.
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"

exec "$PYTHON" "$SCRIPT_DIR/llm_benchmark_preflight_wrapper.py" "$@"
