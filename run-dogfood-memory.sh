#!/usr/bin/env bash
set -euo pipefail
export HERMES_HOME=/tmp/hermes-dogfood-memory
export PYTHONPATH=/tmp/hermes-memory-v2
exec /root/.hermes/hermes-agent/.venv/bin/python /tmp/hermes-memory-v2/cli.py "$@"
