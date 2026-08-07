#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
HERMES_PYTHON="${HERMES_PYTHON:-}"
if [[ -z "$HERMES_PYTHON" ]]; then
  for candidate in "$ROOT/.venv/bin/python" "$ROOT/venv/bin/python" "$HOME/.hermes/hermes-agent/venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      HERMES_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$HERMES_PYTHON" ]]; then
  HERMES_PYTHON="$(command -v python 2>/dev/null || true)"
fi
if [[ -z "$HERMES_PYTHON" || ! -x "$HERMES_PYTHON" ]]; then
  printf 'card-cost.sh: set HERMES_PYTHON to the repository Python\n' >&2
  exit 2
fi
exec "$HERMES_PYTHON" -m agent.run_usage_report "$@"
