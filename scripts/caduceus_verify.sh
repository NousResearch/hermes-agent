#!/usr/bin/env bash
# Caduceus CI gate — run before merging caduceus branch or after install_caduceus.py.
# Does not touch local GPU services or models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV=""
for candidate in "$REPO_ROOT/.venv" "$REPO_ROOT/venv" "$HOME/.hermes/hermes-agent/venv"; do
  if [ -f "$candidate/bin/activate" ]; then
    VENV="$candidate"
    break
  fi
done

if [ -z "$VENV" ]; then
  echo "error: no virtualenv found" >&2
  exit 1
fi

PYTHON="$VENV/bin/python"
cd "$REPO_ROOT"

echo "▶ Caduceus unit tests (tests/caduceus + tests/workflow)"
"$PYTHON" -m pytest tests/caduceus tests/workflow -q -o addopts=

echo "▶ Planning-loop parity rubric (offline)"
"$PYTHON" docs/caduceus/eval/parity_eval.py

echo "▶ Auto Router self-test (offline)"
"$PYTHON" docs/caduceus/eval/auto_router_selftest.py

echo "✓ caduceus_verify passed"