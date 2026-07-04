#!/usr/bin/env bash
# Cheap local validation for the CLI smoke contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.venv/bin/activate" ] \
  || [ -f "$REPO_ROOT/venv/bin/activate" ] \
  || [ -f "$HOME/.hermes/hermes-agent/venv/bin/activate" ]; then
  "$REPO_ROOT/scripts/run_tests.sh" tests/hermes_cli/test_cli_smoke_contract.py -q
else
  cd "$REPO_ROOT"
  python3 -m pytest -o "addopts=" tests/hermes_cli/test_cli_smoke_contract.py -q
fi
