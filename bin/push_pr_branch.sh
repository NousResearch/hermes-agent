#!/usr/bin/env bash
# bin/push_pr_branch.sh — Pushes active feature branch and creates an upstream PR with full label taxonomy and PR template formatting.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$(which python3 || which python)"

"$PYTHON_BIN" "$SCRIPT_DIR/create_upstream_pr.py" "$@"
