#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HERMES_HOME="${HERMES_HOME:-$REPO_DIR/.hermes-home}"
mkdir -p "$HERMES_HOME"

exec "$REPO_DIR/venv/bin/python" -m hermes_cli.main "$@"
