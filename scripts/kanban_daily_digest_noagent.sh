#!/usr/bin/env bash
set -euo pipefail
# Resolve the digest script from the repo (not a hardcoded ~/.hermes path).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIGEST_PY="${HERMES_HOME:-$HOME/.hermes}/scripts/kanban_daily_digest_noagent.py"
# Prefer repo-relative copy if it exists
REPO_RELATIVE="$SCRIPT_DIR/archive/kanban_daily_digest_noagent.py"
if [ -f "$REPO_RELATIVE" ]; then
  DIGEST_PY="$REPO_RELATIVE"
fi
python3 "$DIGEST_PY"