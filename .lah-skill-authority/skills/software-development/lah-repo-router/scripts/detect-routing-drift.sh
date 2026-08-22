#!/usr/bin/env bash
# detect-routing-drift.sh — Read-only drift detector for LAH Stack routing mapping
#
# Compares current filesystem state (lah-stack-repos + lah-stack-worktrees)
# against repo_mappings.json without modifying either.
#
# Usage:
#   ./detect-routing-drift.sh [repo_mappings.json]
#
# Exit codes:
#   0 — no actionable drift
#   1 — actionable drift detected
#   2 — detector or input failure
#
# Output: deterministic structured JSON to stdout.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING="${1:-$HOME/.hermes/skills/software-development/lah-repo-router/references/repo_mappings.json}"

if [ ! -f "$MAPPING" ]; then
  echo "❌ Mapping not found: $MAPPING" >&2
  exit 2
fi

exec node "$SCRIPT_DIR/detect-routing-drift.cjs" "$MAPPING"
