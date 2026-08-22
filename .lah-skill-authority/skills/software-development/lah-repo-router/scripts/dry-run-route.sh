#!/usr/bin/env bash
# dry-run-route.sh — Structured routing decisions for LAH Stack (v4)
# Schema v4 — Role-aware ownership + explicit write intents + fail-closed conflicts
#
# Usage: ./dry-run-route.sh [repo_mappings.json] [missions_file]
#   Reads test missions from stdin (or file), produces structured routing receipts.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING="${1:-$SCRIPT_DIR/../references/repo_mappings.json}"
MISSIONS_FILE="${2:-/dev/stdin}"

if [ ! -f "$MAPPING" ]; then
  echo "❌ Mapping not found: $MAPPING"
  exit 1
fi

exec node "$SCRIPT_DIR/dry-run-route.cjs" "$MAPPING" "$MISSIONS_FILE"
