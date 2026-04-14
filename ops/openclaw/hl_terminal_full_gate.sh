#!/usr/bin/env bash
set -euo pipefail

# HL Terminal FULL gate (heavier checks)
# Exit 0 = PASS
# Exit 2 = FAIL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

run_named() {
  local name="$1"
  shift
  echo "::step::$name" >&2
  "$@" || fail "$name"
  echo "OK: $name" >&2
}

echo "== HL Terminal FULL Gate ==" >&2
echo "ROOT=$ROOT" >&2

run_named "weekly_health" env HL_TERMINAL_ROOT="$ROOT" bash "$SCRIPT_DIR/hl_terminal_weekly_health.sh"

echo "OK: hl-terminal FULL gate passed" >&2
