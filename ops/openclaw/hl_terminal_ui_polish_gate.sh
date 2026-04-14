#!/usr/bin/env bash
set -euo pipefail

# HL Terminal UI_POLISH gate
# Tier: HYGIENE -> UI_POLISH -> STABILITY
# Exit 0 = PASS
# Exit 2 = FAIL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"
TERMINAL="$ROOT/terminal"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

run_step() {
  local name="$1"
  shift
  echo "::step::$name" >&2
  "$@" || fail "$name"
  echo "OK: $name" >&2
}

run_node_test_if_present() {
  local label="$1"
  local rel_test_path="$2"

  if [[ ! -f "$TERMINAL/$rel_test_path" ]]; then
    fail "$label missing ($rel_test_path)"
  fi

  run_step "$label" bash -lc "cd '$TERMINAL' && node --test '$rel_test_path'"
}

echo "== HL Terminal UI_POLISH Gate ==" >&2
echo "ROOT=$ROOT" >&2

if [[ ! -d "$TERMINAL" ]]; then
  fail "terminal directory not found at $TERMINAL"
fi

run_node_test_if_present "keyboard:ia-primary-nav" "tests/ia-primary-nav-contract.test.mjs"
run_node_test_if_present "keyboard:shortcuts-contract" "tests/navigation-shortcuts-contract.test.mjs"

run_step "ui-polish:runtime+a11y+console+perf+tokens" bash -lc "cd '$TERMINAL' && npm run gate:ui-polish"

echo "OK: hl-terminal UI_POLISH gate passed" >&2
