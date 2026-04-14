#!/usr/bin/env bash
set -euo pipefail

# HL Terminal Stability gate
# Runs a deterministic stability suite twice to catch obvious flakiness.
# Exit 0 = PASS
# Exit 2 = FAIL

ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"
TERMINAL="$ROOT/terminal"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

TESTS=(
  "tests/phase6a-stream-client-regressions.test.mjs"
  "tests/phase35-events-ordering.test.mjs"
  "tests/phase66-realtime-control-plane.test.mjs"
)

run_suite_once() {
  local pass_label="$1"
  local args=()
  local rel

  for rel in "${TESTS[@]}"; do
    [[ -f "$TERMINAL/$rel" ]] || fail "missing test: $rel"
    args+=("$rel")
  done

  echo "::step::stability:$pass_label" >&2
  if bash -lc "cd '$TERMINAL' && node --test ${args[*]}"; then
    echo "OK: stability:$pass_label" >&2
  else
    fail "stability suite failed on $pass_label"
  fi
}

echo "== HL Terminal Stability Gate ==" >&2
echo "ROOT=$ROOT" >&2

run_suite_once "pass1"
run_suite_once "pass2"

echo "OK: hl-terminal stability gate passed" >&2
