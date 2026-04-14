#!/usr/bin/env bash
set -euo pipefail

# HL Terminal reliability measurement spec gate
# Enforces deterministic reliability measurement contracts.
# Exit 0 = PASS
# Exit 2 = FAIL

ROOT="${HL_TERMINAL_ROOT:-$(pwd)}"
TERMINAL="$ROOT/terminal"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

run_test() {
  local rel_test="$1"
  local abs_test="$TERMINAL/$rel_test"

  [[ -f "$abs_test" ]] || fail "missing test: $rel_test"

  echo "::step::reliability-measurement:$rel_test" >&2
  if bash -lc "cd '$TERMINAL' && node --test '$rel_test'"; then
    echo "OK: reliability-measurement:$rel_test" >&2
  else
    fail "test failed: $rel_test"
  fi
}

echo "== HL Terminal Reliability Measurement Gate ==" >&2
echo "ROOT=$ROOT" >&2

run_test "tests/reliability-fsm-thresholds.test.mjs"
run_test "tests/reliability-backpressure-contract.test.mjs"
run_test "tests/reliability-measurement-spec.test.mjs"

echo "OK: hl-terminal reliability measurement gate passed" >&2
