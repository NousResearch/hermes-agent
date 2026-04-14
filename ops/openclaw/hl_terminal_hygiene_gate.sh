#!/usr/bin/env bash
set -euo pipefail

# HL Terminal Hygiene gate
# Deterministic contract checks for UX parity + baseline route/security hygiene.
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

  echo "::step::hygiene:$rel_test" >&2
  if bash -lc "cd '$TERMINAL' && node --test '$rel_test'"; then
    echo "OK: hygiene:$rel_test" >&2
  else
    fail "test failed: $rel_test"
  fi
}

echo "== HL Terminal Hygiene Gate ==" >&2
echo "ROOT=$ROOT" >&2

run_test "tests/ia-primary-nav-contract.test.mjs"
run_test "tests/navigation-shortcuts-contract.test.mjs"
run_test "tests/overview-diagnostics-isolation.test.mjs"
run_test "tests/debug-routes-authorization.test.mjs"

echo "OK: hl-terminal hygiene gate passed" >&2
