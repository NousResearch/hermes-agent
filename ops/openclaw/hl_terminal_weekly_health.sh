#!/usr/bin/env bash
set -euo pipefail

# HL Terminal weekly heavier health check
# Exit 0 = OK
# Exit 2 = FAIL (needs operator attention)

ROOT="${HL_TERMINAL_ROOT:-/home/yeqiuqiu/projects/hl-terminal}"
BACKEND="$ROOT/backend"
TERMINAL="$ROOT/terminal"

fail() {
  echo "ALERT: $*" >&2
  exit 2
}

run_step() {
  local name="$1"; shift
  echo "::step::$name" >&2
  "$@"
}

# Backend tests
run_step "backend:npm_test" bash -lc "cd '$BACKEND' && npm test" || fail "backend npm test failed"

# Backend audit (high/critical only)
run_step "backend:npm_audit_high" bash -lc "cd '$BACKEND' && npm audit --audit-level=high" || fail "backend npm audit (high/critical) failed"

# Terminal lint/tests/build
run_step "terminal:lint" bash -lc "cd '$TERMINAL' && npm run -s lint" || fail "terminal lint failed"
run_step "terminal:test" bash -lc "cd '$TERMINAL' && npm test" || fail "terminal npm test failed"
run_step "terminal:build" bash -lc "cd '$TERMINAL' && npm run -s build" || fail "terminal build failed"

# Full smoke (managed)
if bash -lc "cd '$TERMINAL' && npm run -s smoke:managed"; then
  :
else
  fail "terminal smoke:managed failed"
fi

echo "OK: hl-terminal weekly health passed"
