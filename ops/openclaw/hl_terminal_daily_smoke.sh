#!/usr/bin/env bash
set -euo pipefail

# HL Terminal daily smoke + ingestion health
# - Intended to be run by an automated runner.
# - Exit 0 = OK
# - Exit 2 = FAIL (needs operator attention)

ROOT="${HL_TERMINAL_ROOT:-/home/yeqiuqiu/projects/hl-terminal}"
BACKEND="$ROOT/backend"
TERMINAL="$ROOT/terminal"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

run_step() {
  local name="$1"; shift
  echo "::step::$name" >&2
  "$@"
}

# 1) Backend ingestion/data health audit (cheap, purpose-built)
if [[ -f "$BACKEND/scripts/full-data-audit.mjs" ]]; then
  run_step "backend:full-data-audit" bash -lc "cd '$BACKEND' && node scripts/full-data-audit.mjs"
else
  fail "backend/scripts/full-data-audit.mjs missing"
fi

# 2) Terminal smoke (managed lite)
if [[ -f "$TERMINAL/package.json" ]]; then
  # Use npm script if present; otherwise fail.
  if bash -lc "cd '$TERMINAL' && npm run -s smoke:managed:lite"; then
    :
  else
    fail "terminal smoke:managed:lite failed"
  fi
else
  fail "terminal workspace missing"
fi

# 3) Optional: local API probe (best-effort; do not fail the whole check if server is not running)
if command -v curl >/dev/null 2>&1; then
  # If a dev/prod server is up locally, this should succeed. If not, ignore.
  curl -fsS --max-time 2 "http://127.0.0.1:3000/api/operator/health" >/dev/null 2>&1 || true
fi

echo "OK: hl-terminal daily smoke passed"
