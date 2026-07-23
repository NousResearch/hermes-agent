#!/usr/bin/env bash
# Focused + full gates for the universal Anthropic shared OAuth pool.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DIFF_RANGE="${1:-origin/main...HEAD}"

# Scrub credential-shaped env vars
unset ANTHROPIC_API_KEY ANTHROPIC_TOKEN CLAUDE_CODE_OAUTH_TOKEN || true
unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_API_KEY || true

TMPDIR_GATE="$(mktemp -d "${TMPDIR:-/tmp}/anthropic-shared-gates.XXXXXX")"
chmod 700 "$TMPDIR_GATE"
FOCUSED_LOG="$TMPDIR_GATE/focused.log"
FULL_LOG="$TMPDIR_GATE/full.log"
SCAN_LOG="$TMPDIR_GATE/scan.log"

cleanup() {
  rm -rf "$TMPDIR_GATE"
}
trap cleanup EXIT

echo "== secret scan (git diff $DIFF_RANGE) =="
set +e
python3 scripts/scan_auth_secrets.py --git-diff "$DIFF_RANGE" | tee "$SCAN_LOG"
SCAN_RC=${PIPESTATUS[0]}
set -e
if [[ "$SCAN_RC" -ne 0 ]]; then
  echo "secret scanner failed (rc=$SCAN_RC)" >&2
  exit "$SCAN_RC"
fi

FOCUSED=(
  tests/hermes_cli/test_auth_profile_fallback.py
  tests/agent/test_credential_pool_oauth_writethrough.py
  tests/hermes_cli/test_auth_commands.py
  tests/agent/test_anthropic_shared_store.py
  tests/agent/test_anthropic_shared_runtime.py
  tests/agent/test_anthropic_shared_refresh_multiprocess.py
  tests/agent/test_anthropic_shared_file_safety.py
  tests/hermes_cli/test_anthropic_shared_scope.py
  tests/hermes_cli/test_anthropic_shared_backup_guards.py
  tests/hermes_cli/test_anthropic_shared_dashboard_guards.py
  tests/scripts/test_scan_auth_secrets.py
  tests/scripts/test_break_glass_anthropic_scope.py
)

echo "== focused tests =="
set +e
scripts/run_tests.sh "${FOCUSED[@]}" 2>&1 | tee "$FOCUSED_LOG"
FOCUSED_RC=${PIPESTATUS[0]}
set -e

echo "== scan focused log =="
set +e
python3 scripts/scan_auth_secrets.py --input "$FOCUSED_LOG"
FOCUSED_SCAN_RC=$?
set -e

echo "== full suite =="
set +e
scripts/run_tests.sh 2>&1 | tee "$FULL_LOG"
FULL_RC=${PIPESTATUS[0]}
set -e

echo "== scan full log =="
set +e
python3 scripts/scan_auth_secrets.py --input "$FULL_LOG"
FULL_SCAN_RC=$?
set -e

RC=0
for r in "$FOCUSED_RC" "$FOCUSED_SCAN_RC" "$FULL_RC" "$FULL_SCAN_RC"; do
  if [[ "$r" -ne 0 ]]; then
    RC=1
  fi
done
echo "focused_rc=$FOCUSED_RC focused_scan_rc=$FOCUSED_SCAN_RC full_rc=$FULL_RC full_scan_rc=$FULL_SCAN_RC"
exit "$RC"
