#!/usr/bin/env bash
# Gate for phase 01 — independent rebuilt-restart ceiling.
# Idempotent: creates/reuses a venv, never assumes prior state.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
REPO_ROOT="$(pwd)"

# Reset any stale state first (this skill's own rule): a crashed prior attempt must never
# poison this attempt. The only mutable state here is the venv + pytest cache.
rm -rf .pytest_cache 2>/dev/null || true

if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
    python -m venv .venv || { echo "GATE FAIL: could not create venv"; exit 1; }
fi

if [ -x ".venv/Scripts/python.exe" ]; then
    PYBIN=".venv/Scripts/python.exe"
else
    PYBIN=".venv/bin/python"
fi

"$PYBIN" -m pip show pytest >/dev/null 2>&1 || "$PYBIN" -m pip install -q -e . pytest

"$PYBIN" -m pytest \
    tests/agent/test_turn_iteration_prep.py \
    tests/agent/test_truncated_tool_call_boost.py \
    tests/agent/test_failed_turn_site_codes.py::test_previously_bare_exit_reasons_now_carry_a_code \
    -q
RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo "GATE FAIL: pytest exited $RESULT"
    exit $RESULT
fi

# Confirm the new test actually exists and ran (not silently absent/skipped).
"$PYBIN" -m pytest tests/agent/test_turn_iteration_prep.py -q -k "rebuilt_restart_ceiling_scales_with_fallback_chain_depth" --collect-only 2>&1 | grep -qE "^[0-9]+/[0-9]+ tests? collected|^1 test collected" \
    || { echo "GATE FAIL: new test test_rebuilt_restart_ceiling_scales_with_fallback_chain_depth not found"; exit 1; }

echo "GATE PASS"
exit 0
