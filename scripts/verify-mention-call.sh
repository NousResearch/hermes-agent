#!/usr/bin/env bash
# Oracle for the stable-@mention contract (WhatsApp + Telegram), 2026-11-23.
#
# Runs the frozen contract suite on the real payload boundary:
#   Telegram update → adapter mention gate → typed event → context → reply
#   wa-reader row   → whatsapp_event → OwnerCommands → frozen prompt → reply
# Mocks exist only on external edges (loopback LLM, recording adapter, reader file).
#
# Usage: scripts/verify-mention-call.sh [extra pytest args passed to the runner]
# Exit: 0 only when every contract test passes; raw output is printed verbatim.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== mention-call Oracle: tests/mention_oracle/ ==="
"$REPO_ROOT/scripts/run_tests.sh" "$REPO_ROOT/tests/mention_oracle/" "$@"
rc=$?

echo
if [ "$rc" -eq 0 ]; then
    echo "=== Oracle result: PASS (rc=0) ==="
else
    echo "=== Oracle result: FAIL (rc=$rc) ==="
fi
exit "$rc"
