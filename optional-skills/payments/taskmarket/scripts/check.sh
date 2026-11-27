#!/usr/bin/env bash
# Non-destructive Taskmarket skill checks. No paid writes.
set -euo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }

command -v taskmarket >/dev/null || fail "taskmarket CLI missing; npm i -g @lucid-agents/taskmarket"
command -v python3 >/dev/null || fail "python3 missing"

ADDR_JSON="$(taskmarket address)"
echo "$ADDR_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("ok") is True; assert d["data"]["address"].startswith("0x")'
echo "address ok"

LIST_JSON="$(taskmarket task list --status open --limit 3)"
echo "$LIST_JSON" | python3 -c '
import json,sys
d=json.load(sys.stdin)
assert d.get("ok") is True
tasks=d["data"]["tasks"]
assert isinstance(tasks, list)
if tasks:
    t=tasks[0]
    assert t.get("id","").startswith("0x")
    assert "mode" in t
    assert "submissionWindowOpen" in t
'
echo "list ok"

# get first open task if any
TASK_ID="$(echo "$LIST_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); ts=d["data"]["tasks"]; print(ts[0]["id"] if ts else "")')"
if [ -n "$TASK_ID" ]; then
  GET_JSON="$(taskmarket task get "$TASK_ID")"
  echo "$GET_JSON" | python3 -c '
import json,sys
d=json.load(sys.stdin)
assert d.get("ok") is True
data=d["data"]
assert data["id"].startswith("0x")
assert isinstance(data.get("pendingActions"), list)
'
  echo "get ok $TASK_ID"
else
  echo "get skipped (no open tasks)"
fi

echo "taskmarket skill checks passed"
