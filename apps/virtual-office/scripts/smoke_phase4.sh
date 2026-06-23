#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8647}"

curl -fsS "$API_BASE/api/health"
echo

echo "== settings =="
curl -fsS "$API_BASE/api/settings"
echo

echo "== create task =="
TASK_JSON=$(curl -fsS -X POST "$API_BASE/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{"title":"Smoke test task","goal":"Reply with exactly SMOKE_OK and nothing else.","context":"Phase 4 smoke test","room":"main-office","agent":"codex","priority":"medium"}')
echo "$TASK_JSON"
echo

TASK_ID=$(python3 - <<'PY' "$TASK_JSON"
import json, sys
print(json.loads(sys.argv[1])["id"])
PY
)

echo "== run task =="
curl -fsS -X POST "$API_BASE/api/tasks/$TASK_ID/run"
echo

echo "== tasks =="
curl -fsS "$API_BASE/api/tasks"
echo

echo "== handoffs =="
curl -fsS "$API_BASE/api/handoffs"
echo

echo "== logs =="
curl -fsS "$API_BASE/api/logs?task_id=$TASK_ID"
echo
