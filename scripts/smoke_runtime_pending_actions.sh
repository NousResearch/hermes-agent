#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORT="${PORT:-8642}"
DELAY_SECONDS="${DELAY_SECONDS:-30}"

PYTHON=""
for candidate in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/venv/bin/python" "$HOME/.hermes/hermes-agent/venv/bin/python"; do
  if [ -x "$candidate" ]; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  PYTHON="python3"
fi

SERVER_PID=""

cleanup() {
  code=$?
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti ":$PORT" 2>/dev/null | xargs kill 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT SIGINT SIGTERM

json_get() {
  python3 -c "import json,sys; print(json.load(sys.stdin).get('$1',''))"
}

post_json() {
  curl -sf -X POST "$1" -H "Content-Type: application/json" -d "$2"
}

echo "===== Runtime Pending-Action Smoke ====="
echo "Mode: fake pauseable"
echo "Port: $PORT"
echo "Delay seconds: $DELAY_SECONDS"

if command -v lsof >/dev/null 2>&1 && lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT is busy"
  exit 1
fi

cd "$REPO_ROOT"
SERVER_LOG="$(mktemp /tmp/runtime-pending-server-XXXX.log)"
"$PYTHON" scripts/standalone_runtime_server.py --port "$PORT" --fake --fake-delay-seconds "$DELAY_SECONDS" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if grep -q "SERVER_READY" "$SERVER_LOG" 2>/dev/null; then
    echo "Server ready"
    break
  fi
  sleep 0.25
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: server failed to start"
  cat "$SERVER_LOG"
  exit 1
fi

CREATE_RESP="$(post_json "http://127.0.0.1:$PORT/v1/runs" '{"session_id":"phase21-pending","input":"phase21 pending action smoke","execute":true}')"
RUN_ID="$(printf '%s' "$CREATE_RESP" | json_get run_id)"

if [ -z "$RUN_ID" ]; then
  echo "ERROR: run_id missing"
  echo "$CREATE_RESP"
  exit 1
fi

echo "Run created: $RUN_ID"

PENDING_SEEN=0
for _ in $(seq 1 80); do
  STATUS_RESP="$(curl -sf "http://127.0.0.1:$PORT/v1/runs/$RUN_ID")"
  if printf '%s' "$STATUS_RESP" | python3 -c 'import json,sys; d=json.load(sys.stdin); ok=(not d.get("terminal")) and "apr-fake-001" in d.get("pending_approval_ids", []) and "clar-fake-001" in d.get("pending_clarify_ids", []); sys.exit(0 if ok else 1)'; then
    PENDING_SEEN=1
    echo "PASS: pending approval and clarify visible while run is non-terminal"
    break
  fi
  sleep 0.25
done

if [ "$PENDING_SEEN" != "1" ]; then
  echo "ERROR: pending actions were not visible while run was non-terminal"
  echo "$STATUS_RESP"
  exit 1
fi

post_json "http://127.0.0.1:$PORT/v1/runs/$RUN_ID/approval" '{"approval_id":"apr-fake-001","choice":"approve"}' >/dev/null
echo "PASS: approval resolved with choice=approve"

post_json "http://127.0.0.1:$PORT/v1/runs/$RUN_ID/clarify" '{"clarify_id":"clar-fake-001","answer":"yes"}' >/dev/null
echo "PASS: clarify resolved with answer=yes"

STATUS_AFTER="$(curl -sf "http://127.0.0.1:$PORT/v1/runs/$RUN_ID")"
printf '%s' "$STATUS_AFTER" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "apr-fake-001" not in d.get("pending_approval_ids", []), d; assert "clar-fake-001" not in d.get("pending_clarify_ids", []), d'
echo "PASS: pending IDs removed after resolution"

EVENTS_RESP="$(curl -sf "http://127.0.0.1:$PORT/v1/runs/$RUN_ID/events")"
printf '%s' "$EVENTS_RESP" | python3 -c '
import json, sys
events = json.load(sys.stdin).get("events", [])
def has_once(kind, text):
    matches = [e for e in events if e.get("type") == kind and text in str(e.get("payload", {}))]
    if len(matches) != 1:
        raise SystemExit(f"{kind} {text} expected once, got {len(matches)}")
has_once("approval.resolved", "apr-fake-001")
has_once("clarify.resolved", "clar-fake-001")
'
echo "PASS: resolved events appended exactly once"

COMPLETED=0
for _ in $(seq 1 90); do
  FINAL_STATUS="$(curl -sf "http://127.0.0.1:$PORT/v1/runs/$RUN_ID")"
  if printf '%s' "$FINAL_STATUS" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("status") == "completed" else 1)'; then
    COMPLETED=1
    echo "PASS: run completed after pending-action resolution"
    break
  fi
  sleep 0.5
done

if [ "$COMPLETED" != "1" ]; then
  echo "ERROR: run did not complete"
  echo "$FINAL_STATUS"
  exit 1
fi

echo "Result: PASSED"
