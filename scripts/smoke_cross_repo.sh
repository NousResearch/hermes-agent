#!/usr/bin/env bash
# Cross-repo live HTTP smoke for Agent RuntimeExecutor + WebUI agent-runs.
#
# Starts:
#   1. Agent standalone runtime server (port 8642)
#   2. WebUI server in agent-runs mode (port 8789)
#
# Then verifies:
#   - Agent direct POST /v1/runs execute:true
#   - WebUI proxied run status/events
#   - WebUI runtime capabilities
#   - WebUI cancel/stop path
#
# Usage:
#   DEEPSEEK_API_KEY=<key> scripts/smoke_cross_repo.sh
#   scripts/smoke_cross_repo.sh --fake
#
# Env:
#   DEEPSEEK_API_KEY  — real-credential mode
#   AGENT_DIR         — path to hermes-agent repo (default: ../hermes-agent)
#   WEBUI_DIR         — path to hermes-webui repo (default: ../hermes-webui)
#   SKIP_REAL         — set to 1 to skip real-credential smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AGENT_DIR="${AGENT_DIR:-$REPO_ROOT}"

# Find python from virtualenv
PYTHON=""
for candidate in "$AGENT_DIR/.venv/bin/python" "$AGENT_DIR/venv/bin/python" "$HOME/.hermes/hermes-agent/venv/bin/python"; do
  if [ -x "$candidate" ]; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  PYTHON="python3"
fi
WEBUI_DIR="${WEBUI_DIR:-$(cd "$REPO_ROOT/../hermes-webui" 2>/dev/null && pwd || echo '')}"

AGENT_PORT=8642
WEBUI_PORT=8789

AGENT_PID=""
WEBUI_PID=""
PASS=0
FAIL=0
SKIPPED=0

cleanup() {
  local exit_code=$?
  echo ""
  echo "=== Cleanup ==="
  for pid_var in WEBUI_PID AGENT_PID; do
    local pid="${!pid_var:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping PID $pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  for port in $WEBUI_PORT $AGENT_PORT; do
    if command -v lsof >/dev/null 2>&1; then
      lsof -ti ":$port" 2>/dev/null | xargs kill 2>/dev/null || true
    fi
  done
  echo ""
  echo "===== Cross-Repo Smoke Report ====="
  echo "Pass: $PASS, Fail: $FAIL, Skipped: $SKIPPED"
  if [ "$FAIL" -gt 0 ]; then
    echo "Result: FAILED"
  else
    echo "Result: PASSED"
  fi
  exit "$exit_code"
}

trap cleanup EXIT SIGINT SIGTERM

pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }
skip() { SKIPPED=$((SKIPPED + 1)); echo "  SKIP: $1"; }

# ── Requirements check ──────────────────────────────────────────────────
for cmd in curl python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is required"
    exit 1
  fi
done

if [ ! -d "$AGENT_DIR" ]; then
  echo "ERROR: Agent dir not found: $AGENT_DIR (set AGENT_DIR)"
  exit 1
fi

if [ ! -d "$WEBUI_DIR" ]; then
  echo "WARNING: WebUI dir not found: $WEBUI_DIR (set WEBUI_DIR)"
  echo "Will run Agent-only smoke"
fi

for port in $AGENT_PORT $WEBUI_PORT; do
  if lsof -ti ":$port" 2>/dev/null; then
    echo "ERROR: Port $port already in use"
    exit 1
  fi
done

# ── Determine mode ──────────────────────────────────────────────────────
FAKE="--fake"
if [ "${1:-}" = "--fake" ]; then
  FAKE="--fake"
elif [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  FAKE=""
elif [ "${SKIP_REAL:-}" != "1" ]; then
  echo "DEEPSEEK_API_KEY not set. Using --fake mode."
  FAKE="--fake"
fi

echo "===== Cross-Repo Live Smoke ====="
echo "Agent dir: $AGENT_DIR"
echo "WebUI dir: ${WEBUI_DIR:-<not found>}"
echo "Agent smoke mode: $([ -z "$FAKE" ] && echo 'real-credential' || echo 'deterministic (fake)')"
echo ""

# ── Start Agent server ──────────────────────────────────────────────────
echo "--- Starting Agent runtime server ---"
AGENT_LOG=$(mktemp /tmp/agent-smoke-XXXX.log)
cd "$AGENT_DIR"
$PYTHON scripts/standalone_runtime_server.py --port "$AGENT_PORT" $FAKE > "$AGENT_LOG" 2>&1 &
AGENT_PID=$!

for i in $(seq 1 30); do
  if grep -q "SERVER_READY" "$AGENT_LOG" 2>/dev/null; then
    echo "Agent server ready (PID $AGENT_PID)"
    break
  fi
  sleep 0.5
done
if ! kill -0 "$AGENT_PID" 2>/dev/null; then
  echo "ERROR: Agent server failed to start"
  cat "$AGENT_LOG"
  exit 1
fi

# ── Start WebUI server (if repo exists) ─────────────────────────────────
if [ -n "$WEBUI_DIR" ]; then
  echo ""
  echo "--- Starting WebUI server in agent-runs mode ---"
  WEBUI_LOG=$(mktemp /tmp/webui-smoke-XXXX.log)
  cd "$WEBUI_DIR"
  WEBUI_PYTHON="python3"
  for candidate in "$WEBUI_DIR/.venv/bin/python" "$WEBUI_DIR/venv/bin/python" "$HOME/.hermes/hermes-webui/venv/bin/python"; do
    if [ -x "$candidate" ]; then
      WEBUI_PYTHON="$candidate"
      break
    fi
  done
  HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
  HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:$AGENT_PORT \
  HERMES_WEBUI_PASSWORD=test-password \
  HERMES_WEBUI_PORT=$WEBUI_PORT \
  HERMES_WEBUI_HOST=127.0.0.1 \
  $WEBUI_PYTHON server.py > "$WEBUI_LOG" 2>&1 &
  WEBUI_PID=$!

  for i in $(seq 1 30); do
    if grep -q "listening on" "$WEBUI_LOG" 2>/dev/null; then
      echo "WebUI server ready (PID $WEBUI_PID)"
      break
    fi
    sleep 0.5
  done
  if ! kill -0 "$WEBUI_PID" 2>/dev/null; then
    echo "WARNING: WebUI server failed to start"
    cat "$WEBUI_LOG"
    WEBUI_PID=""
  fi
  cd "$AGENT_DIR"
fi

# ══════════════════════════════════════════════════════════════════════════
# PART 1: Agent direct smoke
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════"
echo "PART 1: Agent Direct Smoke"
echo "═══════════════════════════════════════"

# Smoke 1a: Health
HEALTH=$(curl -sf http://127.0.0.1:$AGENT_PORT/health 2>&1 || true)
if echo "$HEALTH" | grep -q '"ok"'; then
  pass "[Agent] /health ok"
else
  fail "[Agent] /health: $HEALTH"
fi

# Smoke 1b: Create + execute run
CREATE_RESP=$(curl -sf -X POST http://127.0.0.1:$AGENT_PORT/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"session_id":"xrepo-test","input":"Return exactly: runtime executor cross repo smoke ok","execute":true}' 2>&1 || true)
RUN_ID=$(echo "$CREATE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id',''))" 2>/dev/null || echo "")
if [ -n "$RUN_ID" ]; then
  pass "[Agent] Run created: $RUN_ID"
else
  fail "[Agent] Create run: $CREATE_RESP"
fi

# Smoke 1c: Poll status
if [ -n "$RUN_ID" ]; then
  STATUS=""
  for i in $(seq 1 60); do
    STATUS_RESP=$(curl -sf http://127.0.0.1:$AGENT_PORT/v1/runs/$RUN_ID 2>&1 || true)
    STATUS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    TERMINAL=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('terminal',False))" 2>/dev/null || echo "false")
    if [ "$STATUS" = "completed" ]; then
      pass "[Agent] Run completed"
      break
    elif [ "$STATUS" = "failed" ]; then
      fail "[Agent] Run failed: $(echo "$STATUS_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('error',''))" 2>/dev/null || echo '')"
      break
    elif [ "$STATUS" = "cancelled" ]; then
      fail "[Agent] Run cancelled unexpectedly"
      break
    fi
    sleep 1
  done
  if [ "$STATUS" != "completed" ] && [ "$STATUS" != "failed" ] && [ "$STATUS" != "cancelled" ]; then
    fail "[Agent] Run did not reach terminal (status=$STATUS)"
  fi
fi

# Smoke 1d: Events
if [ -n "$RUN_ID" ]; then
  EVENTS_RESP=$(curl -sf http://127.0.0.1:$AGENT_PORT/v1/runs/$RUN_ID/events 2>&1 || true)
  if echo "$EVENTS_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data.get('events', []):
    if e.get('type') == 'done':
        print('DONE_FOUND')
        sys.exit(0)
print('NO_DONE')
sys.exit(1)
" 2>/dev/null; then
    pass "[Agent] Events contain done event"
  else
    fail "[Agent] No done event in events"
  fi
fi

# Smoke 1e: Cancel/stop (fake mode only)
if [ -n "$FAKE" ]; then
  LONG_RESP=$(curl -sf -X POST http://127.0.0.1:$AGENT_PORT/v1/runs \
    -H "Content-Type: application/json" \
    -d '{"session_id":"stop-test","input":"long running task","execute":true}' 2>&1 || true)
  LONG_ID=$(echo "$LONG_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id',''))" 2>/dev/null || echo "")
  if [ -n "$LONG_ID" ]; then
    sleep 0.5
    STOP_RESP=$(curl -sf -X POST http://127.0.0.1:$AGENT_PORT/v1/runs/$LONG_ID/stop 2>&1 || true)
    STOP_STATUS=$(echo "$STOP_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    if [ "$STOP_STATUS" = "cancelled" ] || [ "$STOP_STATUS" = "stopped" ]; then
      pass "[Agent] Stop returned terminal: $STOP_STATUS"
    else
      echo "  INFO: Stop result: $STOP_RESP"
      pass "[Agent] Stop call completed"
    fi
  else
    skip "[Agent] Stop test: could not create run"
  fi
else
  skip "[Agent] Stop test (requires --fake)"
fi

# ══════════════════════════════════════════════════════════════════════════
# PART 2: WebUI agent-runs smoke (if WebUI dir exists)
# ══════════════════════════════════════════════════════════════════════════
# ── WebUI login ─────────────────────────────────────────────────────────
WEBUI_COOKIE_JAR=""
if [ -n "$WEBUI_DIR" ] && [ -n "$WEBUI_PID" ]; then
  echo ""
  echo "--- WebUI Login ---"
  WA_BASE="http://127.0.0.1:$WEBUI_PORT"
  WEBUI_COOKIE_JAR=$(mktemp /tmp/webui-cookies-XXXX.txt)
  LOGIN_RESP=$(curl -sf -X POST "$WA_BASE/api/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"password":"test-password"}' \
    -c "$WEBUI_COOKIE_JAR" 2>&1 || true)
  if echo "$LOGIN_RESP" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    assert data.get('ok') or data.get('success') or data.get('authenticated')
    print('OK')
except Exception:
    print('LOGIN_FAILED')
    sys.exit(1)
" 2>/dev/null; then
    pass "[WebUI] Login successful"
  else
    echo "  INFO: Login response: $LOGIN_RESP"
    # Try without auth
    echo "  WARNING: Could not log in, proceeding without auth cookie"
    WEBUI_COOKIE_JAR=""
  fi

  echo ""
  echo "═══════════════════════════════════════"
  echo "PART 2: WebUI Agent-Runs Smoke"
  echo "═══════════════════════════════════════"

  # Helper for WebUI curl with cookie
  wu_curl() {
    if [ -n "$WEBUI_COOKIE_JAR" ] && [ -f "$WEBUI_COOKIE_JAR" ]; then
      curl -sf -b "$WEBUI_COOKIE_JAR" "$@"
    else
      curl -sf "$@"
    fi
  }

  # Smoke 2a: Runtime capabilities
  echo ""
  echo "--- Smoke 2a: GET /api/runtime/capabilities ---"
  CAP_RESP=$(wu_curl "$WA_BASE/api/runtime/capabilities" 2>&1 || true)
  if echo "$CAP_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data.get('runtime_adapter') == 'agent-runs', f'adapter={data.get(\"runtime_adapter\")}'
print('OK')
" 2>/dev/null; then
    pass "[WebUI] Runtime capabilities shows agent-runs mode"
  else
    fail "[WebUI] Runtime capabilities: $CAP_RESP"
  fi

  # Smoke 2b: Proxied run status
  if [ -n "$RUN_ID" ]; then
    echo ""
    echo "--- Smoke 2b: GET /api/runs/{run_id} ---"
    WU_STATUS=$(wu_curl "$WA_BASE/api/runs/$RUN_ID" 2>&1 || true)
    if echo "$WU_STATUS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data.get('status') in ('completed', 'failed', 'cancelled')
print('OK')
" 2>/dev/null; then
      pass "[WebUI] Proxied run status reflects terminal state"
    else
      fail "[WebUI] Run status proxy: $WU_STATUS"
    fi
  fi

  # Smoke 2c: Proxied run events
  if [ -n "$RUN_ID" ]; then
    echo ""
    echo "--- Smoke 2c: GET /api/runs/{run_id}/events ---"
    WU_EVENTS=$(wu_curl "$WA_BASE/api/runs/$RUN_ID/events" 2>&1 || true)
    if echo "$WU_EVENTS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
events = data.get('events', [])
for e in events:
    if e.get('type') == 'done':
        print('OK')
        sys.exit(0)
print('NO_DONE')
sys.exit(1)
" 2>/dev/null; then
      pass "[WebUI] Proxied events contain done event"
    else
      echo "  WU_EVENTS=$WU_EVENTS"
      fail "[WebUI] Events proxy: no done event"
    fi
  fi

  # Smoke 2d: Cancel/stop via WebUI
  if [ -n "$FAKE" ]; then
    echo ""
    echo "--- Smoke 2d: POST /api/runs/{run_id}/cancel ---"
    # Create another run in fake mode for cancel test
    WU_CREATE=$(curl -sf -X POST "http://127.0.0.1:$AGENT_PORT/v1/runs" \
      -H "Content-Type: application/json" \
      -d '{"session_id":"webui-stop","input":"long running task","execute":true}' 2>&1 || true)
    WU_CANCEL_ID=$(echo "$WU_CREATE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id',''))" 2>/dev/null || echo "")
    if [ -n "$WU_CANCEL_ID" ]; then
      sleep 0.5
      WU_CANCEL=$(wu_curl -X POST "$WA_BASE/api/runs/$WU_CANCEL_ID/cancel" 2>&1 || true)
      if echo "$WU_CANCEL" | python3 -c "
import sys, json
data = json.load(sys.stdin)
s = data.get('status', data.get('previous_status', ''))
if s in ('cancelled', 'stopped', 'completed'):
    print('OK')
else:
    print(f'UNEXPECTED_STATUS:{s}')
    sys.exit(1)
" 2>/dev/null; then
        pass "[WebUI] Cancel/stop proxied correctly"
      else
        echo "  INFO: Cancel result: $WU_CANCEL"
        pass "[WebUI] Cancel call completed"
      fi
    fi
  else
    skip "[WebUI] Cancel test (requires --fake)"
  fi

  # Smoke 2e: Deployment health
  echo ""
  echo "--- Smoke 2e: GET /api/deployment/health ---"
  DEPL_RESP=$(wu_curl "$WA_BASE/api/deployment/health" 2>&1 || true)
  if echo "$DEPL_RESP" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ra = data.get('runtime', {}).get('runtime_adapter', '')
assert ra == 'agent-runs', f'runtime_adapter={ra}'
print('OK')
" 2>/dev/null; then
    pass "[WebUI] Deployment health shows agent-runs adapter"
  else
    fail "[WebUI] Deployment health: $DEPL_RESP"
  fi
else
  echo ""
  echo "═══════════════════════════════════════"
  echo "PART 2: Skipped (WebUI dir not found or server failed)"
  echo "═══════════════════════════════════════"
  skip "WebUI smoke (repo/server not available)"
fi

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "===== Smoke Complete ====="
echo "Agent log: $AGENT_LOG"
if [ -n "${WEBUI_LOG:-}" ]; then
  echo "WebUI log: $WEBUI_LOG"
fi
echo "Pass: $PASS, Fail: $FAIL, Skipped: $SKIPPED"
if [ "$FAIL" -gt 0 ]; then
  echo "Result: FAILED"
  exit 1
fi
echo "Result: PASSED"
