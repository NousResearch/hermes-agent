#!/usr/bin/env bash
set -euo pipefail

cd /home/workspace/hermes-agent
if [[ -x "./.venv/bin/hermes" ]]; then
  HERMES="./.venv/bin/hermes"
elif [[ -x "./venv/bin/hermes" ]]; then
  HERMES="./venv/bin/hermes"
else
  HERMES="$(command -v hermes)"
fi

STATE_DB="${HERMES_STATE_DB:-/dev/shm/hermes-state.db}"
HERMES_HOME="$($HERMES config path | python3 -c 'import os,sys; print(os.path.dirname(sys.stdin.read().strip()))')"
SESSIONS_INDEX="$HERMES_HOME/sessions/sessions.json"

flush_sessions() {
  echo "Flushing active sessions..."
  if [[ ! -f "$STATE_DB" ]]; then
    echo "  No state DB at $STATE_DB"
    return 0
  fi
  python3 - "$STATE_DB" <<'PY'
import sqlite3, sys, time
path = sys.argv[1]
conn = sqlite3.connect(path)
now = int(time.time())
rows = conn.execute('SELECT id, source, message_count, input_tokens FROM sessions WHERE ended_at IS NULL').fetchall()
if not rows:
    print('  No active sessions to flush')
else:
    for r in rows:
        print(f'  Killing: {r[0]} src={r[1]} msgs={r[2]} tokens={r[3]}')
    conn.execute('UPDATE sessions SET ended_at = ? WHERE ended_at IS NULL', (now,))
    conn.commit()
    print(f'Flushed {len(rows)} session(s)')
conn.close()
PY

  python3 - "$SESSIONS_INDEX" <<'PY' 2>/dev/null || true
import json, os, sys
idx = sys.argv[1]
if os.path.exists(idx):
    with open(idx) as f:
        data = json.load(f)
    killed = 0
    for key in list(data.keys()):
        s = data[key]
        if s.get('expires_at') is None or s.get('expires_at', 0) == 0:
            del data[key]
            killed += 1
    with open(idx, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Cleaned sessions index: {killed} removed')
PY
}

restart_gateway() {
  echo "Restarting gateway..."
  pkill -TERM -f 'hermes gateway run' 2>/dev/null || true
  sleep 3
  nohup bash start-gateway.sh >> /dev/shm/hermes-gateway.log 2>> /dev/shm/hermes-gateway_err.log &
  echo "Gateway restarted (PID: $!)"
}

case "${1:-status}" in
  glm)
    $HERMES config set model.provider zai
    $HERMES config set model.default glm-5.1
    echo "Switched to GLM (zai/glm-5.1)"
    ;;
  kimi)
    $HERMES config set model.provider kimi-coding
    $HERMES config set model.default k2p6
    echo "Switched to Kimi (kimi-coding/k2p6)"
    ;;
  minimax)
    $HERMES config set model.provider minimax
    $HERMES config set model.default minimax-m2.7
    echo "Switched to MiniMax (minimax/minimax-m2.7)"
    ;;
  mimo)
    $HERMES config set model.provider xiaomi
    $HERMES config set model.default mimo-v2.5-pro
    echo "Switched to MiMo (xiaomi/mimo-v2.5-pro)"
    ;;
  codex)
    $HERMES config set model.provider openai-codex
    $HERMES config set model.default gpt-5.5
    echo "Switched to Codex (openai-codex/gpt-5.5)"
    ;;
  flush)
    flush_sessions
    ;;
  restart)
    restart_gateway
    ;;
  status)
    echo "=== Config ==="
    $HERMES config | head -12
    echo ""
    echo "=== Active sessions ==="
    if [[ ! -f "$STATE_DB" ]]; then
      echo "  No state DB at $STATE_DB"
    else
      python3 - "$STATE_DB" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
rows = conn.execute('SELECT id, source, message_count, input_tokens FROM sessions WHERE ended_at IS NULL ORDER BY started_at DESC').fetchall()
if not rows:
    print('  No active sessions')
else:
    for r in rows:
        print(f'  {r[0][:20]}.. src={r[1]} msgs={r[2]} tokens={r[3]:,}')
    print(f'  Total: {len(rows)} session(s)')
conn.close()
PY
    fi
    echo ""
    echo "=== Gateway process ==="
    pgrep -fa "hermes gateway run" || echo "  Not running"
    ;;
  *)
    echo "Usage: $0 {glm|kimi|minimax|mimo|codex|flush|restart|status} [--restart]"
    echo ""
    echo "Commands:"
    echo "  glm       Switch to GLM (zai/glm-5.1)"
    echo "  kimi      Switch to Kimi (kimi-coding/k2p6)"
    echo "  minimax   Switch to MiniMax (minimax/minimax-m2.7)"
    echo "  mimo      Switch to MiMo (xiaomi/mimo-v2.5-pro)"
    echo "  codex     Switch to Codex (openai-codex/gpt-5.5)"
    echo "  flush     Kill all active sessions (fixes stale model/compression)"
    echo "  restart   Restart gateway without config change"
    echo "  status    Show current config, sessions, and process"
    echo ""
    echo "Flags:"
    echo "  --restart   Append to model switch to restart gateway"
    exit 1
    ;;
esac

if [[ "${2:-}" == "--restart" ]]; then
  flush_sessions
  restart_gateway
fi
