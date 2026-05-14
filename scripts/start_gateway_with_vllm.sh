#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
GATEWAY_LOG="${HERMES_GATEWAY_LOG:-$HERMES_HOME/logs/gateway.log}"
GATEWAY_PID_PATH="${HERMES_GATEWAY_PID:-$HERMES_HOME/gateway.pid}"

mkdir -p "$(dirname "$GATEWAY_LOG")"

"$ROOT_DIR/scripts/start_vllm_qwen35_9b.sh"

if [[ -f "$GATEWAY_PID_PATH" ]]; then
  if python - "$GATEWAY_PID_PATH" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
    pid = int(data.get("pid", 0) or 0)
except Exception:
    pid = 0

if pid > 0:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    print(f"Hermes gateway already running on PID {pid}")
    sys.exit(0)

sys.exit(1)
PY
  then
    exit 0
  else
    rm -f "$GATEWAY_PID_PATH"
  fi
fi

if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  VENV_ACTIVATE="$ROOT_DIR/venv/bin/activate"
elif [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  VENV_ACTIVATE="$ROOT_DIR/.venv/bin/activate"
else
  echo "Missing Hermes virtualenv (.venv or venv)"
  exit 1
fi

cmd=$(cat <<EOF
cd "$ROOT_DIR"
source "$VENV_ACTIVATE"
exec hermes gateway run
EOF
)

setsid bash -lc "$cmd" >>"$GATEWAY_LOG" 2>&1 < /dev/null &
new_pid=$!

echo "Started Hermes gateway bootstrap on PID $new_pid"
echo "Gateway log: $GATEWAY_LOG"
echo "vLLM and gateway startup requested together."
