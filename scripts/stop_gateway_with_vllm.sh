#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATEWAY_PID_PATH="${HERMES_GATEWAY_PID:-$HOME/.hermes/gateway.pid}"

python - "$GATEWAY_PID_PATH" <<'PY'
import json, os, signal, sys, time
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("No gateway pid file")
    raise SystemExit(0)

try:
    data = json.loads(path.read_text())
    pid = int(data.get("pid", 0) or 0)
except Exception:
    pid = 0

if pid <= 0:
    print("Gateway pid file invalid; removing stale file")
    path.unlink(missing_ok=True)
    raise SystemExit(0)

try:
    os.kill(pid, 0)
except OSError:
    print(f"Gateway process {pid} not running; removing stale pid file")
    path.unlink(missing_ok=True)
    raise SystemExit(0)

print(f"Stopping gateway pid {pid}")
os.kill(pid, signal.SIGTERM)
for _ in range(20):
    try:
        os.kill(pid, 0)
    except OSError:
        print("Gateway stopped")
        path.unlink(missing_ok=True)
        raise SystemExit(0)
    time.sleep(0.5)

print(f"Gateway pid {pid} still alive; sending SIGKILL")
os.kill(pid, signal.SIGKILL)
path.unlink(missing_ok=True)
print("Gateway killed")
PY

"$ROOT_DIR/scripts/stop_vllm_qwen35_9b.sh"

echo "vLLM and gateway shutdown requested together."
