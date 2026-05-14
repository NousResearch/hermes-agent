#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PID_PATH="${HERMES_VLLM_PID:-$HERMES_HOME/vllm-qwen35-9b.pid}"

if [[ ! -f "$PID_PATH" ]]; then
  echo "No pid file at $PID_PATH"
  exit 0
fi

pid="$(cat "$PID_PATH" 2>/dev/null || true)"
if [[ -z "$pid" ]]; then
  rm -f "$PID_PATH"
  echo "Pid file was empty; cleaned up."
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped vLLM PID $pid"
else
  echo "Process $pid was not running"
fi

rm -f "$PID_PATH"
