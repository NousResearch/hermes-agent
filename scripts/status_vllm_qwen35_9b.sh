#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PORT="${HERMES_VLLM_PORT:-8010}"
PID_PATH="${HERMES_VLLM_PID:-$HERMES_HOME/vllm-qwen35-9b.pid}"
LOG_PATH="${HERMES_VLLM_LOG:-$HERMES_HOME/logs/vllm-qwen35-9b.log}"

if [[ -f "$PID_PATH" ]]; then
  pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "vLLM is running on PID ${pid}"
  else
    echo "vLLM pid file exists, but process is not alive"
  fi
else
  echo "No vLLM pid file found"
fi

echo "Health check:"
curl -fsS "http://127.0.0.1:${PORT}/health" || true
echo
echo "Models:"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" || true
echo
echo "Log tail:"
tail -n 20 "$LOG_PATH" 2>/dev/null || true
