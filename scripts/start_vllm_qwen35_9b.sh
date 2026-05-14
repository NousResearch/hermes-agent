#!/usr/bin/env bash
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
VENV_PATH="${HERMES_VLLM_VENV:-$HERMES_HOME/vllm-venv}"
PORT="${HERMES_VLLM_PORT:-8010}"
API_KEY="${HERMES_VLLM_API_KEY:-hermes-local}"
MODEL_ID="${HERMES_VLLM_MODEL_ID:-Qwen/Qwen3.5-9B}"
SERVED_MODEL="${HERMES_VLLM_SERVED_MODEL:-qwen3.5:9b}"
LOG_PATH="${HERMES_VLLM_LOG:-$HERMES_HOME/logs/vllm-qwen35-9b.log}"
PID_PATH="${HERMES_VLLM_PID:-$HERMES_HOME/vllm-qwen35-9b.pid}"

mkdir -p "$(dirname "$LOG_PATH")"

if [[ -f "$PID_PATH" ]]; then
  old_pid="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "vLLM already running on PID ${old_pid}"
    exit 0
  fi
  rm -f "$PID_PATH"
fi

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "Missing vLLM virtualenv: $VENV_PATH"
  exit 1
fi

cmd=$(cat <<EOF
source "$VENV_PATH/bin/activate"
export HF_HUB_ENABLE_HF_TRANSFER=1
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
exec vllm serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --served-model-name "$SERVED_MODEL" \
  --language-model-only \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 2 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --quantization bitsandbytes \
  --load-format bitsandbytes
EOF
)

setsid bash -lc "$cmd" >>"$LOG_PATH" 2>&1 < /dev/null &
new_pid=$!
echo "$new_pid" > "$PID_PATH"

echo "Started vLLM on PID $new_pid"
echo "Log: $LOG_PATH"
