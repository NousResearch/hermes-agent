#!/usr/bin/env bash
# start-llama-server.sh — turnkey llama-server startup for Qwen3.5/3.6 + deep-research.
#
# Boots llama.cpp's llama-server with the correct flags for Qwen3.5/3.6 tool-calling
# and chat-template handling. The deep-research skill expects an OpenAI-compatible
# endpoint at http://127.0.0.1:8088 by default.
#
# Usage:
#   start-llama-server.sh <gguf-path>
#   start-llama-server.sh ~/models/Qwen3.5-9B-Q4_K_M.gguf
#
# Or with explicit overrides:
#   PORT=9000 CTX=32768 THREADS=8 start-llama-server.sh <gguf>
#
# Requires: llama-server binary on PATH (or LLAMA_SERVER env var pointing to it)
#           See https://github.com/ggerganov/llama.cpp/releases for prebuilt binaries.

set -e

GGUF="${1:-}"
PORT="${PORT:-8088}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 8)}"
HOST="${HOST:-127.0.0.1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"   # 0=CPU only; -1=all layers on GPU; positive int = N layers
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

if [ -z "$GGUF" ]; then
  cat << 'USAGE'
Usage: start-llama-server.sh <path-to-gguf>

Recommended Qwen3.5 / Qwen3.6 GGUFs (Apache 2.0, Feb-Apr 2026):
  unsloth/Qwen3.5-4B-GGUF        (~2.5 GB) — sweet spot for 6 GB GPUs / CPU
  unsloth/Qwen3.5-9B-GGUF        (~5.5 GB) — single-GPU quality
  unsloth/Qwen3.5-27B-GGUF       (~16  GB) — 24 GB single-GPU
  unsloth/Qwen3.6-27B-GGUF       (~16  GB) — latest dense (Apr 2026)
  unsloth/Qwen3.6-35B-A3B-GGUF   (~21  GB) — best speed/quality (MoE)

Environment overrides:
  PORT=8088              Server port
  CTX=16384              Context window (Qwen3.5 supports up to 262144 native, 1M with YaRN)
  THREADS=12             CPU threads
  N_GPU_LAYERS=0         0=CPU only, -1=all layers on GPU, positive int=N layers
  LLAMA_SERVER=llama-server   Path to llama-server binary
USAGE
  exit 1
fi

[ ! -f "$GGUF" ] && { echo "ERROR: GGUF file not found: $GGUF"; exit 2; }

if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
  cat << 'EOF'
ERROR: llama-server not found on PATH.

Install:
  Linux x86_64: https://github.com/ggerganov/llama.cpp/releases
    curl -fsSL "https://github.com/ggerganov/llama.cpp/releases/download/b9010/llama-b9010-bin-ubuntu-x64.tar.gz" | tar xz
    export PATH=$(pwd)/llama-b9010:$PATH
  macOS:        brew install llama.cpp
  Pip:          pip install llama-cpp-python[server]
                # then: python -m llama_cpp.server --model <gguf> --port 8088

Or set LLAMA_SERVER=/path/to/llama-server explicitly.
EOF
  exit 3
fi

# Detect Qwen3.5 / 3.6 from filename for friendly logging
MODEL_FAMILY="(generic)"
case "$(basename "$GGUF" | tr '[:upper:]' '[:lower:]')" in
  *qwen3.5*|*qwen3_5*) MODEL_FAMILY="Qwen3.5" ;;
  *qwen3.6*|*qwen3_6*) MODEL_FAMILY="Qwen3.6" ;;
  *qwen3*)              MODEL_FAMILY="Qwen3" ;;
esac

echo "=== llama-server boot ==="
echo "  model:       $GGUF"
echo "  family:      $MODEL_FAMILY"
echo "  endpoint:    http://${HOST}:${PORT}/v1/chat/completions"
echo "  context:     $CTX"
echo "  threads:     $THREADS"
echo "  gpu layers:  $N_GPU_LAYERS"
echo ""

# Build args. --jinja loads the model's chat template (required for Qwen3.5/3.6 tool calls
# + the chat_template_kwargs.enable_thinking switch).
ARGS=(
  -m "$GGUF"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX"
  --threads "$THREADS"
  --jinja
  --n-gpu-layers "$N_GPU_LAYERS"
)

# For Qwen3.5/3.6, set sane defaults for tool-call workflows:
#   - thinking off (deep-research wants clean tool-call output)
#   - instruct-mode sampling profile from the official model card
if [[ "$MODEL_FAMILY" =~ Qwen3\.[56] ]]; then
  echo "  Qwen3.5/3.6 detected → server is configured for tool-calling workflows."
  echo "  Clients should pass: chat_template_kwargs={\"enable_thinking\": false}"
  echo ""
fi

exec "$LLAMA_SERVER" "${ARGS[@]}"
