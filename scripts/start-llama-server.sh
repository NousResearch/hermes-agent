#!/usr/bin/env bash
# start-llama-server.sh — turnkey llama-server startup that lines up with
# Hermes' built-in `llama-cpp` provider.
#
# Boots llama.cpp's llama-server on the same host:port that the
# `plugins/model-providers/llama-cpp/` profile defaults to
# (http://127.0.0.1:8088/v1). After this script is running,
#
#   hermes chat --provider llama-cpp --model <whatever>
#
# works with no further config. To override the port/host, set PORT / HOST
# below AND export LLAMA_CPP_BASE_URL=http://${HOST}:${PORT}/v1 so Hermes
# follows.
#
# Usage:
#   start-llama-server.sh <gguf-path>
#   start-llama-server.sh ~/models/some-model.gguf
#
# Or with explicit overrides:
#   PORT=9000 CTX=32768 THREADS=8 start-llama-server.sh <gguf>
#
# Requires: llama-server binary on PATH (or LLAMA_SERVER env var pointing to it)
#           See https://github.com/ggml-org/llama.cpp/releases for prebuilt binaries.

set -e

GGUF="${1:-}"
PORT="${PORT:-8088}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
HOST="${HOST:-127.0.0.1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"   # 0=CPU only; -1=all layers on GPU; positive int = N layers
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

if [ -z "$GGUF" ]; then
  cat << 'USAGE'
Usage: start-llama-server.sh <path-to-gguf>

Environment overrides:
  PORT=8088              Server port (matches Hermes llama-cpp provider default)
  CTX=16384              Context window
  THREADS=12             CPU threads
  N_GPU_LAYERS=0         0=CPU only, -1=all layers on GPU, positive int=N layers
  LLAMA_SERVER=llama-server   Path to llama-server binary

After boot:
  hermes chat --provider llama-cpp --model <model-id-from-/v1/models>
USAGE
  exit 1
fi

[ ! -f "$GGUF" ] && { echo "ERROR: GGUF file not found: $GGUF"; exit 2; }

if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
  cat << 'EOF'
ERROR: llama-server not found on PATH.

Install:
  Linux x86_64: https://github.com/ggml-org/llama.cpp/releases
  macOS:        brew install llama.cpp
  Pip:          pip install llama-cpp-python[server]
                # then: python -m llama_cpp.server --model <gguf> --port 8088

Or set LLAMA_SERVER=/path/to/llama-server explicitly.
EOF
  exit 3
fi

echo "=== llama-server boot ==="
echo "  model:       $GGUF"
echo "  endpoint:    http://${HOST}:${PORT}/v1"
echo "  context:     $CTX"
echo "  threads:     $THREADS"
echo "  gpu layers:  $N_GPU_LAYERS"
echo ""

# Hermes alignment hint
EXPECTED_BASE_URL="http://127.0.0.1:8088/v1"
ACTUAL_BASE_URL="http://${HOST}:${PORT}/v1"
if [ "$ACTUAL_BASE_URL" != "$EXPECTED_BASE_URL" ]; then
  echo "  NOTE: Hermes' llama-cpp provider defaults to ${EXPECTED_BASE_URL}."
  echo "        Override with: export LLAMA_CPP_BASE_URL=${ACTUAL_BASE_URL}"
  echo ""
else
  echo "  Hermes: ready — run 'hermes chat --provider llama-cpp'"
  echo ""
fi

# --jinja loads the model's chat template, which is required for any GGUF
# that needs Jinja-rendered tool-call messages.
ARGS=(
  -m "$GGUF"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX"
  --threads "$THREADS"
  --jinja
  --n-gpu-layers "$N_GPU_LAYERS"
)

exec "$LLAMA_SERVER" "${ARGS[@]}"
