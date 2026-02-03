#!/usr/bin/env bash
set -euo pipefail

# Launch a local llama.cpp OpenAI-compatible server running GLM-4.7-Flash (GGUF).
#
# Requires:
# - `llama-server` installed (e.g. `brew install llama.cpp`)
#
# Default settings are chosen to avoid clashing with Atropos sandbox_server
# (which commonly uses port 8080 in local dev).
#
# Usage:
#   Hermes-Agent/scripts/launch_llama_cpp_glm47_flash.sh
#
# Override defaults:
#   LLAMA_CPP_HOST=127.0.0.1 LLAMA_CPP_PORT=8082 \
#   LLAMA_CPP_HF_REPO=ggml-org/GLM-4.7-Flash-GGUF \
#   LLAMA_CPP_HF_FILE=GLM-4.7-Flash-Q4_K.gguf \
#   Hermes-Agent/scripts/launch_llama_cpp_glm47_flash.sh

HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
PORT="${LLAMA_CPP_PORT:-8080}"
HF_REPO="${LLAMA_CPP_HF_REPO:-ggml-org/GLM-4.7-Flash-GGUF}"
HF_FILE="${LLAMA_CPP_HF_FILE:-GLM-4.7-Flash-Q4_K.gguf}"
ALIAS="${LLAMA_CPP_ALIAS:-glm-4.7-flash}"

if ! command -v llama-server >/dev/null 2>&1; then
  echo "Error: llama-server not found in PATH."
  echo "Install via Homebrew: brew install llama.cpp"
  exit 1
fi

echo "Launching llama.cpp server..."
echo "  host:  $HOST"
echo "  port:  $PORT"
echo "  repo:  $HF_REPO"
echo "  file:  $HF_FILE"
echo "  alias: $ALIAS"
echo
echo "Suggested env vars for Hermes/Atropos integration:"
echo "  export ATROPOS_SERVER_BASE_URL=http://${HOST}:${PORT}"
echo "  export ATROPOS_SERVER_MODEL=${ALIAS}"
echo "  export ATROPOS_SERVER_API_KEY=local"
echo

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Error: port $PORT is already in use."
    echo "Pick a different port, e.g.:"
    echo "  LLAMA_CPP_PORT=8082 Hermes-Agent/scripts/launch_llama_cpp_glm47_flash.sh"
    exit 1
  fi
fi

exec llama-server \
  --host "$HOST" \
  --port "$PORT" \
  --hf-repo "$HF_REPO" \
  --hf-file "$HF_FILE" \
  --alias "$ALIAS" \
  -c 32768 \
  -n -1
