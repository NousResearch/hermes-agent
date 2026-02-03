#!/usr/bin/env bash
set -euo pipefail

# Launch a local llama.cpp OpenAI-compatible server running Hermes 4.3 36B (GGUF).
#
# Requires:
# - `llama-server` installed (e.g. `brew install llama.cpp`)
#
# Note: Port choice can conflict with other local dev servers. If 8080 is already
# in use, override via `LLAMA_CPP_PORT=...`.
#
# Usage:
#   Hermes-Agent/scripts/launch_llama_cpp_hermes_4_36b.sh
#
# Override defaults:
#   LLAMA_CPP_HOST=127.0.0.1 LLAMA_CPP_PORT=8082 \
#   LLAMA_CPP_HF_REPO=NousResearch/Hermes-4.3-36B-GGUF \
#   LLAMA_CPP_HF_FILE=hermes-4_3_36b-Q4_K_M.gguf \
#   LLAMA_CPP_ALIAS=hermes-4-36b \
#   Hermes-Agent/scripts/launch_llama_cpp_hermes_4_36b.sh

HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
PORT="${LLAMA_CPP_PORT:-8080}"
HF_REPO="${LLAMA_CPP_HF_REPO:-NousResearch/Hermes-4.3-36B-GGUF}"
HF_FILE="${LLAMA_CPP_HF_FILE:-hermes-4_3_36b-Q4_K_M.gguf}"
ALIAS="${LLAMA_CPP_ALIAS:-hermes-4-36b}"

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
    echo "  LLAMA_CPP_PORT=8082 Hermes-Agent/scripts/launch_llama_cpp_hermes_4_36b.sh"
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
