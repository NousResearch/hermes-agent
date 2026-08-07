#!/usr/bin/env bash
# Discover the current Hermes install environment for architecture diagram generation.
# Outputs key-value pairs that the agent reads to customize the diagram.
# Uses only public Hermes CLI commands — no direct config file reads.
set -euo pipefail

echo "=== IDENTITY ==="
echo "OS=$(uname -s)"
echo "ARCH=$(uname -m)"
echo "USERNAME=$(whoami)"
echo "HOSTNAME=$(hostname)"

echo "=== HERMES CONFIG ==="
if command -v hermes &>/dev/null; then
  echo "HERMES_CLI=available"
  # Use hermes config to read values through the official interface
  MODEL=$(hermes config 2>/dev/null | grep -iE '^\s*default:' | head -1 | awk '{print $2}' || echo "unknown")
  PROVIDER=$(hermes config 2>/dev/null | grep -iE '^\s*provider:' | head -1 | awk '{print $2}' || echo "unknown")
  BACKEND=$(hermes config 2>/dev/null | grep -iE '^\s*backend:' | head -1 | awk '{print $2}' || echo "local")
  TTS=$(hermes config 2>/dev/null | grep -iA3 'tts:' | grep 'provider:' | head -1 | awk '{print $2}' || echo "unknown")
  echo "MODEL=${MODEL:-unknown}"
  echo "PROVIDER=${PROVIDER:-unknown}"
  echo "BACKEND=${BACKEND:-local}"
  echo "TTS=${TTS:-unknown}"
else
  echo "HERMES_CLI=unavailable"
  echo "MODEL=unknown"
  echo "PROVIDER=unknown"
  echo "BACKEND=local"
  echo "TTS=unknown"
fi

echo "=== TOOLS ==="
hermes tools list 2>/dev/null | grep -E 'enabled|disabled' | head -30 || echo "TOOLS=unavailable"
