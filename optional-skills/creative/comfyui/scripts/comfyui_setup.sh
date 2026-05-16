#!/usr/bin/env bash
# Initialize a comfyui-skill workspace directory.
# Usage: bash scripts/comfyui_setup.sh [WORKSPACE_DIR]
#
# Creates the workspace, adds a default local server config,
# and verifies the connection.

set -euo pipefail

WORKSPACE="${1:-$HOME/.hermes/comfyui}"
COMFY="${COMFY:-uvx --from comfyui-skill-cli comfyui-skill}"

echo "==> Initializing ComfyUI skill workspace at: $WORKSPACE"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# If config.json doesn't exist, create it with a default local server
if [ ! -f config.json ]; then
    echo "==> Creating default config (local server at 127.0.0.1:8188)"
    $COMFY --json server add --id local --url http://127.0.0.1:8188 --name "Local ComfyUI"
    echo "==> Config created: $WORKSPACE/config.json"
else
    echo "==> config.json already exists, skipping"
fi

# Verify connection
echo "==> Checking server connection..."
if $COMFY --json server status 2>/dev/null | grep -q '"online"'; then
    echo "==> ComfyUI is reachable!"
    $COMFY --json server stats 2>/dev/null || true
else
    echo "==> ComfyUI is not reachable at the configured URL."
    echo "    Start ComfyUI first, or update the server URL:"
    echo "    cd $WORKSPACE && $COMFY server add --id local --url <YOUR_URL>"
    echo ""
    echo "    Install ComfyUI: https://docs.comfy.org/installation"
fi

echo ""
echo "==> Workspace ready: $WORKSPACE"
echo "    Always cd here before running comfyui-skill commands."
