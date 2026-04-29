#!/usr/bin/env bash
# Initialize a comfyui-skill workspace directory.
# Usage: bash scripts/comfyui_setup.sh [WORKSPACE_DIR] [--url COMFYUI_URL]
#
# Creates the workspace, adds a default local server config,
# and verifies the connection.

set -euo pipefail

WORKSPACE="${1:-$HOME/.hermes/comfyui}"
COMFYUI_URL="http://127.0.0.1:8188"

# Parse optional --url flag
for arg in "$@"; do
    case "$prev" in
        --url) COMFYUI_URL="$arg"; prev="";;
        *) prev="$arg";;
    esac
done

# Detect CLI: prefer uvx, fall back to direct command
if command -v uvx &>/dev/null; then
    COMFY="uvx --from comfyui-skill-cli comfyui-skill"
elif command -v comfyui-skill &>/dev/null; then
    COMFY="comfyui-skill"
else
    echo "ERROR: Neither uvx nor comfyui-skill found."
    echo "Install one of:"
    echo "  pip install uv          # then uvx handles everything"
    echo "  pip install comfyui-skill-cli"
    exit 1
fi

echo "==> Initializing ComfyUI skill workspace at: $WORKSPACE"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Create config if missing
if [ ! -f config.json ]; then
    echo "==> Creating default config (server at $COMFYUI_URL)"
    $COMFY --json server add --id local --url "$COMFYUI_URL" --name "Local ComfyUI"
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
    echo "==> ComfyUI is not reachable at $COMFYUI_URL"
    echo "    Start ComfyUI first, or re-run with a different URL:"
    echo "    bash scripts/comfyui_setup.sh $WORKSPACE --url http://YOUR_HOST:PORT"
    echo ""
    echo "    Install ComfyUI: https://docs.comfy.org/installation"
fi

echo ""
echo "==> Workspace ready: $WORKSPACE"
echo "    Always cd here before running comfyui-skill commands."
