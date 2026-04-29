#!/usr/bin/env bash
# ComfyUI Setup — Install, launch, and verify using the official comfy-cli.
# Usage: bash scripts/comfyui_setup.sh [--nvidia|--amd|--m-series|--cpu]
#
# Prerequisites: Python 3.10+, pip
# What it does:
#   1. Installs comfy-cli (if not present)
#   2. Disables analytics tracking
#   3. Installs ComfyUI + ComfyUI-Manager
#   4. Launches server in background
#   5. Verifies server is reachable

set -euo pipefail

GPU_FLAG="${1:---nvidia}"  # Default to NVIDIA

echo "==> ComfyUI Setup"
echo "    GPU flag: $GPU_FLAG"
echo ""

# Step 1: Install comfy-cli
if command -v comfy >/dev/null 2>&1; then
    echo "==> comfy-cli already installed: $(comfy -v 2>/dev/null || echo 'unknown version')"
else
    echo "==> Installing comfy-cli..."
    pip install comfy-cli
fi

# Step 2: Disable tracking (avoid interactive prompt)
echo "==> Disabling analytics tracking..."
comfy --skip-prompt tracking disable 2>/dev/null || true

# Step 3: Install ComfyUI
if comfy which 2>/dev/null | grep -q "ComfyUI"; then
    echo "==> ComfyUI already installed at: $(comfy which 2>/dev/null)"
else
    echo "==> Installing ComfyUI ($GPU_FLAG)..."
    comfy --skip-prompt install $GPU_FLAG
fi

# Step 4: Launch in background
echo "==> Launching ComfyUI in background..."
comfy launch --background 2>/dev/null || {
    echo "==> Background launch failed. Trying foreground check..."
    echo "    You may need to run: comfy launch"
    exit 1
}

# Step 5: Wait for server to be ready
echo "==> Waiting for server..."
MAX_WAIT=30
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
        echo "==> Server is running!"
        curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool 2>/dev/null || true
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "==> Server did not start within ${MAX_WAIT}s."
    echo "    Check logs with: comfy launch (foreground) to see errors."
    exit 1
fi

echo ""
echo "==> Setup complete!"
echo "    Server: http://127.0.0.1:8188"
echo "    Web UI: http://127.0.0.1:8188 (open in browser)"
echo "    Stop:   comfy stop"
echo ""
echo "    Next steps:"
echo "    - Download a model: comfy model download --url <URL> --relative-path models/checkpoints"
echo "    - Run a workflow:   python3 scripts/run_workflow.py --workflow <file.json> --args '{...}'"
