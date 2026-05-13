---
name: hermes-gateway-openwebui
description: "Set up Hermes Agent gateway with Open WebUI: install, configure providers, expose models in /v1/models, stream reasoning/thinking content, and apply patches. Works on Windows and WSL2 Ubuntu."
version: 1.0.0
author: syncoe6368
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, gateway, open-webui, setup, reasoning, models, dashboard, wsl2]
    related_skills: [hermes-agent]
---

# Hermes Gateway + Open WebUI Setup

Complete guide to deploy Hermes Agent's API server (OpenAI-compatible gateway) and connect it to Open WebUI as a frontend dashboard. Covers both **Windows native** and **WSL2 Ubuntu** environments.

## What You Get

- Open WebUI at `http://localhost:8080` with a model selector showing all configured models
- Hermes gateway at `http://localhost:8642/v1` acting as an OpenAI-compatible proxy
- Reasoning/thinking content rendered as collapsible blocks (GLM, DeepSeek, Qwen, etc.)
- All models from `config.yaml` providers exposed in the model selector

---

## Architecture

```
Open WebUI (port 8080)
    │
    │  OPENAI_API_BASE_URL=http://localhost:8642/v1
    │
    ▼
Hermes Gateway (port 8642)
    │
    ├── /v1/models          → lists all provider models
    ├── /v1/chat/completions → streams content + reasoning_content
    │
    ▼
Provider APIs (Z.ai GLM, OpenRouter, NVIDIA NIM, etc.)
```

---

## Step 1: Install Hermes Agent

### Linux / WSL2 Ubuntu / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex
```

### Verify

```bash
hermes --version
hermes doctor
```

---

## Step 2: Configure Providers

Edit `~/.hermes/config.yaml` (Linux/WSL2: `$HOME/.hermes/config.yaml`, Windows: `C:\Users\<user>\.hermes\config.yaml`).

Example provider configuration with GLM models:

```yaml
model:
  default: zai/glm-5.1
  provider: zai
providers:
  zai-proxy:
    name: Z.ai GLM
    api: https://api.z.ai/api/anthropic
    key_env: ZAI_API_KEY
    api_mode: anthropic_messages
    default_model: glm-5-turbo
    models:
      turbo: glm-5-turbo
      full: glm-5.1
      light: glm-4.7
  openrouter-gemma:
    name: OpenRouter - Gemma 4 31b
    api: https://openrouter.ai/api/v1
    key_env: OPENROUTER_API_KEY
    api_mode: chat_completions
    default_model: google/gemma-4-31b-it:free
  nvidia-nim:
    name: NVIDIA NIM
    api: https://integrate.api.nvidia.com/v1
    key_env: NVIDIA_API_KEY
    api_mode: chat_completions
    default_model: minimaxai/minimax-m2.7
    models:
      minimax-m27: minimaxai/minimax-m2.7
      minimax-m25: minimaxai/minimax-m2.5
fallback_providers:
- provider: nvidia-nim
  model: minimaxai/minimax-m2.7
- provider: openrouter-gemma
  model: google/gemma-4-31b-it:free
```

### API Keys

Edit `~/.hermes/.env`:

```bash
# Z.ai (GLM models)
ZAI_API_KEY="your-zai-api-key"

# OpenRouter
OPENROUTER_API_KEY="sk-or-v1-..."

# NVIDIA NIM
NVIDIA_API_KEY="nvapi-..."

# API Server (gateway auth)
API_SERVER_ENABLED=true
API_SERVER_KEY="your-secret-api-key-for-open-webui"
API_SERVER_CORS_ORIGINS=http://localhost:3456

# Provider timeouts
HERMES_STREAM_STALE_TIMEOUT=600
HERMES_API_TIMEOUT=1800
```

---

## Step 3: Install Open WebUI

```bash
pip install open-webui
```

---

## Step 4: Start the Gateway

```bash
# Start the Hermes gateway (Discord, API server, etc.)
hermes gateway run
```

The gateway binds `127.0.0.1:8642` by default. Verify:

```bash
curl -s -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8642/v1/models
```

---

## Step 5: Start Open WebUI

Create a startup script:

### Linux / WSL2 Ubuntu: `~/start-open-webui.sh`

```bash
#!/bin/bash
export OPENAI_API_BASE_URL=http://localhost:8642/v1
export OPENAI_API_KEY="your-secret-api-key-for-open-webui"
export DATA_DIR="$HOME/.open-webui"

echo "Starting Open WebUI on http://localhost:8080"
echo "Connected to Hermes API: http://localhost:8642/v1"

open-webui serve --port 8080
```

```bash
chmod +x ~/start-open-webui.sh
```

### Windows: `start-open-webui.cmd`

```cmd
@echo off
set OPENAI_API_BASE_URL=http://localhost:8642/v1
set OPENAI_API_KEY=your-secret-api-key-for-open-webui
set DATA_DIR=C:\Users\%USERNAME%\.open-webui

title Open WebUI - Hermes Dashboard

echo Starting Open WebUI...
echo Dashboard: http://localhost:8080
echo Connected to Hermes API: http://localhost:8642/v1
echo.

open-webui.exe serve --port 8080
pause
```

Run it, then open `http://localhost:8080`.

---

## Step 6: Apply Patches (Expose Models + Reasoning Content)

The upstream Hermes gateway has two limitations when used with Open WebUI:

1. **`/v1/models` only returns a single model** — Open WebUI can't show a model selector
2. **Reasoning/thinking content is silently dropped** — no collapsible thinking blocks for GLM/DeepSeek

These are fixed by patches in the `feat/api-server-expose-provider-models-and-reasoning` branch. Apply them to your installed package:

### Option A: Patch script (recommended for repeated use)

Save this as `patch-hermes-gateway.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Find the installed gateway package
GATEWAY_FILE="$(python -c "
import importlib.metadata, os, pathlib
dist = importlib.metadata.distribution('hermes-agent')
# Walk RECORD to find api_server.py
for line in dist.read_text('RECORD').splitlines():
    if 'gateway/platforms/api_server.py' in line:
        rel = line.split(',')[0]
        print(pathlib.Path(dist._path).parent / rel)
        break
" 2>/dev/null)"

if [ -z "$GATEWAY_FILE" ] || [ ! -f "$GATEWAY_FILE" ]; then
    echo "ERROR: Cannot find installed gateway/platforms/api_server.py"
    exit 1
fi

echo "Target: $GATEWAY_FILE"

# Check if already patched
if grep -q "__reasoning__" "$GATEWAY_FILE" 2>/dev/null; then
    echo "✓ Patches already applied"
    exit 0
fi

# Apply from source repo if available
SOURCE_REPO="${1:-./hermes-agent}"
BRANCH="feat/api-server-expose-provider-models-and-reasoning"

if [ -d "$SOURCE_REPO/.git" ]; then
    cd "$SOURCE_REPO"
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        echo "Extracting patched file from branch '$BRANCH'..."
        git show "$BRANCH:gateway/platforms/api_server.py" > "$GATEWAY_FILE"
        echo "✓ Patched file written"
    else
        echo "ERROR: Branch '$BRANCH' not found"
        exit 1
    fi
else
    echo "ERROR: Source repo not found at $SOURCE_REPO"
    echo "Clone it: git clone https://github.com/NousResearch/hermes-agent.git"
    exit 1
fi

# Verify
if grep -q "__reasoning__" "$GATEWAY_FILE" 2>/dev/null; then
    echo "✓ SUCCESS — patches applied. Restart gateway: hermes gateway restart"
else
    echo "✗ FAILED"
    exit 1
fi
```

Usage:

```bash
# Clone the repo (one-time)
git clone https://github.com/NousResearch/hermes-agent.git ~/hermes-agent

# Apply patches
bash patch-hermes-gateway.sh ~/hermes-agent

# Re-apply after upgrades
pip install --upgrade hermes-agent
bash patch-hermes-gateway.sh ~/hermes-agent
```

### Option B: Manual file copy

```bash
# Clone the branch
git clone -b feat/api-server-expose-provider-models-and-reasoning \
    https://github.com/syncoe6368/hermes-agent.git /tmp/hermes-patch

# Find and replace the installed file
python -c "
import importlib.metadata, pathlib
dist = importlib.metadata.distribution('hermes-agent')
for line in dist.read_text('RECORD').splitlines():
    if 'gateway/platforms/api_server.py' in line:
        rel = line.split(',')[0]
        target = pathlib.Path(dist._path).parent / rel
        print(target)
"  # Copy the patched file to the path printed above
```

### What the patches do

1. **`_get_exposed_models()`** — reads all models from `config.yaml` providers and advertises them in `/v1/models`
2. **`_on_reasoning()` callback** — wires the agent's reasoning/thinking output through to the SSE stream
3. **`delta.reasoning_content`** — reasoning chunks emitted in standard OpenAI format so Open WebUI renders collapsible thinking blocks

---

## WSL2 Ubuntu Specifics

If running the gateway on Windows but using WSL2 for development:

1. **Install Hermes in WSL2** (recommended for Linux-native path handling):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
   ```

2. **Access Windows host from WSL2**: Use `$(hostname).local` or the WSL2 IP:
   ```bash
   # If Open WebUI runs on Windows host
   export OPENAI_API_BASE_URL=http://$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):8642/v1

   # Or if everything runs in WSL2
   export OPENAI_API_BASE_URL=http://localhost:8642/v1
   ```

3. **Access WSL2 from Windows host**: Use `localhost` — WSL2 ports are auto-forwarded.

4. **Shared config**: Hermes in WSL2 uses `~/.hermes/` (Linux home). Config is NOT shared with Windows `C:\Users\<user>\.hermes\`. Copy or symlink as needed.

---

## Verify Everything Works

```bash
# 1. Gateway is running
curl -s http://localhost:8642/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY" | python -m json.tool

# Should show all configured models:
# hermes-agent, glm-5-turbo, glm-5.1, glm-4.7, etc.

# 2. Reasoning content streams correctly
curl -s -N -X POST http://localhost:8642/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-turbo","messages":[{"role":"user","content":"What is 17*23?"}],"stream":true}' | head -10

# Should show both reasoning_content and content chunks

# 3. Open WebUI
# Open http://localhost:8080
# - Select a GLM model from the dropdown
# - Send a message → thinking blocks should appear
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Gateway not binding port 8642 | Check `API_SERVER_KEY` is set in `.env` — gateway refuses to start without it |
| Open WebUI shows no models | Ensure patches are applied; restart gateway |
| No thinking blocks in UI | Verify patches applied (`grep __reasoning__` on api_server.py); check model supports reasoning |
| `model_access_denied` errors | API key may not have access to that model on the provider; try a different model |
| Gateway crashes on start | Run `hermes gateway run` in foreground to see errors; check `~/.hermes/.env` for typos |

---

## File Locations Reference

| File | Linux / WSL2 | Windows |
|------|-------------|---------|
| Hermes config | `~/.hermes/config.yaml` | `C:\Users\<user>\.hermes\config.yaml` |
| Hermes env | `~/.hermes/.env` | `C:\Users\<user>\.hermes\.env` |
| Open WebUI data | `~/.open-webui/` | `C:\Users\<user>\.open-webui\` |
| Patch target | `<venv>/.../gateway/platforms/api_server.py` | `<python>/Lib/site-packages/gateway/platforms/api_server.py` |
| Patch source | `~/hermes-agent/` (cloned repo) | `D:\projects\hermes-agent\` (cloned repo) |
