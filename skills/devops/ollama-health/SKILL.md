---
name: ollama-health
description: Monitor and manage local Ollama models — check loaded models, VRAM usage, pull/remove models, and diagnose issues
version: 1.0.0
author: het4rk
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Ollama, LLM, Local, Models, DevOps, Monitoring]
    requires_toolsets: [terminal]
---

# Ollama Health Monitor

Monitor and manage your local Ollama instance — check which models are running, how much VRAM they consume, pull or remove models, and diagnose slow inference.

## When to Use

- Verify Ollama is running and its API is responsive
- See which models are currently loaded into VRAM
- Inventory all installed models and their disk sizes
- Pull new models or update existing ones
- Diagnose slow or timed-out inference requests
- Free disk space by removing unused models
- Restart the Ollama service after config changes

## Quick Reference

| Command | Purpose |
|---|---|
| `ollama ps` | Show running models with VRAM usage |
| `ollama list` | List all installed models with sizes |
| `curl http://localhost:11434/api/tags` | API: list installed models (JSON) |
| `curl http://localhost:11434/api/ps` | API: list running models (JSON) |
| `ollama pull <model>` | Download or update a model |
| `ollama rm <model>` | Remove an installed model |
| `ollama show <model>` | Show model details (params, quantization, size) |
| `ollama serve` | Start the Ollama server manually |
| `ollama version` | Print Ollama version |

---

## Procedures

### 1. Health Check

Verify that the Ollama process is running and the API is responsive:

```bash
# Check if Ollama process is running
pgrep -x ollama && echo "Ollama process: running" || echo "Ollama process: not found"

# Confirm API responds
curl -sf http://localhost:11434/ && echo "API: OK" || echo "API: unreachable"

# Full version check via API
curl -s http://localhost:11434/api/version | python3 -m json.tool
```

If the API is unreachable, start Ollama:
```bash
ollama serve &       # foreground (dev)
# or use the service manager — see Restart section below
```

---

### 2. Model Inventory

List what is installed vs. what is currently loaded in VRAM:

```bash
# All installed models (name, size, modified date)
ollama list

# Models currently loaded into VRAM
ollama ps
```

Via API (useful inside scripts):
```bash
# Installed models
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    size_gb = m['size'] / 1e9
    print(f\"{m['name']:<45} {size_gb:.2f} GB\")
"

# Running models + VRAM
curl -s http://localhost:11434/api/ps | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if not models:
    print('No models currently loaded in VRAM.')
else:
    for m in models:
        vram_gb = m.get('size_vram', 0) / 1e9
        print(f\"{m['name']:<45} VRAM: {vram_gb:.2f} GB\")
"
```

For a formatted full report, run the bundled script:
```bash
python3 skills/devops/ollama-health/scripts/ollama_status.py
```

---

### 3. Diagnose Slow Inference

Slow first-response times are almost always caused by the model not being pre-loaded.

**Step 1 — Is the model loaded?**
```bash
ollama ps
```
If the target model does not appear, it will be loaded on the next request (cold-start latency is normal).

**Step 2 — Check VRAM pressure**
```bash
# macOS (Apple Silicon)
sudo powermetrics --samplers gpu_power -n 1 2>/dev/null | grep -i "gpu active\|gpu memory"

# Linux (NVIDIA)
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader

# Linux (AMD)
rocm-smi --showmeminfo vram 2>/dev/null
```

If available VRAM is low, unload unused models:
```bash
# Unload a model by setting keep_alive to 0
curl -s http://localhost:11434/api/generate \
  -d "{\"model\": \"<model-to-unload>\", \"keep_alive\": 0}" > /dev/null
```

**Step 3 — Check quantization level**
```bash
ollama show <model>
# Look for "quantization" in the output — lower Q (Q2/Q3) is faster but less accurate
```

**Step 4 — Review Ollama logs for errors**
```bash
# macOS
tail -n 100 ~/.ollama/logs/server.log

# Linux (systemd)
journalctl -u ollama -n 100 --no-pager
```

---

### 4. Pull / Update a Model

```bash
# Pull a model (or update to latest)
ollama pull llama3.2

# Pull a specific version/tag
ollama pull mistral:7b-instruct-q4_K_M

# Track pull progress
ollama pull nomic-embed-text
```

To pull and immediately verify:
```bash
MODEL="llama3.2"
ollama pull "$MODEL" && ollama show "$MODEL"
```

---

### 5. Clean Up Unused Models

Free disk space by removing models you no longer use:

```bash
# List all installed models with sizes
ollama list

# Remove a specific model
ollama rm mistral:7b

# Remove multiple models
for model in phi3:mini orca-mini gemma:2b; do
  ollama rm "$model" && echo "Removed $model"
done
```

To find the largest installed models:
```bash
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
models = json.load(sys.stdin).get('models', [])
for m in sorted(models, key=lambda x: x['size'], reverse=True):
    print(f\"{m['size']/1e9:.2f} GB  {m['name']}\")
"
```

---

### 6. Restart Ollama Service

**macOS (launchctl)**
```bash
# Check if the launchd service is loaded
launchctl list | grep ollama

# Stop Ollama
launchctl stop com.ollama.ollama 2>/dev/null || pkill ollama

# Start Ollama (via app or launchd)
open -a Ollama                          # if installed as macOS app
# or
launchctl start com.ollama.ollama       # if registered with launchd
# or
ollama serve &                          # manual start
```

**Linux (systemd)**
```bash
# Check status
systemctl status ollama

# Restart
sudo systemctl restart ollama

# Enable on boot
sudo systemctl enable ollama

# View recent logs
journalctl -u ollama -n 50 --no-pager
```

---

## Bundled Script

`scripts/ollama_status.py` prints a full status report covering:
- API reachability and Ollama version
- All installed models with sizes
- Models currently loaded in VRAM

```bash
python3 skills/devops/ollama-health/scripts/ollama_status.py

# Or with a custom host:
OLLAMA_HOST=http://192.168.1.10:11434 python3 skills/devops/ollama-health/scripts/ollama_status.py
```

---

## Pitfalls

- **Cold-start latency is expected.** The first inference call loads the model from disk into VRAM, which can take several seconds to over a minute for large models. Subsequent calls within the `keep_alive` window (default: 5 minutes) are fast.
- **MoE models can time out on cold start.** Mixture-of-Experts architectures like `qwen2.5:72b-instruct-a22b` or Hermes models using MoE routing load a large number of expert weights and may appear to hang for 60–120 seconds. Increase your client timeout or pre-warm the model with a dummy prompt.
- **VRAM limits depend on hardware.** On Apple Silicon, system RAM is shared with the GPU — ensure you have enough free unified memory. On NVIDIA/AMD, check `nvidia-smi` / `rocm-smi` for accurate VRAM usage.
- **Multiple models can share VRAM** if they fit, but Ollama will evict the least-recently-used model automatically when memory is needed.
- **`ollama ps` shows zero entries** when no model has been used recently — models are evicted after the `keep_alive` timeout.

---

## Verification Steps

1. `curl -sf http://localhost:11434/ && echo OK` — API responds
2. `ollama list` — returns at least one model (or empty list with no error)
3. `ollama ps` — runs without error (may show empty if no model loaded)
4. `python3 skills/devops/ollama-health/scripts/ollama_status.py` — status report renders cleanly
5. `ollama pull <small-model>` (e.g. `ollama pull tinyllama`) — pull succeeds end-to-end
