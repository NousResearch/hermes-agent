---
name: llama-model-manager
description: Start, stop, health-check, and monitor local GGUF language models via llama.cpp server. Covers model loading, think mode toggling, inference performance monitoring, and multi-model switching.
version: 1.0.0
author: ligl0325
license: MIT
platforms:
  - linux
  - macos
metadata:
  hermes:
    tags:
      - llama.cpp
      - gguf
      - local-model
      - llm
      - mlops
    category: mlops
triggers:
  - start llama model
  - stop llama model
  - switch model
  - model health
  - llama performance
  - change think mode
toolsets:
  - terminal
---

# llama.cpp Local Model Manager

This skill enables an agent to manage local GGUF language models running via the `llama.cpp` server binary (`llama-server`). It covers the full lifecycle: discovering available models, starting and stopping server instances, checking health and performance, toggling reasoning mode, and switching between loaded models.

The agent must execute real shell commands via the `terminal` toolset to interact with the filesystem, process manager, and network.

## Safety Guardrails

- **Never start a model that would exceed 80% of available RAM.** Before loading a GGUF model, check free memory with `free -h` and estimate the model's footprint (file size × ~1.5 for runtime overhead). If the estimate exceeds 80% of available RAM, refuse to start and explain why.
- **Always check if the target port is in use before binding.** Use `ss -tlnp | grep :<port>` or `lsof -i :<port>` to verify the port is free. If it is occupied, either select a different port or warn the user.
- **Warn the user about CPU-only inference speed.** If no GPU acceleration is detected (check `nvidia-smi` on Linux, `sysctl` for Metal on macOS), inform the user that CPU-only inference typically yields 5–20 tokens/second for 1B–7B parameter models.

## Phase 1: Discover & Check

Before taking any action, gather system state.

### System resources

```bash
# Available memory (look at "available" column)
free -h

# Number of CPU threads
nproc

# GPU check (Linux)
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"

# GPU check (macOS)
sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -qi "apple" && echo "Apple Silicon (Metal available)"
```

### Find existing llama-server processes

```bash
# Check for any running llama-server instances
ps aux | grep llama-server | grep -v grep

# Get PIDs and ports (if --port was used)
ps aux | grep "llama-server" | grep -v grep | awk '{for(i=1;i<=NF;i++) if($i=="--port") print $(i+1),$2}'
```

### Find available GGUF models

Search common locations in order of likelihood:

```bash
# Check common model directories
for dir in ~/models ~/downloads ~/.cache/llama.cpp /models /opt/models; do
  if [ -d "$dir" ]; then
    echo "=== $dir ==="
    find "$dir" -name "*.gguf" -type f 2>/dev/null | head -20
  fi
done
```

### Check port availability

```bash
# Default llama-server port is 8080; check it and common alternatives
for port in 8080 8081 8082 8083 8084; do
  if ss -tlnp | grep -q ":$port "; then
    echo "Port $port is IN USE"
  else
    echo "Port $port is FREE"
  fi
done
```

## Phase 2: Start Model

### Select the right model

1. Parse the list of discovered `.gguf` files.
2. For each model, estimate its file size in GB: `ls -lh <model>`.
3. Apply the rule of thumb: **model file size (GB) × 1.5 ≤ available free RAM (GB)**. If over 80% RAM, skip that model.
4. Prefer the largest model that fits the budget.

### Start the server

```bash
# Basic start command
llama-server \
  -m /path/to/model.gguf \
  -c 4096 \                    # Context window (tokens)
  --port 8080 \                # Server port
  -t $(nproc) \                # Thread count (all available)
  --no-kv-offload \            # Keep KV cache in system RAM
  --mlock                      # Lock model in RAM (prevents swapping)
```

**Optional flags:**
- `--reasoning on` — Enable chain-of-thought / reasoning mode (if supported by model).
- `--reasoning off` — Disable reasoning mode (default).
- `--no-mmap` — Disable memory-mapped model loading (use if you want full control over memory usage).
- `--temp 0.7` — Set sampling temperature.

### Verify the server is healthy

After starting, wait a moment then check:

```bash
# Health check via the llama.cpp API
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health

# If health endpoint returns 200, the model is loaded and ready
# Alternative: check the v1/models endpoint
curl -s http://localhost:8080/v1/models | head -100
```

**Do not declare success until the health endpoint returns HTTP 200.** If it does not respond within 60 seconds, report failure and suggest checking logs.

### Complete start example

```bash
# 1. Check resources
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
echo "Available RAM: ${FREE_RAM}GB"

# 2. Find model
MODEL=$(find ~/models -name "*.gguf" -type f | head -1)
MODEL_GB=$(ls -lh "$MODEL" | awk '{print $5}' | sed 's/G//')
echo "Model: $MODEL (${MODEL_GB}GB)"

# 3. Check fit
if (( $(echo "$MODEL_GB * 1.5 > $FREE_RAM * 0.8" | bc -l) )); then
  echo "ERROR: Model too large for available RAM"
  exit 1
fi

# 4. Start
llama-server -m "$MODEL" -c 4096 --port 8080 -t $(nproc) --no-kv-offload --mlock &

# 5. Wait for health
for i in $(seq 1 30); do
  sleep 2
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null | grep -q 200; then
    echo "Model loaded successfully"
    break
  fi
done
```

## Phase 3: Monitor & Debug

### Inference performance

Query the OpenAI-compatible API to check timing:

```bash
# List loaded models with metadata
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# Check server metrics (if available)
curl -s http://localhost:8080/metrics 2>/dev/null | head -50

# Measure inference speed with a simple prompt
time curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100, "temperature": 0}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Tokens generated:', len(d.get('choices',[{}])[0].get('text','').split())); print('Usage:', json.dumps(d.get('usage',{}), indent=2))"
```

### Memory and CPU usage

```bash
# Check llama-server memory usage
ps -o pid,%cpu,%mem,rss,comm -p $(pgrep -d, llama-server) 2>/dev/null

# Real-time monitoring
htop -p $(pgrep -d, llama-server) 2>/dev/null || top -p $(pgrep -d, llama-server | tr ',' ' ') -n 1

# Overall system memory impact
free -h
```

### Think mode toggle

If the user wants to enable or disable reasoning/think mode:

```bash
# 1. Get the current model path and port from the running process
MODEL_PATH=$(ps aux | grep llama-server | grep -v grep | grep -oP '\-m \K\S+')
PORT=$(ps aux | grep llama-server | grep -v grep | grep -oP '\-\-port \K\d+')
THREADS=$(ps aux | grep llama-server | grep -v grep | grep -oP '\-t \K\d+')

# 2. Stop the current server gracefully
kill $(pgrep llama-server)
sleep 2

# 3. Restart with the new reasoning flag
llama-server -m "$MODEL_PATH" -c 4096 --port "$PORT" -t "$THREADS" --no-kv-offload --mlock --reasoning on &
# or --reasoning off to disable

# 4. Verify health
```

### Log tail

```bash
# If llama-server logs to syslog/journald
journalctl -u llama-server --since "5 minutes ago" 2>/dev/null || echo "Not running as systemd service"

# If logs are in a file (e.g., output redirection)
# Check common locations
ls -la /tmp/llama*.log 2>/dev/null
ls -la ~/llama*.log 2>/dev/null

# If the server was started in background, check for output capture
# (the agent should have captured startup output from the terminal tool)
```

## Phase 4: Stop & Switch

### Graceful stop

```bash
# Find the PID
PID=$(pgrep llama-server)
if [ -n "$PID" ]; then
  echo "Sending SIGTERM to llama-server (PID: $PID)..."
  kill "$PID"

  # Wait for port release (up to 10 seconds)
  for i in $(seq 1 10); do
    sleep 1
    if ! ss -tlnp | grep -q ":8080 "; then
      echo "Port 8080 released after ${i}s"
      break
    fi
  done
  echo "Server stopped gracefully"
else
  echo "No llama-server process found"
fi
```

### Force stop

If the process does not respond to SIGTERM:

```bash
PID=$(pgrep llama-server)
if [ -n "$PID" ]; then
  echo "Force killing llama-server (PID: $PID)..."
  kill -9 "$PID"
  sleep 1
  echo "Server killed"
fi
```

### Multi-model switch

Switching from one model to another:

```bash
# 1. Gracefully stop the current model
kill $(pgrep llama-server)
sleep 2

# 2. Verify port is released
if ss -tlnp | grep -q ":8080 "; then
  echo "Port still in use, forcing..."
  kill -9 $(pgrep llama-server) 2>/dev/null
  sleep 2
fi

# 3. Start the new model
llama-server -m /path/to/new-model.gguf -c 4096 --port 8080 -t $(nproc) --no-kv-offload --mlock &

# 4. Health check
sleep 5
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health
```

## Pitfalls

### RAM exhaustion with multiple models

Running more than one model simultaneously can quickly exhaust system memory. Each model's runtime memory usage is approximately 1.5× its file size, plus the KV cache (about 2MB per token of context). If the user requests multiple models, warn them and suggest stopping the current model first.

### Port conflicts

The default port 8080 may be in use by other services (web servers, proxies, other llama.cpp instances). Always check with `ss -tlnp | grep :8080` before starting. If occupied, offer to use an alternative port (8081, 8082, etc.).

### Think mode vs content field behavior

Some models (especially those fine-tuned for reasoning) output their chain-of-thought in a `reasoning_content` field rather than the standard `content` field in the chat completion response. If the user reports seeing blank or truncated responses after enabling `--reasoning on`, check the raw API response JSON for a `reasoning_content` key. The model may need a specific prompt template or chat format to surface reasoning output in the expected field.

### CPU-only inference speed

Without GPU acceleration, inference is slow. Expect approximately:
- 1B–3B models: 10–20 tok/s
- 7B models: 5–10 tok/s
- 13B models: 2–5 tok/s
- 30B+ models: < 2 tok/s (often unusable)

If the user tries to load a model larger than 7B on CPU only, warn them explicitly about expected performance degradation.

### Broken health check after loading

Some models load successfully but return HTTP 500 or hang on the health endpoint, especially if the GGUF file is corrupted or incompatible with the installed `llama.cpp` version. If health checks fail after a reasonable wait, suggest:
1. Verifying the GGUF file integrity
2. Checking the `llama-server` version (`llama-server --version`)
3. Trying a smaller or different model as a diagnostic step
