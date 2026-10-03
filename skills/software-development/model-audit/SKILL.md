---
name: model-audit
description: Audit local LLM models for Hermes compatibility — checks tool calling support, context window size, and memory fit
version: 1.0.0
author: BigBossRabbit
license: MIT
prerequisites:
  commands: [lms, ollama, python3]
metadata:
  hermes:
    tags: [devops, models, local-llm, lm-studio, ollama, configuration]
    related_skills: []
    requires_toolsets: [terminal]
---

# Model Audit

Audit locally installed LLM models (LM Studio, Ollama) for compatibility with Hermes Agent.

## When to Use

- After installing new models
- When getting "Compute error" or "n_keep >= n_ctx" crashes
- Before configuring `smart_model_routing` in config.yaml
- To free up disk space by removing incompatible models

## The Problem

Hermes sends tool definitions in its system prompt (~8-15K tokens). If a model:
- **Doesn't support tool calling** (`trainedForToolUse: false`) → tools silently fail or crash
- **Has context < system prompt tokens** → server rejects with `Compute error`
- **Is too large for available RAM** → slow inference, memory pressure, crashes

## Audit Procedure

### Step 1: Detect system specs

```bash
# macOS
TOTAL_RAM=$(sysctl -n hw.memsize | awk '{printf "%.0f", $0/1073741824}')
echo "RAM: ${TOTAL_RAM} GB"

# Linux
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
echo "RAM: ${TOTAL_RAM} GB"
```

### Step 2: Audit LM Studio models

```bash
lms ls --json | python3 -c "
import json, sys
data = json.load(sys.stdin)
seen = {}
for m in data:
    key = m['modelKey']
    dev = m.get('deviceIdentifier') or 'local'
    if key not in seen or dev == 'local':
        seen[key] = m

print(f'{\"Model\":<50s} {\"Tool\":<6s} {\"Ctx\":>7s} {\"Size\":>7s}')
print(f'{\"─\"*50} {\"─\"*5} {\"─\"*7} {\"─\"*7}')
for key, m in sorted(seen.items(), key=lambda x: (x[1].get('trainedForToolUse', False), x[1].get('maxContextLength', 0)), reverse=True):
    tool = m.get('trainedForToolUse', None)
    ctx = m.get('maxContextLength', 0)
    size_gb = m.get('sizeBytes', 0) / (1024**3)
    if m.get('type') == 'embedding':
        continue
    tool_str = 'Yes' if tool is True else ('No' if tool is False else '?')
    flag = ' ⚠ NO TOOLS' if tool is False else ''
    print(f'{key[:50]:<50s} {tool_str:<6s} {ctx:>7,} {size_gb:>6.2f}G{flag}')
"
```

### Step 3: Audit Ollama models

```bash
ollama list
```

### Step 4: Score and rank

For each model, score on three axes:

| Factor | Points |
|--------|--------|
| Tool calling (`trainedForToolUse: true`) | +3 |
| Context ≥ 262K | +3 |
| Context ≥ 128K | +2 |
| Context ≥ 32K | +1 |
| Size ≤ 3 GB | +1 |
| Size > 6 GB | -1 |

**Score guide:**
- ≥6: ★ Top tier — use as primary model
- ≥4: ✔ Keep — solid for specific tasks
- <2: ✘ Delete — wastes disk, won't work with Hermes

### Step 5: Determine safe context length

Hermes system prompt is ~8-15K tokens. Minimum context length:

| RAM | Safe context (≤4B models) | Safe context (8B+) | Batch size |
|-----|---------------------------|---------------------|------------|
| ≤8 GB | 16,384 | 16,384 | 64 |
| ≤16 GB | 32,768 | 16,384 | 128 |
| ≤32 GB | 32,768 | 32,768 | 256 |
| 64+ GB | 65,536 | 32,768 | 512 |

**Never set context below 16384** — the system prompt won't fit.

### Step 6: Load models with correct settings

```bash
# LM Studio API
curl http://localhost:1234/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-key>",
    "context_length": 16384,
    "eval_batch_size": 64,
    "flash_attention": true,
    "offload_kv_cache_to_gpu": true
  }'
```

### Step 7: Configure Hermes smart routing

In `~/.hermes/config.yaml`:

```yaml
model:
  default: <your-best-tool-calling-model>

smart_model_routing:
  enabled: true
  max_simple_chars: 160
  max_simple_words: 28
  cheap_model:
    provider: custom
    model: <your-lightest-ollama-model>
    base_url: http://localhost:11434/v1
    api_key: ollama
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Compute error` | Context too small for system prompt | Load model with context ≥ 16384 |
| `n_keep >= n_ctx` | System prompt exceeds context window | Increase context_length |
| `Insufficient Memory` (Metal) | GPU ran out of memory | Reduce batch_size to 64, or set `offload_kv_cache_to_gpu: false` |
| Tools not being called | Model doesn't support tools | Switch to a model with `trainedForToolUse: true` |

## Verification

After configuration:
1. `hermes doctor` — should pass model compatibility checks
2. `hermes chat -q "Hello"` — should respond without errors
3. `hermes chat -q "List files in the current directory"` — should use terminal tool successfully
