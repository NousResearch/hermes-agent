# llm-switch — Local LLM Server Manager Plugin

Auto-manage a local LLM server (llama.cpp, vLLM, etc.) from within Hermes.
Switch models via `/model` and the server starts, stops, and swaps automatically.

## Motivation

Many users run local models alongside cloud providers — using llama.cpp for
drafting, coding, or research while keeping cloud models for complex tasks.
Today this requires manually starting/stopping the server outside of Hermes.

This plugin bridges that gap: configure your local models in a YAML file,
and Hermes handles the server lifecycle transparently. Switch between local
and cloud models with `/model` — the same command you already use.

## How it works

The plugin uses two lifecycle hooks:

- **`pre_llm_call`** — fires before every LLM API call. If the active model
  matches a locally configured model and the correct server isn't running,
  the plugin starts it automatically.
- **`on_session_end`** — kills the server when you exit Hermes.

It also registers a **`switch_local_llm` tool** so the agent can switch
models autonomously (e.g., "switch to the code model for this task").

## Setup

1. Copy the plugin to your Hermes plugins directory:

   ```bash
   cp -r examples/plugins/llm-switch ~/.hermes/plugins/llm-switch
   ```

2. Create your model config:

   ```bash
   cd ~/.hermes/plugins/llm-switch
   cp models.yaml.example models.yaml
   # Edit models.yaml — set your models_dir, GGUF paths, and sampling params
   ```

3. Add a custom provider for your local endpoint in `~/.hermes/config.yaml`:

   ```yaml
   custom_providers:
     - name: local
       base_url: http://localhost:8080/v1
       api_key: "sk-local"
   ```

4. Start Hermes and switch to a local model:

   ```
   /model custom:write
   ```

   The server starts automatically on your next message.

## Usage

```
/model custom:write          # Switch to local "write" model — server auto-starts
/model custom:research       # Swap to "research" — old server killed, new one starts
/model anthropic:claude-opus-4.6  # Switch to cloud — server stays running
/model custom:code           # Back to local with "code" model
```

The agent can also switch models via the tool:

```
User: "This task needs the code model"
Agent: [calls switch_local_llm(model="code")]
```

## Configuration

See `models.yaml.example` for the full schema. Key sections:

### `server` — shared server settings

| Field             | Default          | Description                    |
|-------------------|------------------|--------------------------------|
| `binary`          | `llama-server`   | Server binary name or path     |
| `models_dir`      | `~/llama-models` | Base directory for model files |
| `host`            | `0.0.0.0`        | Listen address                 |
| `port`            | `8080`           | Listen port                    |
| `gpu_layers`      | `99`             | GPU offload layers (-ngl)      |
| `flash_attention` | `true`           | Flash attention (-fa)          |
| `parallel`        | `1`              | Concurrent slots (-np)         |
| `jinja`           | `true`           | Chat templates (--jinja)       |

### `models.<name>` — per-model settings

| Field         | Required | Description                              |
|---------------|----------|------------------------------------------|
| `gguf`        | yes      | Path to GGUF file (relative to models_dir) |
| `description` | no       | Human-readable purpose                   |
| `context`     | no       | Context size in tokens (default: 8192)   |
| `kv_cache`    | no       | `{key: q8_0, value: q4_0}` quantization  |
| `sampling`    | no       | Default sampling: temp, top_p, top_k, etc. |
| `alias`       | no       | Name reported by /v1/models              |

## Environment variables

| Variable             | Description                                  |
|----------------------|----------------------------------------------|
| `LLM_SWITCH_MODELS`  | Custom path to models.yaml (default: plugin dir) |
