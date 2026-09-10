---
name: local-ollama-fallback
description: Use when configuring Hermes Agent to fail over from a hosted primary model to a local Ollama backup during quota, rate-limit, or provider outages.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, ollama, fallback, local-llm, providers, gateway]
    related_skills: [webhook-subscriptions]
---

# Local Ollama Fallback for Hermes

## Overview

Hermes can keep a cloud model as the primary provider and add a local Ollama model as a fallback provider. This is useful for always-on gateway agents: if the hosted provider hits quota, rate limits, or a temporary outage, Hermes can attempt the next provider in `fallback_providers` instead of going completely offline.

The key detail is Hermes' minimum context requirement. Hermes requires a model context window of at least 64,000 tokens. Many Ollama models report a smaller training context even when `num_ctx` is increased in a Modelfile, so the reliable setup uses both:

1. an Ollama wrapper model with `PARAMETER num_ctx 65536`, and
2. a `custom_providers` context override in `~/.hermes/config.yaml`.

## When to Use

Use this skill when:

- The user asks for a local LLM backup for ChatGPT, OpenAI, Anthropic, OpenRouter, or another hosted provider.
- Hermes Gateway should keep responding from Discord, Telegram, WhatsApp, Slack, or another platform during provider outages.
- The machine already has Ollama, or the user is willing to install it.
- The user wants local inference only as fallback, not as the primary model.

Do not use this for:

- Fully local Hermes setup where Ollama is the primary provider; use the local Ollama docs instead.
- Security-sensitive setups where local fallback quality is not acceptable without benchmarking.
- Machines without enough RAM/VRAM to run the selected local model at 64K context.

## Procedure

### 1. Verify Ollama and pick a model

Check that Ollama is installed, serving, and has local models:

```bash
command -v ollama
lsof -nP -iTCP:11434 -sTCP:LISTEN
ollama list
```

Prefer the strongest local model available that can follow tool-use instructions. Qwen and Gemma families are common candidates. Small models can work as an emergency fallback, but they may be worse at tool calling.

### 2. Create a 64K wrapper model

Create a Modelfile. Adjust `FROM` and the wrapper name for the user's model.

```Modelfile
FROM qwen3:8b
PARAMETER num_ctx 65536
PARAMETER temperature 0.2
SYSTEM "You are a local backup model for Hermes Agent. Be concise and direct. Do not reveal hidden reasoning; provide only the final answer."
```

Save it somewhere durable, for example:

```bash
mkdir -p ~/.hermes
$EDITOR ~/.hermes/ollama-qwen3-hermes.Modelfile
ollama create qwen3:8b-hermes -f ~/.hermes/ollama-qwen3-hermes.Modelfile
ollama show qwen3:8b-hermes --parameters
```

Confirm the output includes `num_ctx 65536`.

### 3. Add the fallback provider

Edit `~/.hermes/config.yaml`. Keep the user's current `model:` section unchanged so the hosted provider remains primary. Add or extend `fallback_providers`:

```yaml
fallback_providers:
  - provider: custom
    model: qwen3:8b-hermes
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required
```

If the user already has fallback providers, append this entry at the desired priority. Earlier entries are tried first.

### 4. Add a custom provider alias and context override

Add a named provider for manual testing and force Hermes' model metadata to accept the wrapper as 64K:

```yaml
custom_providers:
  - name: ollama-local
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required
    api_mode: chat_completions
    model: qwen3:8b-hermes
    context_length: 65536
    models:
      qwen3:8b-hermes:
        context_length: 65536

providers:
  ollama-local:
    name: Ollama Local
    base_url: http://127.0.0.1:11434/v1
    api_key: no-key-required
    default_model: qwen3:8b-hermes
    transport: chat_completions
```

This context override is important. Without it, Hermes may still detect the underlying model context, such as 40,960 tokens, and reject startup with:

```text
Model <name> has a context window of <N> tokens, which is below the minimum 64,000 required by Hermes Agent.
```

### 5. Validate config and test local mode directly

```bash
hermes config check
hermes chat --provider ollama-local --model qwen3:8b-hermes \
  -q 'Reply with exactly: local-ok' \
  --toolsets '' \
  --quiet
```

Expected response:

```text
local-ok
```

If the test fails, fix the local provider before restarting a gateway.

### 6. Restart and verify gateway agents

For an installed gateway service:

```bash
hermes gateway restart
hermes gateway status
```

Then check logs for platform reconnection:

```bash
tail -100 ~/.hermes/logs/gateway.log
```

Look for markers such as `Gateway running`, `discord connected`, `telegram connected`, `whatsapp connected`, or the user's configured platforms.

## Optional: Keep the fallback model warm

Ollama may unload idle models. For a gateway that should fail over quickly, keep the model loaded longer:

```bash
curl http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen3:8b-hermes", "keep_alive": "24h"}'
```

Or configure Ollama's service environment with `OLLAMA_KEEP_ALIVE=24h`.

## Common Pitfalls

1. **Wrapper model still rejected below 64K.** `PARAMETER num_ctx 65536` is not always enough for Hermes metadata detection. Add `custom_providers.models.<model>.context_length: 65536`.

2. **Using `localhost` inconsistently.** Prefer `http://127.0.0.1:11434/v1` for config to avoid IPv6 or resolver surprises.

3. **Replacing the primary provider by accident.** For backup-only setups, do not overwrite `model.provider` or `model.default`; only add `fallback_providers` and the optional `ollama-local` alias.

4. **Expecting fallback for model-quality mistakes.** Fallbacks trigger on provider failures/errors, not every bad answer. Small local models may still answer poorly if they are manually selected.

5. **Underpowered hardware.** 64K context increases memory use and latency. If the machine swaps, choose a smaller model or lower expectations for emergency fallback quality.

6. **Leaking secrets in examples.** Ollama does not need a real API key. Use `no-key-required` or another placeholder; never commit hosted provider keys.

## Verification Checklist

- [ ] `ollama list` shows the base model.
- [ ] `ollama show <wrapper> --parameters` shows `num_ctx 65536`.
- [ ] `~/.hermes/config.yaml` keeps the hosted provider as primary.
- [ ] `fallback_providers` includes the local `custom` Ollama endpoint.
- [ ] `custom_providers` includes a 64K context override for the wrapper model.
- [ ] `hermes config check` passes.
- [ ] Direct test with `--provider ollama-local` returns the expected response.
- [ ] Gateway has been restarted and platform logs show reconnection.
