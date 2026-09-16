---
name: hermes-rate-limit-prevention
description: Configure Hermes to prevent API rate limit failures using a fallback model chain, smart routing, and local Ollama as a free provider. Use when rate limits (HTTP 429) are being hit, or to proactively avoid them in high-usage setups.
tags: [hermes, rate-limit, fallback, model-routing, ollama, minimax, gpt, config]
---

# Hermes Rate Limit Prevention

## When to use
- HTTP 429 errors appear in Hermes responses or cron job logs
- Multiple concurrent sessions/cron jobs hit the same API simultaneously
- You want cron jobs or monitoring tasks to use cheap/free models instead of expensive ones
- You want automatic failover when a provider is overloaded

---

## Config location

```
~/.hermes/config.yaml
```

---

## Step 1 — Set primary model

Switch from default to preferred primary (e.g. gpt-5.4 via openai-codex):

```yaml
model:
  default: gpt-5.4
  provider: openai-codex
```

---

## Step 2 — Add local Ollama as free provider

If Ollama runs on a remote machine accessible via Tailscale/LAN, register it as an OpenAI-compatible provider:

```yaml
providers:
  ollama-pluto:
    type: openai_compatible
    base_url: http://100.108.223.25:11434/v1
    api_key: ollama
```

For local Ollama (same machine):
```yaml
providers:
  ollama-local:
    type: openai_compatible
    base_url: http://127.0.0.1:11434/v1
    api_key: ollama
```

---

## Step 3 — Configure fallback model

Uncomment and set the `fallback_model` block (at the bottom of config.yaml, in the comments section):

```yaml
fallback_model:
  provider: minimax
  model: MiniMax-Text-01
```

MiniMax has a 200k context window and is well-suited as fallback. Triggers automatically on 429, 503, 529, or connection failures from the primary.

---

## Step 4 — Enable smart model routing

Routes short/simple messages to a cheap model, saves the primary for complex work:

```yaml
smart_model_routing:
  enabled: true
  max_simple_chars: 300
  max_simple_words: 50
  cheap_model:
    provider: minimax
    model: MiniMax-Text-01
```

Tune `max_simple_chars` / `max_simple_words` based on what counts as "simple" for your use case.

---

## Step 5 — Set cron jobs to lightweight models

When creating cron jobs (especially monitoring/health checks), explicitly assign a cheap model:

```python
mcp_cronjob(action="update", job_id="...", model="MiniMax-Text-01", provider="minimax")
```

Or at creation time:
```python
mcp_cronjob(action="create", ..., model="MiniMax-Text-01", provider="minimax")
```

Health-check crons never need a powerful model — use minimax or a local Ollama model.

---

## Full config snippet (combined)

```yaml
model:
  default: gpt-5.4
  provider: openai-codex

providers:
  ollama-pluto:
    type: openai_compatible
    base_url: http://100.108.223.25:11434/v1
    api_key: ollama

fallback_model:
  provider: minimax
  model: MiniMax-Text-01

smart_model_routing:
  enabled: true
  max_simple_chars: 300
  max_simple_words: 50
  cheap_model:
    provider: minimax
    model: MiniMax-Text-01
```

---

## Pitfalls

- **fallback_model is commented out by default**: it lives at the bottom of config.yaml inside a large comment block. You must uncomment it (remove the `#` prefix) for it to take effect.
- **smart_model_routing default is off**: `enabled: false` by default. Must be explicitly set to `true`.
- **Cron jobs inherit default model**: unless you specify `model`/`provider` at creation or update, cron jobs use the config default — which burns rate-limit quota.
- **Ollama provider needs the service running**: if Ollama is down, calls to `ollama-pluto` will fail. The fallback chain doesn't cover this automatically — keep a health-check cron running.
- **MiniMax-Text-01 context window**: 200k tokens, good for large context tasks as fallback.
- **Compression model**: if `compression.summary_model` is set to a paid provider (e.g. `google/gemini-3-flash-preview`), it will still burn quota during long sessions. Consider pointing it at a cheap/local model too.
- **Always verify Ollama model names before writing config**: do NOT guess model names. Query the live API first:
  `curl -s http://<host>:11434/api/tags | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin)['models']]"`
  Use the exact name returned. Wrong model names silently fail at fallback time — the worst moment to discover it.
- **MiniMax fallback may route internally through Anthropic**: if MiniMax is configured and Anthropic is also rate-limited, the fallback achieves nothing. Use ollama-pluto as fallback instead — it has no API rate limits.
- **Credential pool exhaustion is a silent failure mode**: logs show "credential pool: no available entries (all exhausted or empty)" before every retry. When the pool is empty, Hermes silently bypasses the configured primary model and falls back to whatever auth it can find — often Anthropic. If Anthropic is also rate-limited, the whole chain collapses. Symptom: unexpected model appears in logs despite different config default.
- **"response.output is empty" for gpt-5.4 means Hermes is outdated**: this error indicates an API shape mismatch between the installed Hermes version and the gpt-5.4 response format. Fix: run `hermes update`. Check version with `hermes --version` — if it shows "X commits behind", update immediately before debugging further.
- **Hermes version staleness breaks entire model categories**: running `hermes update` is the first diagnostic step when a model that should work gives shape errors. New models (gpt-5.4, claude-sonnet-4-6, etc.) require up-to-date Hermes to parse responses correctly.
