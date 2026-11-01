---
name: llm-provider-healthcheck
author: rafael.zendron22@gmail.com
description: "Test and diagnose LLM API provider integrations."
tags: [llm, providers, healthcheck, api, diagnostics, hermes]
version: 1.0.0
---

# LLM Provider Health Check

Test every LLM API provider integration systematically. Catch auth failures, quota exhaustion, discontinued models, and key sync issues before they bite in production.

## When to Use

- After setting up new providers or API keys
- Periodically (cron) to detect key expiration or model deprecation
- When debugging "model not found" or "auth failed" errors in Hermes gateway logs
- Before relying on fallback providers in production

## Steps

### 1. MAP providers from config

Read all providers and their keys from the Hermes config:

```
config.yaml -> providers.{name}.api_key, providers.{name}.base_url
config.yaml -> fallback_providers[].{provider, model}
.env -> {PROVIDER}_API_KEY entries
```

**KEY PITFALL**: A provider can have keys in BOTH `config.yaml` (under `providers.{name}.api_key`) AND `.env` (as `{NAME}_API_KEY`). They may differ. Hermes may use one or the other depending on context. Always check both and flag mismatches.

### 2. Test each provider with a minimal prompt

Send `"Say only: OK"` to each provider. Measure latency. Record HTTP status.

**CRITICAL: Shell quoting pitfall** — Do NOT pass JSON payloads inline via curl through the agent `terminal()` tool. Shell quoting (single quotes in JSON, curly braces, env var interpolation) breaks unpredictably. Instead:

**Preferred approach — Python subprocess:**
```python
import json, urllib.request, ssl
ctx = ssl.create_default_context()
payload = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say only: OK"}], "max_tokens": 50}).encode()
req = urllib.request.Request(f"{base_url}/chat/completions", data=payload, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
})
with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
    data = json.loads(resp.read())
```

**NIM-specific note**: NVIDIA NIM may return `{"error": {"message": "Missing request extension"}}` via `urllib.request` due to HTTP/2 negotiation. If this happens, use the `openai` Python library instead: `client = openai.OpenAI(base_url=base_url, api_key=key); client.chat.completions.create(...)`.

**Batch testing multiple models**: When testing 10+ models on a single provider (e.g., NVIDIA NIM's 117-model catalog), use **individual timeouts of 25-30s** per model. Many models timeout on free tier — don't let one slow model block the entire batch. Process models sequentially (not parallel) to avoid triggering rate limits.

**Alternative — write JSON to file, use `@/tmp/payload.json`:**
```bash
curl -s -X POST "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d @/tmp/payload.json
```

**Avoid** — inline JSON with curl through terminal():
```bash
# THIS BREAKS: single quotes in payload, curly brace expansion, nested quotes
curl ... -d '{"model":"x","messages":[{"role":"user","content":"OK"}]}'
```

### 3. Diagnose failures by category

| HTTP Status | Meaning | Fix |
|-------------|---------|-----|
| 401/403 | Auth error | Key expired or wrong. Check key in config vs .env. Regenerate if needed. |
| 404 | Model not found | Model discontinued. List available models via `GET /models` endpoint. Update config. |
| 402 | Credits exhausted | OpenRouter/paid provider out of credits. Remove from fallback chain or top up. |
| 410 | Gone | Model permanently removed. Replace with current equivalent. |
| 429 | Rate limit / Quota | Free tier exhausted. Enable billing or wait for reset. |
| 400 "reasoning_content unsupported" | **Cascade failure** | Primary reasoning model (GLM-5/DeepSeek) saved `reasoning_content` in history. Fallback provider doesn't support this field. See `references/provider-quirks.md` → "Reasoning Content Cascade Failure". |
| 200 + empty content | Likely a reasoning model | Check for `reasoning_content` field. Increase `max_tokens` to 500+. |
| Timeout (>30s) | Network or model unavailable | Check base_url. Model may be under load. |

### 4. List available models for failed providers

When a model returns 404/410, query the provider's `/models` or `/v1/models` endpoint to find replacements:

```python
req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
with urllib.request.urlopen(req, timeout=15) as resp:
    models = json.loads(resp.read())
    for m in models.get("data", []):
        print(m.get("id"))
```

### 5. Fix config and restart gateway

- Remove invalid fallback entries from `fallback_providers` in config.yaml
- Remove providers with expired/invalid keys from `providers` section
- Sync keys between .env and config.yaml if they differ
- Restart gateway: `hermes gateway run --replace` (background)

## Fallback Chain Design Strategy

Organize fallbacks in **speed tiers** — fastest providers first, same-provider models last:

```
T1 — ULTRA FAST (< 2s):   Groq, Cerebras, NVIDIA fast models (mistral-small-4, etc.)
T2 — MEDIUM (2-8s):       NVIDIA quality models (llama-3.3-70b), SambaNova, Mistral direct
T3 — SAME PROVIDER (8s+):  Same provider as primary (Z.AI fallbacks) — slow but API-compatible
T4 — LAST RESORT:          Cloudflare Workers AI — always available but limited quality
```

**Why this order matters**: When the primary fails, you want the FASTEST possible response to minimize user-visible latency. Putting a 10s model before a 500ms model means the user waits 10+ seconds before even trying the fast one.

**Same-provider tier (T3)**: Models from the same provider as the primary (e.g., Z.AI glm-4.7 when primary is glm-5.1) are API-compatible but slow — the provider may still be rate-limiting you. Put these AFTER cross-provider fallbacks.

**Delegation config is SEPARATE**: Hermes `delegation` config has its own `provider`, `model`, `base_url`, and `api_key`. It does NOT inherit from the primary or fallback chain. If delegation points to a dead provider (e.g., OpenRouter with no credits), all `delegate_task` calls will fail silently. Always check and fix delegation config alongside fallbacks.

```yaml
# Fast delegation (Groq ~360ms) — best for subagents
delegation:
  provider: "custom:groq"
  model: "llama-3.3-70b-versatile"
  reasoning_effort: "low"  # subagents don't need heavy reasoning
  # NO base_url or api_key — inherits from providers.groq section

# If Groq is down, NVIDIA NIM works but slower (~4s):
# delegation:
#   provider: "custom:nvidia"
#   model: "meta/llama-3.3-70b-instruct"
#   base_url: "https://integrate.api.nvidia.com/v1"
#   api_key: "nvapi-..."
#   reasoning_effort: "low"
```

## Rate Limit & Retry Tuning

When a provider starts returning 429 (rate limit), every retry burns time and potentially more quota. Tune these parameters:

- **`api_max_retries`**: Default is 5. Reduce to 3 so the fallback chain activates faster (~15s instead of ~60s on Z.AI 429).
- **Fallback chain order matters**: T1 models (<2s) should come before T3/T4. A 500ms Groq model should NOT be after a 10s same-provider fallback.
- **Delegation should use the fastest free provider**: Groq `llama-3.3-70b-versatile` (~360ms) is 10x faster than NVIDIA NIM equivalent (~4s). Set `delegation.reasoning_effort: low` since subagents don't need heavy reasoning.

```yaml
# Optimized rate-limit config
api_max_retries: 3  # was 5 — faster fallback on 429

delegation:
  provider: "custom:groq"
  model: "llama-3.3-70b-versatile"
  reasoning_effort: "low"  # subagents don't need heavy reasoning
```

## Pitfalls

- **OpenRouter credits exhaust silently**: Free-tier credits can be consumed by a single large-context request. This produces HTTP 402, not 429. Remove those fallbacks immediately — they'll never recover without topping up.
- **Delegation config is independent from fallbacks**: The `delegation` section in config.yaml has its own `provider`, `model`, `base_url`, and `api_key`. It does NOT inherit from the primary or fallback chain. When you remove a broken provider from fallbacks, ALSO check delegation — if it still points to the dead provider, `delegate_task` will fail on every call with no visible error to the user.
- **Delegation base_url must match provider**: When switching delegation from one provider to another (e.g., NVIDIA → Groq), you MUST also update `base_url` and `api_key`. Otherwise Hermes sends Groq model names to NVIDIA's endpoint, causing confusing errors. Remove the old `base_url` and `api_key` fields to let delegation inherit from the named provider section.
- **NVIDIA NIM free tier models timeout**: DeepSeek V4 Flash and other heavy models timeout at 25-30s on free tier. Only use fast NIM models as fallbacks (mistral-small-4-119b ~500ms, llama-3.3-70b ~4s, glm-5.1 ~8-11s).

## Provider-Specific Quirks

Each provider has unique behaviors (auth formats, rate limits, model deprecation). Test empirically with the healthcheck script above and record findings here as you discover them.

## Free Tier Model Catalog & Task Routing

See `references/free-tier-model-catalog.md` for the complete free model catalog (Tier S/A), task-type routing matrix (planning → qwen3-coder, review → kimi-k2.6, enactment → groq/llama-70b), multi-model voting strategy, and all provider endpoints. Use this when configuring delegation model routing or choosing fallback models.

## Reusable Scripts

- `scripts/healthcheck_providers.py` — Standalone script that tests all providers from Hermes config. Run: `python3 scripts/healthcheck_providers.py [--verbose] [--timeout 30]`. Exit code 0 if all pass, 1 if any fail.

## Output Contract

For each provider tested, record:
- **provider** name
- **model** tested
- **status**: pass | fail | timeout | auth_error | quota_exceeded | skip
- **latency_ms**: round-trip time
- **response_preview**: first 80 chars of response or error message
- **error**: diagnostic message on failure

## Validation

- [ ] Every provider in config tested
- [ ] Latency measured
- [ ] Failed providers diagnosed with root cause
- [ ] Invalid models/keys removed from config
- [ ] Gateway restarted and platforms reconnected
- [ ] Report saved to project planning directory
