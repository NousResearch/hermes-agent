---
name: quantcheck
description: "Check which weight precision (fp16/bf16/fp8/int4) OpenRouter or Nous Portal actually serves for an open-weight LLM, then set provider_routing.quantizations to the best achievable level. Trigger with /quantcheck or when the user asks about quantized models, full-precision routing, weight quality on OpenRouter/Nous, or 'is model X quantized'."
version: 1.1.0
author: horstenegger
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [openrouter, nous-portal, quantization, provider-routing, llm-quality, weights]
    related_skills: [hermes-agent]
    config:
      - key: quantcheck.allow_probe_spend
        description: "Allow tiny live API probes (~5 tokens) to verify routing filters work before writing them"
        default: "true"
        prompt: "Allow quantcheck to spend a few tokens on live verification requests?"
---

# QuantCheck — parity-with-maker precision routing

OpenRouter and Nous Portal route each request to a backend endpoint. Many
open-weight models are served at reduced precision (fp8, fp4, int4), sometimes
exclusively — and neither aggregator's public models API tells you which. This
skill finds out what precision a model actually gets, and (for OpenRouter)
configures Hermes so requests always use the highest-precision weights
available.

## The core principle

The right target is **parity with the precision the model maker serves via
their own API** — not automatically the highest bit width:

- Many recent open-weight models are **quantization-aware-trained (QAT)**:
  their official checkpoints are natively MXFP4/fp8 (DeepSeek-V4-Flash, for
  example), and the maker's own API serves exactly that. For such models an
  fp8 endpoint is full precision, not a downgrade.
- True quality loss happens when a model ships bf16 natively but hosts serve
  reduced-precision copies of it.
- First-party endpoints (xAI serving Grok, Alibaba serving Qwen) are usually
  listed with `unknown` quantization and ARE the reference implementation.

Precision ranking: `fp32 > bf16 ≈ fp16 > fp8/mxfp8 > fp6 > nvfp4/mxfp4/fp4 > int8 > int4`.

## Procedure

### 1. Determine the current model and provider

```bash
python3 -c "import sys; sys.path.insert(0,'$HOME/.hermes/hermes-agent'); from hermes_cli.config import load_config; c=load_config(); print(c.get('model',{}).get('default',''), c.get('model',{}).get('provider',''))"
```

Strip any `:variant` suffix (`:free`, `:nitro`, `:thinking`, …) before probing.
If the provider is Nous Portal, jump to step 5 (observe-only); otherwise
continue with steps 2–6.

### 2. Enumerate endpoints + quantization (OpenRouter data)

Run the helper script (no API key needed — public pages):

```bash
python3 "[Skill directory]/scripts/or_endpoints.py" "<model-id>"
```

It prints one line per endpoint (`provider_slug  quantization`) plus the unique
set of quantization levels found.

If the scrape returns nothing (page layout changed, brand-new model): say so,
do NOT guess, and suggest checking `https://openrouter.ai/<model>/providers`
manually later.

### 3. Pick the best achievable level

From the unique quantization set found in step 2, choose by ranking:

| Available levels contain | Set to write |
|---|---|
| `fp16` or `bf16` | `["fp16","bf16"]` |
| else `fp8` or `mxfp8` | `["fp8","mxfp8","unknown"]` |
| else `fp6` | `["fp6","fp8","mxfp8","unknown"]` |
| else `nvfp4`/`mxfp4`/`fp4` | `["fp4","nvfp4","mxfp4","unknown"]` |
| else `int8` | `["int8","unknown"]` |
| else only `int4` | `["int4","unknown"]` + WARN: int4-only serving exists |
| else all `unknown` | `["unknown"]` |

Always include `"unknown"` in non-fp16 sets unless the model has declared
fp16/bf16 endpoints: undeclared endpoints are usually first-party (xAI,
Cloudflare, Alibaba) and excluding them can leave zero endpoints for some
models. If research shows the official checkpoint is natively MXFP4/fp8 (QAT),
prefer `[<native-level>, "unknown"]`.

### 4. Verify live before writing (recommended)

```bash
KEY=$(grep "^OPENROUTER_API_KEY=" ~/.hermes/.env | cut -d= -f2)
curl -s --max-time 30 -X POST https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"provider":{"quantizations":["<level1>","<level2>"]}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('provider') or d.get('error',{}).get('message','?')[:100])"
```

- Prints a provider name → filter works.
- Prints `No endpoints found...` → widen the set one rank down and retry once.

This spends a few tokens (~5 max_tokens). Skip if probing spend is disabled or
the user objects.

### 5. Write the config (OpenRouter only)

```bash
hermes config set 'provider_routing.quantizations' '["<l1>","<l2>"]'
grep -A4 provider_routing ~/.hermes/config.yaml   # verify
```

A `/reset` (TUI/Desktop) or gateway restart applies it to live sessions; newly
spawned agents and cron runs pick it up immediately.

## Nous Portal models: observe, don't steer

The Portal rejects caller-supplied `provider` routing filters outright (HTTP
400, "This endpoint does not honor caller-supplied `provider` routing
preferences"), so the config write has no effect there. But you can still find
out which precision a Nous-routed model gets:

1. Make one tiny chat request through the Portal and read the `provider` field
   in the response — that's the backend that served it (e.g.
   deepseek-v4-flash → Novita, kimi-k2.5 → Moonshot AI).
2. Cross-reference that backend on the model's openrouter.ai page to get its
   quantization level.

```bash
python3 "[Skill directory]/scripts/nous_backend.py" "<model-id>" --crossref
```

It prints the Portal backend, all OpenRouter endpoints for the model with their
quantization levels, the matched level for the Nous backend, and a verdict.
Caveats:

- The Portal's backend choice can change over time; results are point-in-time.
- First-party backends (Moonshot serving Kimi, Arcee serving Trinity) are
  almost certainly native precision regardless of what OpenRouter lists.
- Some responses have no `provider` field (first-party direct serving).
- The token in ~/.hermes/auth.json expires hourly; if requests fail with 403,
  run `hermes login --provider nous` first.

Real examples observed (Aug 2026): deepseek-v4-flash → Novita fp8 (parity, QAT
model); trinity-large-thinking → Parasail fp4 (reduced); kimi-k2.5 → Moonshot
AI first-party.

## Report

Tell the user:

- Model probed and its precision ladder (e.g. `bf16: DeepInfra/CoreWeave, fp8: Chutes, unknown: Alibaba`)
- What was configured and why (or, for Nous: what was observed and that steering isn't possible)
- That a restart applies config changes to live sessions
- If int4-only: warn clearly that no better option exists today

## Pitfalls

- **The models API lies by omission**: `GET /v1/models` on both aggregators has
  NO quantization data. Only the scraped openrouter.ai page has per-endpoint quants.
- **Don't write `[fp16, bf16]` blindly**: models like deepseek-v4-flash,
  kimi-k2.5, and glm-5.2 have ZERO fp16/bf16 endpoints — that filter makes every
  request fail with HTTP 404 "No endpoints found". Always scrape first.
- **First-party endpoints hide behind `unknown`**: filtering them out silently
  removes e.g. xAI's own serving of grok models.
- **Auxiliary tasks** (compression, title-gen) use their own
  `auxiliary.<task>.extra_body.provider` settings, not `provider_routing`.
- Requires the `provider_routing.quantizations` passthrough from
  hermes-agent PR #91800 (or equivalent). On stock Hermes without that patch,
  the probe/report steps still work but the config write is inert.

## When this skill retires

If `provider_routing.quantizations` support ships in stock Hermes (#91800 or
equivalent), step 5 becomes a plain config write that works everywhere, and the
probe/report steps remain useful for choosing sensible per-model values.
