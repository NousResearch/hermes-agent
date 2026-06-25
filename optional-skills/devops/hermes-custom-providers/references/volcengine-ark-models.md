# Volcengine Ark (火山引擎) Custom Provider — Complete Reference

> **Provider**: ByteDance (字节跳动) Volcengine Ark Platform
> **Users**: Tens of millions across China — the primary API gateway for doubao-seed, deepseek-v4, glm-5.2, kimi-k2, minimax-m3 on the Chinese market
> **Last researched**: 2026-06-26 (model lists change frequently — re-check if models don't appear)

## About Volcengine Ark

Volcengine Ark (火山方舟) is ByteDance's unified AI model platform. It's the primary way Chinese developers access models from multiple vendors through a single API key and billing system. The platform supports both Anthropic Messages API and OpenAI Chat Completions API protocols.

### Subscription Model

- **Lite Plan (Lite 套餐)**: Monthly subscription with auto-renewal. Most common for individual developers.
- **Pay-as-you-go**: Available for enterprise accounts.
- Models are activated from the [console](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement).
- Switching a model takes 3-5 minutes to propagate.

## Endpoints

| Endpoint | Protocol | Extra Cost | Prompt Caching | Recommended |
|----------|----------|:----------:|:--------------:|:-----------:|
| `https://ark.cn-beijing.volces.com/api/coding` | Anthropic Messages | ❌ No | ✅ Yes | ✅ **USE THIS** |
| `https://ark.cn-beijing.volces.com/api/coding/v3` | OpenAI Chat | ❌ No | ❌ No | OK (no cache) |
| `https://ark.cn-beijing.volces.com/api/v3` | Responses API | ⚠️ **YES** | N/A | ❌ **NEVER USE** |

### Why `/api/coding` (Anthropic) is the Best Choice

1. **No extra charges** — the subscription page explicitly warns against `/api/v3`
2. **Prompt caching works automatically** — Hermes relies on this for performance
3. **Full feature parity** — streaming, tool calling, deep thinking, structured output all work
4. **Responses API is irrelevant** — Hermes doesn't support the Responses API protocol, and `/api/v3` costs extra

## Model List (Lite Plan — 11 models as of 2026-06)

| Model ID | Type | Context Window | Max Output | Notes |
|----------|------|:-------------:|:----------:|-------|
| `deepseek-v4-flash` | General | **1,000,000** (1M) | 384,000 | Flagship fast model — best default |
| `deepseek-v4-pro` | Reasoning | **1,000,000** (1M) | 384,000 | Flagship reasoning — higher quality, slower |
| `glm-5.2` | General | **1,000,000** (1M) | 128,000 | Zhipu AI / Tsinghua model |
| `doubao-seed-2.0-pro` | General | 256,000 | 128,000 | ByteDance pro model |
| `doubao-seed-2.0-lite` | General | 256,000 | 128,000 | ByteDance lite model |
| `doubao-seed-2.0-mini` | General | 256,000 | 128,000 | ByteDance mini model |
| `doubao-seed-2.0-code` | Coding | 256,000 | 128,000 | ByteDance coding specialist |
| `kimi-k2.7-code` | Coding | 256,000 | 32,000 | Moonshot coding model |
| `kimi-k2.6` | General | 256,000 | 32,000 | Moonshot general model |
| `minimax-m3` | General | **512,000** | 128,000 | MiniMax latest — large context |
| `minimax-m2.7` | General | 200,000 | 128,000 | MiniMax previous generation |

### Capabilities (all models)

✅ Streaming output
✅ Deep thinking (深度思考)
✅ Tool calling / Function calling
✅ Structured output / JSON mode (beta)
✅ Prefill / continuation mode
✅ Multi-turn conversation
✅ Prompt caching (Anthropic endpoint only)

## Working Config Template

```yaml
# In ~/.hermes/config.yaml

model:
  base_url: https://ark.cn-beijing.volces.com/api/coding
  default: deepseek-v4-pro          # Change to your preferred default
  provider: custom:ark-custom       # Points to the manual-list provider

custom_providers:
  # ── Provider A: Auto-discovery (shows what Ark returns — often stale) ──
  - api_key: ark-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xxxxx
    api_mode: anthropic_messages
    base_url: https://ark.cn-beijing.volces.com/api/coding
    model: deepseek-v4-flash
    models: {}
    name: ark

  # ── Provider B: Manual list (YOUR ACTUAL WORKING MODELS) ──
  - api_key: ark-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xxxxx
    api_mode: anthropic_messages
    base_url: https://ark.cn-beijing.volces.com/api/coding
    discover_models: false             # ← CRITICAL: don't pull Ark's stale model list
    model: deepseek-v4-flash
    models:
      deepseek-v4-flash:
        context_length: 1000000
        name: deepseek-v4-flash
      deepseek-v4-pro:
        context_length: 1000000
        name: deepseek-v4-pro
      glm-5.2:
        context_length: 1000000
        name: glm-5.2
      doubao-seed-2.0-pro:
        context_length: 256000
        name: doubao-seed-2.0-pro
      doubao-seed-2.0-lite:
        context_length: 256000
        name: doubao-seed-2.0-lite
      doubao-seed-2.0-mini:
        context_length: 256000
        name: doubao-seed-2.0-mini
      doubao-seed-2.0-code:
        context_length: 256000
        name: doubao-seed-2.0-code
      kimi-k2.7-code:
        context_length: 256000
        name: kimi-k2.7-code
      kimi-k2.6:
        context_length: 256000
        name: kimi-k2.6
      minimax-m3:
        context_length: 512000
        name: minimax-m3
      minimax-m2.7:
        context_length: 200000
        name: minimax-m2.7
    name: ark-custom
```

## Setup Checklist for New Machines

1. **Get API Key**: [Ark API Key Management](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)
2. **Activate models**: [Model Service Management](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement)
3. **Copy the config template above** into `~/.hermes/config.yaml`
4. **Replace** `ark-xxxxx...` with your actual API key
5. **Launch Hermes**: `hermes`
6. **Select** `ark-custom` provider and verify models appear

## Important Links

| Resource | URL |
|----------|-----|
| Ark Console | https://console.volcengine.com/ark/region:ark+cn-beijing/ |
| API Key Management | https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey |
| Model Activation | https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement |
| Usage Tracking | https://console.volcengine.com/ark/region:ark+cn-beijing/usageTracking |
| Official Model List Docs | https://www.volcengine.com/docs/82379/1330310 |
| Token Calculator | https://console.volcengine.com/ark/region:ark+cn-beijing/tokenCalculator |

## Known Issues & Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Models don't appear after editing config | Used `/reset` instead of full restart | `/quit` → relaunch `hermes` |
| "Model not found" error | Model name has wrong hyphens/version suffix | Match exactly: `deepseek-v4-flash` not `deepseek-v4-flash-260425` |
| 124 stale models in list | `discover_models` is not set to `false` | Set `discover_models: false` in the provider config |
| Config changes ignored | Old `providers:` format still in config | Migrate to `custom_providers:` list format |
| Unexpected charges | Using `/api/v3` endpoint | Switch to `/api/coding` |
