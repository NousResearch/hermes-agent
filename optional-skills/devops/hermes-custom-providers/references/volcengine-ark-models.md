# Volcengine Ark (火山引擎) Custom Provider — Complete Reference

> **Provider**: ByteDance (字节跳动) Volcengine Ark Platform
> **Users**: Tens of millions across China — the primary API gateway for doubao-seed, deepseek-v4, glm-5.2, kimi-k2, minimax-m3 on the Chinese market
> **Last researched**: 2026-06-26 (model lists change frequently — re-check if models don't appear)

## About Volcengine Ark

Volcengine Ark (火山方舟) is ByteDance's unified AI model platform. It's the primary way Chinese developers access models from multiple vendors through a single API key and billing system. The platform supports both Anthropic Messages API and OpenAI Chat Completions API protocols.

## Two Subscription Plans: Agent Plan vs Coding Plan

Volcengine Ark offers **two separate subscription plans** for individual users. They share the same API protocol and endpoints but differ in scope, billing model, and use restrictions.

### Quick Comparison

| Dimension | Agent Plan (Agent Plan 个人版) | Coding Plan (Coding Plan 个人版) |
|-----------|-------------------------------|----------------------------------|
| **Target audience** | AI Agent / general-purpose users | Developer / coding-focused users |
| **Billing unit** | AFP (Agent Fuel Point) — unified points | Token-based quota |
| **Tiers** | Small / Medium / Large / Max (4 tiers) | Lite / Pro (2 tiers) |
| **Model scope** | Text + Image + Video + Embedding + Harness (豆包搜索等) | Text LLMs + Embedding only |
| **Multi-modal** | ✅ Image gen, video gen, TTS, ASR | ❌ Text-only LLMs |
| **Use restriction** | Can be used in AI tools + API | **AI coding tools ONLY** — API calls flagged as abuse |
| **Rate limit** | Per-plan tier | 5-hour rolling window + weekly + monthly caps |
| **Official doc** | [Agent Plan 套餐概览](https://www.volcengine.com/docs/82379/2366394) | [Coding Plan 套餐概览](https://www.volcengine.com/docs/82379/2366394) (same page, different tab) |
| **Hermes compatibility** | ✅ Fully compatible | ⚠️ **RISKY** — Coding Plan forbids non-coding-tool API usage |

### ⚠️ Critical: Coding Plan API Restriction

The Coding Plan documentation explicitly states:

> "套餐额度仅在 AI 编程工具中生效，不可用于 API 调用。在非 AI 编程工具中使用方舟 Coding Plan 权益对应的 Base URL 和 API Key 有可能被识别为滥用/违规，会导致订阅停用或账号封禁。"

Translation: **Coding Plan quotas are only valid in AI coding tools. Using the Coding Plan's Base URL and API Key outside of coding tools may be flagged as abuse, resulting in subscription suspension or account ban.**

This is a significant risk for Hermes users — Hermes is an AI agent, not strictly a coding tool. If Volcengine's detection flags Hermes as a non-coding-tool client, your subscription could be terminated.

### ✅ Recommendation: Use Agent Plan

For Hermes users, **the Agent Plan is the safe choice**:

1. **No tool restrictions** — Agent Plan is designed for general AI agent usage and explicitly lists Hermes Agent as a supported tool
2. **More models** — includes multi-modal models (image, video, TTS) not available in Coding Plan
3. **Same LLMs** — all Coding Plan text models are also available in Agent Plan
4. **AFP billing is predictable** — unified points system vs token counting
5. **4 tiers** — Small/Medium/Large/Max gives more flexibility than Lite/Pro

If you already have a Coding Plan subscription, it **will work** from a purely technical standpoint (same endpoints, same API protocol) — but be aware of the account risk.

## Endpoints

Both plans use the same endpoints:

| Endpoint | Protocol | Extra Cost | Prompt Caching | Recommended |
|----------|----------|:----------:|:--------------:|:-----------:|
| `https://ark.cn-beijing.volces.com/api/coding` | Anthropic Messages | ❌ No | ✅ Yes (verified) | ✅ **USE THIS** |
| `https://ark.cn-beijing.volces.com/api/coding/v3` | OpenAI Chat | ❌ No | ❌ No | OK (no cache) |
| `https://ark.cn-beijing.volces.com/api/v3` | Responses API | ⚠️ **YES** | N/A | ❌ **NEVER USE** |

> **Prompt caching confirmed**: Ark's Anthropic endpoint honors `cache_control` markers. Hermes automatically enables caching for Ark when detected via provider name (`volcengine-ark`, `ark`, `volcengine`) or hostname (`ark.cn-beijing.volces.com`). The system prompt and last 3 messages are cached with 5-minute TTL, reducing input costs by ~75% on multi-turn conversations. Without caching, the full system prompt (including tool definitions and skills) is re-billed on every turn.

### Why `/api/coding` (Anthropic) is the Best Choice

1. **No extra charges** — the subscription page explicitly warns against `/api/v3`
2. **Prompt caching works automatically** — Hermes relies on this for performance
3. **Full feature parity** — streaming, tool calling, deep thinking, structured output all work
4. **Responses API is irrelevant** — Hermes doesn't support the Responses API protocol, and `/api/v3` costs extra

## Model List (Agent Plan — all tiers, as of 2026-06)

### Text Generation Models (shared by both plans)

| Model ID | Type | Context Window | Max Output | Notes |
|----------|------|:-------------:|:----------:|-------|
| `deepseek-v4-flash` | General | **1,000,000** (1M) | 384,000 | Flagship fast model — best default. ⚠️ Trial version, may throttle. |
| `deepseek-v4-pro` | Reasoning | **1,000,000** (1M) | 384,000 | Flagship reasoning. High AFP cost, use for complex tasks. |
| `glm-5.2` | General | **1,000,000** (1M) | 128,000 | Zhipu AI / Tsinghua flagship. |
| `doubao-seed-2.0-pro` | General | 256,000 | 128,000 | ByteDance flagship — complex reasoning, long-chain tasks. |
| `doubao-seed-2.0-lite` | General | 256,000 | 128,000 | ByteDance — balanced quality/speed for production. |
| `doubao-seed-2.0-mini` | General | 256,000 | 128,000 | ByteDance — fastest, lightest model. |
| `doubao-seed-2.0-code` | Coding | 256,000 | 128,000 | ByteDance coding specialist with vision. |
| `kimi-k2.7-code` | Coding | 256,000 | 32,000 | Moonshot latest coding model. |
| `kimi-k2.6` | General | 256,000 | 32,000 | Moonshot — strong reasoning, multi-step tool calls. High AFP cost. |
| `minimax-m3` | General | **512,000** | 128,000 | MiniMax latest — Agent reasoning, tool calling, code. |
| `minimax-m2.7` | General | 200,000 | 128,000 | MiniMax — complex Agent harness. High AFP cost. |

### Agent Plan Exclusive (not in Coding Plan)

| Model ID | Domain | Notes |
|----------|--------|-------|
| `doubao-seedream-5.0-lite` | Image Generation | Text-to-image |
| `doubao-seedance-1.5-pro` | Video Generation | Text-to-video |
| `doubao-seedance-2.0` | Video Generation | Next-gen video gen |
| `doubao-seedance-2.0-fast` | Video Generation | Fast video gen |
| `doubao-seed-tts-2.0` | TTS | Text-to-speech |
| `doubao-seed-asr-2.0` | ASR | Speech recognition |
| `doubao-embedding-vision` | Embedding | Vision embedding |

> **Note**: Small tier does not support video generation. Medium+ recommended.

### Deprecated Models (avoid)

| Model | Status |
|-------|--------|
| `deepseek-v3.2` | ⚠️ Retiring soon — migrate to v4 |
| `kimi-k2.5` | ⚠️ Retiring soon |
| `glm-5.1` | ⚠️ Retiring soon — migrate to glm-5.2 |
| `glm-4.7` | ⚠️ Retiring soon |
| `minimax-m2.5` | ⚠️ Retiring soon |

### Capabilities (all text models)

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

1. **Choose a plan**: [Agent Plan](https://console.volcengine.com/ark/region:ark+cn-beijing/) (recommended) or Coding Plan
2. **Get API Key**: [Ark API Key Management](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey)
3. **Activate models**: [Model Service Management](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement)
4. **Copy the config template above** into `~/.hermes/config.yaml`
5. **Replace** `ark-xxxxx...` with your actual API key
6. **Launch Hermes**: `hermes`
7. **Select** `ark-custom` provider and verify models appear

## Important Links

| Resource | URL |
|----------|-----|
| Ark Console | https://console.volcengine.com/ark/region:ark+cn-beijing/ |
| Plan Overview (Agent + Coding) | https://www.volcengine.com/docs/82379/2366394 |
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
| Account flagged / banned | Using Coding Plan outside coding tools | Switch to Agent Plan |
| `max_tokens` 400 error on OpenAI endpoint | Server ceiling < 65536 (e.g., 32768 for kimi-k2.7) | Use `anthropic_messages` protocol instead |
| User-Agent 400 error on OpenAI endpoint | Ark blocks default OpenAI SDK UA | Use `anthropic_messages` protocol instead |
| Quota drains too fast, `cache_read_input_tokens: 0` | Provider not in caching whitelist | Ensure provider name is `volcengine-ark` or host is `ark.cn-beijing.volces.com` — Hermes auto-enables caching for these |
