---
name: hermes-custom-providers
description: "Configure custom API providers in Hermes — custom_providers section of config.yaml, model lists, protocol selection (Anthropic vs OpenAI), context_length, and keeping model sync with provider updates. Includes battle-tested reference for Volcengine Ark (火山引擎)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, config, providers, custom, setup, api, china-providers, volcengine-ark]
    related_skills: [hermes-agent]
---

# Hermes Custom Providers

Configuring custom API providers in Hermes via the `custom_providers` section of `~/.hermes/config.yaml`. Use when your API provider is not one of the built-in providers, or when you need to maintain a specific model list that differs from what the provider advertises.

## Why This Skill Exists

Hermes ships with 29+ built-in providers (Anthropic, OpenAI, DeepSeek, Gemini, etc.), but many widely-used providers — especially **Chinese API providers** — are not included as first-class providers. These include:

- **Volcengine Ark (火山引擎)** — ByteDance's model platform, serving doubao-seed-2.0, deepseek-v4, glm-5.2, kimi-k2.x, minimax-m3/m2.7
- **Alibaba DashScope (阿里百炼)** — Alibaba's AI model platform
- **Zhipu AI / GLM (智谱)** — Tsinghua-backed LLM provider
- **Moonshot / Kimi (月之暗面)** — Standalone Kimi API
- **MiniMax (稀宇科技)** — MiniMax standalone API

These providers collectively serve **tens of millions of users** across China and are the primary AI API providers for the Chinese-speaking developer ecosystem. While some offer OpenAI/Anthropic-compatible endpoints, getting them configured correctly in Hermes requires understanding several non-obvious details.

This skill provides the canonical configuration guide for any custom provider, plus a **battle-tested reference implementation** for Volcengine Ark — the most common entry point for Chinese users who want to use Hermes with domestic models.

## When to Use

- Your API provider has a custom endpoint (e.g., Chinese providers like Volcengine Ark, Alibaba DashScope, Z.AI/GLM, MiniMax, Kimi/Moonshot)
- You have a self-hosted or proxied endpoint
- The provider's latest model list differs from what Hermes auto-discovers
- You want to restrict available models to a subset
- You're migrating from the old `providers:` format to the new `custom_providers:` format (Hermes v0.17.0+)

## Config Structure

```yaml
custom_providers:
  - api_key: <your-api-key>
    api_mode: anthropic_messages    # or chat_completions
    base_url: <endpoint-url>
    discover_models: false          # Recommended: set false and manually list models
    model: <default-model-for-this-provider>
    models:
      <model-name>:
        context_length: <tokens>
        name: <model-name>
    name: <provider-label>
```

### Fields Explained

| Field | Required | Description |
|-------|----------|-------------|
| `api_key` | Yes | Your API key for this provider |
| `api_mode` | Yes | `anthropic_messages` or `chat_completions` |
| `base_url` | Yes | The API endpoint URL |
| `discover_models` | No | **Default: true.** Set `false` to disable auto-discovery. Critical — see Pitfall 1 below. |
| `model` | Yes | Default model for this provider (used when no model specified) |
| `models` | Yes | Map of model names to their configs |
| `name` | Yes | A label to reference this provider (used in `provider: custom:<name>`) |

### Protocol Selection: Anthropic vs OpenAI

This is the most important decision when setting up a custom provider:

| Aspect | `anthropic_messages` | `chat_completions` |
|--------|---------------------|-------------------|
| Prompt caching | ✅ Supported (caches system prompt + tools) | ❌ Not supported |
| Compatible endpoints | Anthropic-compatible proxies | OpenAI-compatible endpoints |
| Context window | Set via `context_length` in config | Set via `context_length` in config |
| Use when | Provider offers an Anthropic-API-compatible endpoint | Provider only has OpenAI-compatible endpoint |

**Prefer `anthropic_messages` when the provider supports it** — Hermes relies heavily on prompt caching to reduce latency and cost.

### Top-Level Model Config

After defining custom providers, reference them in the top-level `model` section:

```yaml
model:
  base_url: <same-as-provider-base-url>
  default: <model-name>
  provider: custom:<provider-name>
```

The `provider` value must match `custom:<name>` where `<name>` is the `name` field in your custom_providers entry.

## Migration: Old `providers:` Format → New `custom_providers:` Format

Hermes v0.17.0+ dropped the old `providers:` (nested dict) format. If you upgraded from an earlier version, your config may still have the old format — Hermes silently ignores it, so **none of your models appear**.

### Symptom

You configured models in `providers:` but Hermes shows none of them in the model list / `/model` picker.

### Old Format (DOES NOT WORK on v0.17.0+)

```yaml
# ❌ OLD — completely ignored
providers:
  ark:
    api: https://ark.cn-beijing.volces.com/api/coding
    name: ark
    api_key: ark-xxxxx
    models:
      deepseek-v4-flash:
        context_length: 1000000
        name: deepseek-v4-flash
    default_model: deepseek-v4-flash    # ← old field name
    transport: anthropic_messages       # ← old field name
```

### New Format (REQUIRED)

```yaml
# ✅ NEW — use this
custom_providers:
  - api_key: ark-xxxxx
    api_mode: anthropic_messages        # ← renamed from transport
    base_url: https://ark.cn-beijing.volces.com/api/coding  # ← renamed from api
    discover_models: false
    model: deepseek-v4-flash            # ← renamed from default_model
    models:
      deepseek-v4-flash:
        context_length: 1000000
        name: deepseek-v4-flash
    name: ark-custom
```

### Field Mapping

| Old (`providers:`) | New (`custom_providers:`) | Notes |
|----|----|-------|
| `providers:` (dict key) | `custom_providers:` (list) | Top-level key changed from dict to list of maps |
| `api:` | `base_url:` | The endpoint URL |
| `transport:` | `api_mode:` | `anthropic_messages` or `chat_completions` |
| `default_model:` | `model:` | Default model for this provider |
| `name:` | `name:` | Same — the provider label used in `provider: custom:<name>` |

### Detection

To check if your config has the old format:

```bash
grep -n "^providers:" ~/.hermes/config.yaml
```

If it returns a match, you need to migrate. The `custom_providers:` section can coexist alongside (just delete the old `providers:` block after migration).

## Keeping Models in Sync

Providers frequently add new models. When the user reports "models don't work" or "can't find model X":

1. **Get the provider's current model list** — check the provider's console/dashboard, subscription page, or API docs
2. **Compare against current `models:` list** in config.yaml
3. **For each new model**, determine:
   - **Model name** — exact string the provider uses (hyphens matter!)
   - **Context length** — check provider docs or delegate a research task
4. **Update config** — add new entries to the `models:` map under the provider

### Context Length Research

Not all providers document context_length clearly. Common patterns:

- **deepseek-v4-flash / deepseek-v4-pro**: typically **1M tokens** (1,000,000)
- **glm-5.x series**: typically **1M tokens** (1,000,000)
- **doubao-seed-2.0 series** (pro/lite/mini/code): typically **256K tokens** (256,000)
- **kimi-k2.x series**: typically **256K tokens** (256,000)
- **minimax-m3**: typically **512K tokens** (512,000)
- **minimax-m2.7**: typically **200K tokens** (200,000)

When in doubt, delegate a web research task to look up the provider's official docs.

## Pitfalls

### Pitfall 1: `discover_models` Auto-Discovery Can Pull Stale Models

**The #1 most common issue.** If you don't set `discover_models: false`, Hermes will call the provider's model list API to auto-discover models. Some providers (notably Volcengine Ark) return **outdated model lists** — Ark returns 124+ old model IDs that don't match the user's actual subscription.

✅ **Solution**: Always set `discover_models: false` and manually list your models in the `models:` map.

### Pitfall 2: Old Format Silently Ignored

If models don't appear, check for old `providers:` key first. No error is logged — Hermes just acts as if no custom providers exist. See Migration section above.

### Pitfall 3: `/reset` Does NOT Reload Config

After editing `config.yaml`, running `/reset` only resets the conversation — it does **not** re-read the configuration file.

✅ **Correct steps**:
1. `/quit` to exit Hermes completely
2. Relaunch `hermes`

### Pitfall 4: Model Names Must Match Exactly

Provider model names in your config must match the provider's API **exactly**, including all hyphens and version suffixes:

- ✅ `deepseek-v4-flash` (not `deepseek-v4-flash-260425` which is a stale version suffix)
- ✅ `doubao-seed-2.0-pro` (not `doubao-seed-2-0-pro`)
- ✅ `glm-5.2` (not `glm-5-2`)

### Pitfall 5: Watch Out for Extra-Charge Endpoints

Some providers charge differently for different endpoint paths. **Always check the provider's pricing page.** For Volcengine Ark specifically:

| Endpoint | Protocol | Extra Cost | Recommendation |
|----------|----------|:----------:|----------------|
| `https://ark.cn-beijing.volces.com/api/coding` | Anthropic Messages | ❌ None | ✅ **Use this** |
| `https://ark.cn-beijing.volces.com/api/coding/v3` | OpenAI Chat | ❌ None | OK, but no prompt caching |
| `https://ark.cn-beijing.volces.com/api/v3` | Responses API | ⚠️ **YES** | ❌ **Do NOT use** |

The Responses API (`/api/v3`) endpoint incurs additional charges and Hermes does not support the Responses API protocol anyway.

### Pitfall 6: Two-Provider Pattern for Safe Experimentation

A useful pattern is defining **two** custom providers sharing the same API key but with different `name` and `discover_models` settings:

```yaml
custom_providers:
  # Provider 1: Auto-discovery (shows what the API returns — may be stale)
  - name: ark
    api_key: ark-xxxxx
    api_mode: anthropic_messages
    base_url: https://ark.cn-beijing.volces.com/api/coding
    model: deepseek-v4-flash
    models: {}

  # Provider 2: Manual list (your actual working models)
  - name: ark-custom
    api_key: ark-xxxxx
    api_mode: anthropic_messages
    base_url: https://ark.cn-beijing.volces.com/api/coding
    discover_models: false
    model: deepseek-v4-flash
    models:
      deepseek-v4-flash:
        context_length: 1000000
        name: deepseek-v4-flash
      # ... more models
```

Then set `model.provider: custom:ark-custom` as your default. This gives you a reference view (`ark`) and a working view (`ark-custom`).

## Verification

After updating:
1. Check config syntax: `hermes doctor`
2. **Completely exit and relaunch** Hermes (not just `/reset`)
3. In the model picker, you should see your custom provider with its models
4. Try a simple prompt to verify API connectivity

## Provider-Specific References

- `references/volcengine-ark-models.md` — Complete guide for Volcengine Ark (火山引擎), including model list, context lengths, subscription types, and working config templates.
