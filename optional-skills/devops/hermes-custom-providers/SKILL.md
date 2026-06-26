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

Hermes ships with 29+ built-in providers (Anthropic, OpenAI, DeepSeek, Gemini, etc.) and **Volcengine Ark (火山引擎) is now included as a built-in provider** — just look for "Volcengine Ark" in Settings → Providers and paste your API key.

This skill serves as a **companion reference** for:

- **Alibaba DashScope (阿里百炼)** — Alibaba's AI model platform
- **Zhipu AI / GLM (智谱)** — Tsinghua-backed LLM provider
- **Moonshot / Kimi (月之暗面)** — Standalone Kimi API
- **Any other provider** with OpenAI/Anthropic-compatible endpoints not yet in the built-in list

For Volcengine Ark specifically, this skill is still useful when you need:

- **Custom base_url** — if you're on a Coding Plan with a different endpoint
- **Custom model list** — if your subscription tier has different models than the Agent Plan defaults
- **Model reference** — context lengths, capabilities, deprecated models
- **Troubleshooting** — known issues and workarounds

These providers collectively serve **tens of millions of users** across China and are the primary AI API providers for the Chinese-speaking developer ecosystem. This skill provides the canonical configuration guide for any custom provider, plus a **battle-tested reference implementation** for Volcengine Ark.

## How to Discover and Use This Skill

> ⚠️ **This is an optional skill, not a built-in provider.** You won't find "Volcengine" or "Ark" in the Settings → Providers dropdown. Instead, follow these steps:

### Step 1: Discover the Skill

Once this skill is merged into the Hermes repository, there are two ways to find it:

**Method A — Tips hint (automatic discovery):**

Hermes' startup tips will show a hint when the user is looking for a provider:

> "Don't see your provider (Volcengine Ark, 火山引擎, DashScope, Kimi)? Run `hermes skills search volcengine`."

This hint is displayed in the Hermes terminal interface and directly links the problem ("can't find my provider") to the solution.

**Method B — Manual search:**

```bash
# Search for Ark-related skills
hermes skills search volcengine
hermes skills search ark
hermes skills search 火山

# Or browse all official optional skills
hermes skills browse --source official
```

### Step 2: Install the Skill

```bash
hermes skills install hermes-custom-providers
```

This copies the skill (including the Volcengine Ark reference) to `~/.hermes/skills/`.

### Step 3: Load the Skill in a Hermes Session

Once installed, the skill is available in any Hermes session. Simply ask Hermes to help you configure Ark:

> "Help me set up Volcengine Ark as a custom provider"

Hermes will load this skill, read the reference file, and guide you through editing `~/.hermes/config.yaml`.

### Step 4: Restart Hermes

After editing `config.yaml`:

```bash
/quit          # Exit Hermes completely
hermes          # Relaunch
```

You'll now see `ark-custom` in the model provider picker with all 11 models.

### Why Not a Built-in Provider?

If you're wondering why Ark isn't a first-class provider you can select from Settings → Providers → API Key, see the [FAQ](#faq-why-is-this-a-skill-instead-of-a-built-in-provider) at the bottom.

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

## FAQ: Ark Is Now a Built-in Provider AND a Skill — Why Both?

Volcengine Ark is now available as a **built-in provider** in Settings → Providers. You can paste your API key and start using it immediately. The skill (`hermes-custom-providers`) remains as a **companion reference** for:

### What the built-in provider gives you:

- ⚡ **One-click setup** — select "Volcengine Ark (火山引擎)" from Providers menu, paste API key, done
- 🎯 **Clean model list** — shows exactly the 11 active Agent Plan models, no stale entries
- 🔄 **Zero config** — no YAML editing required

### What the skill adds on top:

- 📋 **Complete model reference** — context lengths, capabilities, deprecated models
- 🔧 **Custom configurations** — if you need a different base_url (Coding Plan) or model subset
- 🩺 **Troubleshooting** — known issues and workarounds for common Ark problems
- 🌐 **Cross-provider patterns** — same approach works for Alibaba DashScope, Zhipu GLM, Kimi, etc.

### The tradeoff

| Aspect | Built-in Provider | Skill + custom_providers |
|--------|:-----------------:|:------------------------:|
| Setup speed | ⚡ Pick from dropdown | 🐢 Install skill → edit YAML → restart |
| Discoverability | ✅ "Volcengine Ark" in Providers menu | ⚠️ Need to know `hermes skills search` exists |
| Model accuracy | ✅ Hardcoded maintained list | ✅ Exactly your activated models |
| Config portability | ❌ Hidden in Hermes internals | ✅ Visible in `config.yaml` |
| Works across providers | ❌ One provider per integration | ✅ Same pattern for all providers |

**Recommendation**: Use the built-in provider for Ark. Use the skill when you need custom endpoint configurations or are setting up other Chinese providers.

## Provider-Specific References

- `references/volcengine-ark-models.md` — Complete guide for Volcengine Ark (火山引擎), including Agent Plan vs Coding Plan comparison, model list, context lengths, working config templates, and troubleshooting.
