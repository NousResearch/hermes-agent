---
name: freellmapi-free-tier
description: "Setup FreeLLMAPI free-tier routing for Hermes."
version: 1.0.0
author: Bob Nyan, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [FreeLLMAPI, Free-Tier, Failover, Routing, Local-Proxy]
    category: autonomous-ai-agents
    related_skills: [opencode-free-rotation, hermes-agent]
---

# FreeLLMAPI Free Tier Skill

Wire Hermes to a local [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) proxy so ~16 free provider tiers share one OpenAI-compatible endpoint. The proxy rotates on 429/5xx internally; Hermes adds a second failover layer via `fallback_providers`.

## When to Use

- You want Hermes on stacked free LLM tiers without managing 16 API keys in the agent
- FreeLLMAPI is (or will be) running on this machine at `http://127.0.0.1:3001/v1`
- You need setup, doctor checks, or fallback-chain wiring after installing the bundled plugin

## Prerequisites

- Hermes checkout with `plugins/model-providers/freellmapi/` and `plugins/freellmapi/`
- FreeLLMAPI running locally (Docker, desktop `.exe`, or `npm run dev`)
- Unified API key from the FreeLLMAPI dashboard → `FREELLMAPI_API_KEY` in `~/.hermes/.env`
- Optional: upstream provider keys added inside FreeLLMAPI (Google, Groq, OpenRouter free, etc.)

## How to Run

1. Enable the integration plugin and wire fallback:

```
terminal(command="hermes freellmapi setup")
```

2. Run health checks (models probe when the API key is set):

```
terminal(command="hermes freellmapi doctor")
```

3. Optional — make FreeLLMAPI the primary model backend:

```
terminal(command="hermes freellmapi setup --apply-model")
```

4. Confirm Hermes-wide diagnostics:

```
terminal(command="hermes doctor")
```

## Quick Reference

| Action | Command |
|--------|---------|
| Enable plugin + fallback | `hermes freellmapi setup` |
| Set primary provider | `hermes freellmapi setup --apply-model` |
| Plugin-only status | `hermes freellmapi status` |
| Deep probe | `hermes freellmapi doctor` |
| Enable allow-list entry | `hermes plugins enable freellmapi` |

## Procedure

### 1. Install and start FreeLLMAPI

**Docker (Linux/macOS/WSL):**

```
terminal(command="curl -fsSL https://freellmapi.co/install.sh | bash")
```

**Windows:** install the `.exe` from [FreeLLMAPI Releases](https://github.com/tashfeenahmed/freellmapi/releases/latest).

Open `http://localhost:3001`, add upstream provider keys on **Keys**, reorder **Fallback Chain**, copy the unified `freellmapi-…` bearer token.

### 2. Store credentials

Add to `~/.hermes/.env` (secrets only):

```
FREELLMAPI_API_KEY=freellmapi-your-unified-key
```

Optional URL override:

```
FREELLMAPI_BASE_URL=http://127.0.0.1:3001/v1
```

### 3. Enable Hermes plugin

```
terminal(command="hermes plugins enable freellmapi")
terminal(command="hermes freellmapi setup")
```

Expected: `freellmapi` appears in `plugins.enabled`; `fallback_providers[0]` becomes `{provider: freellmapi, model: auto}`.

### 4. Recommended config

```yaml
model:
  provider: freellmapi
  default: auto
  base_url: http://127.0.0.1:3001/v1

fallback_providers:
  - provider: freellmapi
    model: auto
  - provider: opencode-zen
    model: auto-free
```

Use `read_file` on `~/.hermes/config.yaml` before editing. Apply with `hermes freellmapi setup --apply-model` or `patch`.

### 5. Verify routing

```
terminal(command="hermes freellmapi doctor")
terminal(command="hermes doctor")
```

Start a short chat. FreeLLMAPI responses include `X-Routed-Via` / `X-Fallback-Attempts` headers when probed with `terminal` + curl.

## Pitfalls

- FreeLLMAPI must be running before `doctor` can pass the models probe — connection refused is expected when the container is stopped.
- The unified key is **not** an upstream Google/Groq key; upstream keys live only inside FreeLLMAPI.
- `model.provider: freellmapi` without `FREELLMAPI_API_KEY` fails at runtime — doctor catches this early.
- Free tiers are for personal experimentation; do not expect production SLA.
- On Windows native Hermes, prefer the desktop `.exe` or WSL Docker over binding Docker to LAN unless you trust the network.

## Verification

- `hermes freellmapi doctor` → `model_provider_profile` and `plugin_enabled` checks pass
- With key + running proxy → `models_probe` ok and `model_count` > 0
- `hermes doctor` shows no blocking errors for the active provider
- Multi-turn sessions send `X-Session-Id` (sticky 30 min) via the bundled model-provider profile

## Optional: context handoff on model switch

In FreeLLMAPI `.env`:

```
FREELLMAPI_CONTEXT_HANDOFF=on_model_switch
```

Injects one compact system message when the proxy falls over to a different upstream model mid-conversation.
