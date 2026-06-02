---
name: smart-model-router
description: "Use when optimizing model selection across providers to maximize free tier usage and minimize costs. Routes tasks by intensity (light/medium/heavy) to the optimal model. Auto-detects new providers added to config.yaml and syncs them into routing tables. Use when spawning subagents with model overrides, setting up fallback_providers, configuring credential pool strategies, or when the user adds a new provider and wants it integrated into routing."
version: 1.0.0
author: xtoor
license: MIT
metadata:
  hermes:
    tags: [routing, providers, cost-optimization, model-selection, fallbacks]
    related_skills: [hermes-agent, github-pr-workflow]
---

# Smart Model Router

Route every task to the optimal model across all configured providers. Goal: **maximize free tier usage, minimize paid token burn**.

## Architecture

This skill works at two levels:

1. **Config-level** (passive): `fallback_providers` list in `config.yaml` handles automatic failover on errors (429/503/connection fail). Hermes tries them in order.

2. **Agent-level** (active): Apply the routing table below when choosing which model to use for a given task. When spawning subagents, delegating work, or when the user asks "use X for this", pick the right model explicitly. After adding a new provider, run `--check` then `--sync` to integrate it.

## Task Intensity Classification

### Light (cheapest/fastest model available)
- Simple questions, lookups, confirmations
- Text formatting, summaries
- Reading and explaining code (no writing)
- Cron jobs that fetch/compare data
- Session search, file reading

**Cost target: FREE**

### Medium (mid-tier model)
- Writing code (small features, bug fixes)
- Multi-step tool use (3-5 calls)
- Documentation writing
- Refactoring existing code
- Research synthesis

**Cost target: cheapest paid or free tier**

### Heavy (best model available)
- Large feature implementation (100+ lines)
- Architecture design decisions
- Complex debugging (systematic-debugging skill)
- Multi-file refactoring
- Subagent delegation orchestration
- Code review of large diffs

**Cost target: use the best model you can afford**

---

## Provider Model Tiers

### Anthropic (Claude Pro OAuth — $20/mo flat)

Claude Pro has **per-model rate limits** shared across the whole account:
- **Haiku**: Highest rate limit — light tasks
- **Sonnet**: Mid rate limit — medium tasks
- **Opus**: Lowest rate limit, best quality — heavy tasks only

| Tier | Model | Task Intensity | Notes |
|------|-------|---------------|-------|
| Light | `claude-3-5-haiku-20241022` | Light | Burns the least Opus/Sonnet allowance |
| Medium | `claude-sonnet-4-20250514` | Medium | Best balance of quality and rate limit |
| Heavy | `claude-opus-4-20250514` | Heavy | Save for genuinely hard tasks |

**Strategy**: Use Haiku as default for all chat. Default medium tasks to Sonnet. Fire Opus only for the hardest stuff.

### OpenRouter

Model format: `openrouter/<provider>/<model>`

| Tier | Model | Cost | Notes |
|------|-------|------|-------|
| Light | `owl-alpha` | Free | Primary free model, very capable |
| Light | `<any-free-model>` | Free | Check openrouter.ai/models?type=free |
| Medium | `anthropic/claude-sonnet-4` | Paid | If you need Sonnet-tier and anthropic is rate-limited |
| Heavy | `anthropic/claude-opus-4` | Paid | Heaviest openrouter option |

**Strategy**: Free models for light tasks. Paid only when free models can't handle it.

### OpenCode Zen

| Tier | Model | Cost | Notes |
|------|-------|------|-------|
| Light | `glm-4.7-free` | Free tier | Good for light-medium tasks |

### Additional Providers (as configured)

- **Local LLM** (LM Studio, Ollama, etc.): Free, no rate limits. Good for light/medium tasks.
- **Groq**: Free tier available, fast inference.
- **Any other configured provider** is auto-detected and added to routing.

---

## Routing Decision Matrix

```
1. Is the task LIGHT?
   → Try: local model (lmstudio/qwen2.5-3b-instruct or ollama)
   → Fallback: openrouter free tier (owl-alpha, etc.)
   → Last: zen/glm-4.7-free

2. Is the task MEDIUM?
   → Try: anthropic/claude-sonnet-4 (subscription)
   → Fallback: zen/glm-4.7-free or openrouter free
   → Last: anthropic/haiku

3. Is the task HEAVY?
   → Try: anthropic/claude-opus-4 (subscription)
   → Fallback: anthropic/claude-sonnet-4
   → Last: Split into parallel medium subagents

4. Is the config_fallback_providers exhausted?
   → Run detect-providers.py --check to find available alternatives
```

---

## How to Apply This In Practice

### When Choosing a Model
- Classify the incoming task before choosing tools
- For subagent delegation (`delegate_task`), pick the model per intensity
- When you hit a 429, note which provider and escalate to next tier
- After the user adds a new provider, run `--check` then `--sync`

### When Spawning Subagents
```
# Light task subagent
delegate_task(goal="...", model="openrouter/owl-alpha")

# Heavy task subagent
delegate_task(goal="...", model="anthropic/claude-opus-4-20250514")
```

### When Building Cron Jobs
- Use explicit model override per job intensity
- Light monitoring cron → free tier
- Heavy analysis cron → Sonnet or Opus

---

## Auto-Discovery: New Providers

When a new provider is added to `config.yaml` (or `.env`), the routing table
**automatically picks it up**.

### How It Works

1. **Detection**: `scripts/detect-providers.py` scans `config.yaml` and
   `auth.json`. It checks:
   - `model.provider` (primary), `fallback_providers`, `custom_providers`
   - `auth.json` for OAuth provider key names (not token values)
   - Live reachability for local services (LM Studio) and custom endpoints
   - Does **NOT** read `.env` — see README.md for security rationale
   - Custom providers: entries in `config.yaml` → `custom_providers`
   - Local services: LM Studio reachability on port 1234
   - Primary provider: `model.provider` in config

2. **Sync**: New providers are added to `references/routing-table.yaml`
   with `auto_detected: true`

3. **Routing chains are rebuilt**: Free/freemium models go to the front of each
   chain. Paid models fill medium/heavy slots.

4. **Stale providers are removed**: Auto-detected providers no longer in config
   are dropped (unless `pinned: true`).

### Commands

```bash
# After installing the skill to ~/.hermes/skills/devops/smart-model-router/:

# List all detected providers
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py

# Check for new providers not yet in routing table
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --check

# Sync routing table (auto-add new providers)
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --sync

# Preview without writing
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --sync --dry-run

# Watch config.yaml for changes (auto-sync on edit)
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --watch

# Get config.yaml fallback snippet for new providers
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --check --patch

# JSON output
python3 ~/.hermes/skills/devops/smart-model-router/scripts/detect-providers.py --check --json
```

### When to Run Sync

- **After adding a new provider** → run `--check` then `--sync`
- **Before spawning heavy subagents** → `--check` to verify latest providers
- **In cron jobs** → pre-step running `--sync` keeps routing fresh
- **With `--watch`** → continuous, auto-syncs on config change

---

## Credential Pool Setup

To enable automatic rotation across multiple API keys, add entries per provider
in `~/.hermes/auth.json` and set the rotation strategy:

```yaml
credential_pool_strategies:
  anthropic: round_robin
  openrouter: least_used
  opencode-zen: fill_first
```

Pool strategies:
- `fill_first`: Use first credential until exhausted, then move to next (DEFAULT)
- `round_robin`: Rotate evenly across all credentials
- `random`: Pick randomly (good for distributing load)
- `least_used`: Pick the one with fewest requests (best for rate limit optimization)

---

## Config Recommendations

```yaml
# Primary model (free tier)
model:
  api_mode: chat_completions
  default: owl-alpha
  provider: openrouter

# Fallback chain: free first, then paid tiers
fallback_providers:
  - model: glm-4.7-free
    provider: opencode-zen
  - model: claude-sonnet-4-20250514
    provider: anthropic
  - model: claude-opus-4-20250514
    provider: anthropic
```

---

## CLI Helper

For quick model selection without the full detector:

```bash
python3 ~/.hermes/skills/devops/smart-model-router/scripts/model-router.py --task heavy
# Recommends: anthropic/claude-opus-4-20250514

python3 ~/.hermes/skills/devops/smart-model-router/scripts/model-router.py --task medium
# Recommends: anthropic/claude-sonnet-4-20250514

python3 ~/.hermes/skills/devops/smart-model-router/scripts/model-router.py --task light --prefer-free
# Recommends: lmstudio/qwen2.5-3b-instruct
```

---

## Common Pitfalls

1. **Opus for everything** — Burns rate limits fast. Use it only for genuinely hard tasks (architecture, large features, complex debugging).

2. **Ignoring credential pools** — If you have multiple API keys for a provider, pool rotation on `round_robin` or `least_used` maximizes your total throughput before hitting 429s.

3. **Not running sync after config changes** — The routing table is a cache. After adding/removing providers, run `--check` and `--sync` to update it.

4. **Setting paid model as primary** — Your `model.default` should be free-tier. Paid models should be in `fallback_providers` or used via explicit model override on heavy tasks.

5. **Forgetting free models on OpenRouter** — Many models are free on OpenRouter. Run the free-models query in `references/openrouter-free-models.md` to find them.

## Verification Checklist

- [ ] Routing table is synced with current config (`--check` shows no new providers)
- [ ] Primary model is free-tier
- [ ] `fallback_providers` ordered: free → paid medium → paid heavy
- [ ] Credential pool strategies configured if using multiple keys
- [ ] Opus is NOT the default model (only used for heavy subagent delegation)
