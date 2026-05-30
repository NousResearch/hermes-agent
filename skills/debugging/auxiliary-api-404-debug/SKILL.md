---
name: auxiliary-api-404-debug
description: "Debug auxiliary task 404 errors in Hermes Agent."
version: 1.0.0
author: Hermes Agent (user:ruoxi001)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [debugging, auxiliary, 404, minimax, config]
    related_skills: [systematic-debugging]
---

# Auxiliary Tasks 404 Debugging

Fixes auxiliary tasks (compression, title generation, etc.) returning HTTP 404 while the main model call works normally.

## When to Use

Triggered when:
- Main model works fine
- Auxiliary tasks all return **404** (or similar HTTP errors)
- `grep -r "anthropic" ~/.hermes/config.yaml` returns results
- You recently changed API provider or model configuration

## Prerequisites

Environment variable:
- `MINIMAX_CODING_API_KEY` — MiniMax API key (set in `~/.hermes/.env`)

## Quick Reference

MiniMax uses the **OpenAI-compatible endpoint** (`/v1/chat/completions`), NOT the Anthropic messages endpoint (`/anthropic/v1/messages`).

| Field | Correct value | Common wrong value |
|-------|--------------|-------------------|
| `base_url` | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` |
| `api_mode` | `chat_completions` | `anthropic_messages` |

## Verification Commands

```bash
# Check auxiliary base_url
grep -A5 "^auxiliary:" ~/.hermes/config.yaml | grep base_url

# Check compression base_url
grep -A5 "^compression:" ~/.hermes/config.yaml | grep base_url

# Expected: https://api.minimaxi.com/v1 (not /anthropic)
# Expected: api_mode: chat_completions (not anthropic_messages)
```

## Procedure

### Step 1: Verify the correct API endpoint

Directly call the API to confirm the working URL:

```bash
curl -s -X POST https://api.minimaxi.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -d '{"model":"MiniMax-M2.7","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

- Returns a valid JSON response → endpoint is correct
- Returns 404 or error → check your API key

### Step 2: Inspect config files

Check all three config locations:

```bash
# 1. Main model config
grep -E "base_url|api_mode" ~/.hermes/config.yaml | grep -v "^#"

# 2. Provider config
cat ~/.hermes/providers/minimax_coding.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('provider:', d.get('base_url'), d.get('api_mode'))"

# 3. .env override (takes priority over config.yaml)
grep MINIMAX ~/.hermes/.env 2>/dev/null
```

All must show `https://api.minimaxi.com/v1` and `chat_completions`.

### Step 3: Fix all locations

If any location is wrong, update it:

```bash
# Update ~/.hermes/.env (example)
sed -i 's|api.minimaxi.com/anthropic|api.minimaxi.com/v1|g' ~/.hermes/.env
```

### Step 4: Restart Hermes Gateway

Config changes require a restart:

```bash
systemctl --user restart hermes-gateway
```

### Step 5: Verify auxiliary tasks work

After restart, test an auxiliary task (e.g., compression) and confirm it no longer returns 404.

## Configuration Locations Reference

The following **all** must be consistent:

| File | Field | Correct value |
|------|-------|---------------|
| `~/.hermes/config.yaml` → `model` | `base_url` | `https://api.minimaxi.com/v1` |
| `~/.hermes/config.yaml` → `model` | `api_mode` | `chat_completions` |
| `~/.hermes/config.yaml` → `auxiliary.*` (all tasks) | `base_url` | `https://api.minimaxi.com/v1` |
| `~/.hermes/config.yaml` → `auxiliary.*` (all tasks) | `api_mode` | `chat_completions` |
| `~/.hermes/providers/minimax_coding.json` | `base_url` | `https://api.minimaxi.com/v1` |
| `~/.hermes/providers/minimax_coding.json` | `api_mode` | `chat_completions` |
| `~/.hermes/.env` | `MINIMAX_CODING_BASE_URL` | `https://api.minimaxi.com/v1` |

> **Note:** `.env` is a systemd EnvironmentFile with higher priority than `config.yaml`. Changes to `.env` also require `systemctl --user restart hermes-gateway`.

## Two Endpoints Are Different

| Tool | base_url | Endpoint |
|------|----------|----------|
| Hermes auxiliary tasks (compression/title) | `/v1` | `/v1/chat/completions` |
| Claude Code | `/anthropic` | `/anthropic/v1/messages` |

Both use the same API key but point to different endpoints. The `api_mode` field controls which endpoint is used.

## Pitfalls

1. **Restart is mandatory.** Gateway reads config at startup. Editing config without restart has no effect.
2. **`.env` takes priority.** A correct `config.yaml` with a wrong `.env` still fails — check both.
3. **Search for `/anthropic` in all files.** Any config file containing `/anthropic` in a MiniMax base_url is wrong.
4. **Provider alias.** Auxiliary tasks use `custom:minimax_coding` to reference `providers.minimax_coding`. Both the provider and auxiliary configs must be correct independently.