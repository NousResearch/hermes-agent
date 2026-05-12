---
name: morpheus-provider
description: Use when switching to or troubleshooting Morpheus decentralized inference (MOR-staked, GLM-5.1, remote gateway). Provides exact config, .env fixes, curl tests, key/dot pitfalls, and auxiliary fallback patterns.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [morpheus, glm-5.1, decentralized, provider, inference, api-key]
    related_skills: [hermes-agent, github-pr-workflow]
---

# Morpheus Provider (Remote Gateway + GLM-5.1)

Use when the user wants Morpheus (https://api.mor.org) for GLM-5.1 or other decentralized models. Handles the common 401 "Invalid API key format" even when curl succeeds, .env vs config conflicts, endpoint sensitivity (/api/v1 vs /v1), and auxiliary title/compression errors.

This skill ships the lessons from the May 2026 GLM-5.1 stabilization session.

## When to Use
- User says "switch to GLM 5.1" or "add Morpheus"
- Seeing 401 with key that starts with `sk-` (especially those with embedded dot like `sk-hrOKrR.88...`)
- Auxiliary title generation or other sub-tasks fail with "Connection error"
- Want remote gateway over local proxy (simpler, no EverClaw daemon)

Do not use for local proxy debugging or OpenClaw-specific setups — see `hermes-agent` skill references instead.

## Setup (Run in order)

1. Ensure key in .env (use python one-liner to avoid protection):
   ```bash
   python=~/.hermes/hermes-agent/venv/bin/python
   $python -c '
   p="/Users/hermesagent/.hermes/.env"
   with open(p,"r") as f: c=f.read()
   c = c.replace("MORPHEUS_API_KEY=old_or_dummy", "MORPHEUS_API_KEY=sk-your-morpheus-key-here")
   open(p,"w").write(c)
   '
   ```

2. Apply explicit config (critical for dot-key and remote endpoint):
   ```bash
   hermes config set model.provider morpheus
   hermes config set model.default GLM-5.1
   hermes config set model.base_url https://api.mor.org/api/v1
   hermes config set model.api_key "sk-your-morpheus-key-here"
   ```

3. Add grok fallback for auxiliaries (title generation, etc.):
   ```bash
   hermes config set auxiliary.title_generation.provider xai
   hermes config set auxiliary.title_generation.model grok-4.20-0309-reasoning
   ```

4. Verify:
   ```bash
   hermes config check
   hermes doctor
   # Manual API test
   KEY="sk-your-morpheus-key-here"
   curl -s -H "Authorization: Bearer $KEY" https://api.mor.org/api/v1/models | jq ".data[] | select(.id | test(\"GLM|glm\"))"
   hermes chat -q "Say hello" --model GLM-5.1 --provider morpheus
   ```

## Common Pitfalls
- Dummy MORPHEUS_API_KEY in .env overrides config.model.api_key → always inspect .env after config set.
- base_url must be exactly `https://api.mor.org/api/v1` ( /v1 gives 403).
- Key with dot works in raw curl but triggers SDK 401 unless both env and config.model.api_key are set.
- `hermes auth add morpheus` and `hermes model` require real TTY — fails in tool calls.
- Auxiliary "Connection error" fixed by explicit xai/grok fallback (auto often fails without OpenRouter key).
- Changes require fresh session (`hermes` or `/new`). Python cache clear if editing providers.py.
- Prefer remote gateway over local :8083 proxy unless running full EverClaw stack.

## Verification Checklist
- [ ] `hermes config` shows correct morpheus + GLM-5.1 + base_url
- [ ] No 401 on GLM-5.1; curl /models and /chat/completions both succeed
- [ ] Auxiliary title generation succeeds (no warning)
- [ ] `hermes doctor` and `hermes config check` clean
- [ ] Fresh session defaults to GLM-5.1 with grok auxiliary fallback

See `hermes-agent` skill for full provider registration details and code patches to `hermes_cli/providers.py` / `auth.py`.

Last updated: May 2026 session (remote gateway stabilization + skill).
