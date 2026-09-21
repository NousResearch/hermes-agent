# Fallback Change Policy

## Core rule — MUST-ASK

**Provider/auth/credential/fallback changes are ALWAYS must-ask.**

Do not modify fallback providers (global or per-profile) without explicitly presenting the change to the user and getting a greenlight. This applies even when:

- The change is technically correct (e.g. removing a broken provider)
- The change is for consistency with other profiles
- The change restores a previously-working state
- You believe the user would want this

## Why this is must-ask

1. **Credentials are live connections** — every profile and cron depends on provider auth. Changing fallbacks affects system reliability silently.
2. **User may have fixed the issue** — a provider showing auth-failed in one session may be re-authenticated by the user before the next session.
3. **Provider relationships change** — the user may have a specific reason to keep a provider in fallback (e.g. waiting for a key renewal, preferring one provider for certain model capabilities).
4. **Batch changes risk cascading** — changing fallback across 44 profiles simultaneously means a single misconfiguration affects every system at once.

## Process

1. Present the proposed change: current fallback chain → proposed fallback chain
2. State the problem (e.g. "OpenRouter shows 403 auth failed, fallback chain will skip it")
3. Let the user decide. Do not execute without approval.

## Current fallback chain (as of 2026-05-18)

Global and all 44 profile configs:

```
Primary: ollama-cloud (key pool, fill_first)
  ↓ failover
ollama-cloud → deepseek-v4-pro
  ↓ failover
nous → qwen/qwen3.6-plus
  ↓ failover
openrouter → nvidia/nemotron-3-super-120b-a12b
```
