---
name: model-selection
description: "Guide for picking which AI model Hermes should use. Default + automatic fallback are configured in environment/config; this skill is for cases where the user explicitly asks for a different model, or when a task warrants a deliberate switch (e.g. escalation to Opus)."
version: 1.0.0
author: Gordon Rouse
license: MIT
metadata:
  hermes:
    tags: [hermes, models, routing]
---

# model-selection

## Default behavior — no skill action needed

On every container start, the entrypoint enforces `model.default = anthropic/claude-sonnet-4.6` (set via `HERMES_ENFORCED_MODEL` Railway env var).

`fallback_providers` in `~/.hermes/config.yaml` is configured to fall back to MiniMax-M2.7 (OpenRouter) automatically if the primary provider fails — rate limit hit, OAuth expired, 5xx error, connection timeout. The fallback is **silent**: no user-visible message, just keeps answering. Switch happens inside `agent/run_agent.py`'s fallback loop.

Don't invoke this skill for routine work. The default + auto-fallback handles it.

## When to manually switch

Use `/model <name>` (session-scoped) when:

| Situation | Action |
|---|---|
| User says "use Claude" / "use Sonnet" | already default — confirm and continue |
| User says "use Opus" / "use the smartest model" / "think harder about this" | `/model anthropic/claude-opus-4.6` |
| User says "use MiniMax" / "stop using Anthropic" / "save my Pro/Max quota" | `/model MiniMax-M2.7` |
| You've gotten Sonnet wrong twice on the same hard problem | escalate: `/model anthropic/claude-opus-4.6`, retry |
| Multi-step planning across an unfamiliar domain, where mistakes cost real time | escalate to Opus before starting |
| Bulk processing (many similar tasks, e.g. processing a long list) | step down to MiniMax to preserve Pro/Max quota |

Don't escalate to Opus for routine chat, simple lookups, format conversions, or tasks Sonnet handled correctly. Opus burns Pro/Max quota faster — use it deliberately.

## Model identifiers

| Model | Identifier | Auth | Context | Notes |
|---|---|---|---|---|
| Sonnet 4.6 (default) | `anthropic/claude-sonnet-4.6` | Claude Pro/Max OAuth (`~/.claude/.credentials.json`) | 1M tokens | Best general balance |
| Opus 4.6 | `anthropic/claude-opus-4.6` | same OAuth | 1M tokens | Smartest; quota burns faster |
| MiniMax-M2.7 | `MiniMax-M2.7` | OpenRouter (`OPENROUTER_API_KEY`) | 1M tokens | Paid per-token; resilient floor |

## Switching mechanics

- `/model <name>` — switch **for this session only** (lives in `_session_model_overrides` in memory; lost on restart).
- `/model <name> --global` — write to `~/.hermes/config.yaml`. **But:** the entrypoint re-applies `HERMES_ENFORCED_MODEL` on every restart, so `--global` only sticks until next container start. To make a permanent change, the user has to update the Railway env var.
- `/model --provider anthropic` — auto-pick best Anthropic model. Useful if the user just wants a provider switch.
- The `/model` command refuses to switch while an agent loop is running (see `gateway/run.py`). Wait until the current request finishes.

## Mid-conversation switch — what to expect

Switching to a model with a **smaller context window** (currently moot — all three are 1M, but if a smaller-context model is added later) triggers `agent/context_compressor.py` on the next request. It summarizes middle turns to fit the new window, preserving system prompt + first 3 turns + last ~20 messages. Older middle detail becomes lossy.

Switching **upward** in context is free.

## When a fallback fires

You won't see it directly — the agent loop swallows the failure and retries with the next provider in `fallback_providers`. If you suspect Anthropic is down or rate-limited, check `/opt/data/logs/agent.log` for `fallback_activated` lines. Don't apologize to the user about a fallback that worked; only surface it if the *fallback also failed* and the request errored visibly.
