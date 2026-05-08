---
name: model-selection
description: "Guide for picking which AI model Hermes should use. Default + automatic fallback are configured in environment/config; this skill is for cases where the user explicitly asks for a different model, or when a task warrants a deliberate switch (e.g. escalation to Sonnet for quality, or to Opus for hard reasoning)."
version: 1.1.0
author: Gordon Rouse
license: MIT
metadata:
  hermes:
    tags: [hermes, models, routing]
---

# model-selection

## Default behavior — no skill action needed

On every container start, the entrypoint enforces `model.default = MiniMax-M2.7` (set via `HERMES_ENFORCED_MODEL` Railway env var). MiniMax is paid per-token via the configured base URL — bulk-friendly, doesn't touch Pro/Max quota.

`fallback_providers` in `~/.hermes/config.yaml` is a two-step chain: **Sonnet 4.6 (Anthropic OAuth)** then **MiniMax-M2.7**. The chain serves two purposes:
- If MiniMax (default) fails → falls back to Sonnet (Pro/Max OAuth rescue).
- If the user has manually switched to Sonnet or Opus and that hits a rate limit → falls through Sonnet (one redundant retry when Sonnet is the active model) → lands on MiniMax (paid-tier rescue).

Fallbacks are **silent** — no user-visible message, just keeps answering. Switch happens inside `agent/run_agent.py`'s fallback loop.

`auxiliary.compression.{provider,model}` is pinned to `gemini` / `gemini-3-flash` (using `GEMINI_API_KEY`). Compaction uses Gemini's large context window, so summarization is robust regardless of which main model is active — even after a fallback to a smaller-context model, compaction still succeeds.

Don't invoke this skill for routine work. The default + auto-fallback handles it.

## When to manually switch

> **MANDATORY VERSION PINS — these override your training-time defaults.** When the user says "Sonnet" without a version, that means **`claude-sonnet-4-6`** (not 4-7, not anything newer). When the user says "Opus" without a version, that means **`claude-opus-4-6`** (not 4-7, not 4-1, not 4-20250514). The user has chosen these versions deliberately. Do NOT silently route to a newer Opus or Sonnet variant just because it exists — even if your training data tells you 4-7 is "the latest." Only deviate from these defaults if the user *explicitly* names a different version (e.g. "switch to opus 4-7").

Use `/model <name>` (session-scoped) when:

| Situation | Action |
|---|---|
| User says "use Claude" / "use Sonnet" / "use better model" | `/model anthropic/claude-sonnet-4-6` |
| User says "use Opus" / "use the smartest model" / "think harder about this" | `/model anthropic/claude-opus-4-6` |
| User explicitly says "use opus 4-7" / "use the latest opus" | `/model anthropic/claude-opus-4-7` |
| User explicitly says "use sonnet 4-7" / similar | use the named identifier verbatim |
| User says "use MiniMax" / "go back to default" | `/model MiniMax-M2.7` |
| You've gotten MiniMax wrong twice on the same problem | escalate: `/model anthropic/claude-sonnet-4-6`, retry |
| You've gotten Sonnet wrong twice on the same hard problem | escalate further: `/model anthropic/claude-opus-4-6`, retry |
| Multi-step planning across an unfamiliar domain, where mistakes cost real time | escalate to Sonnet (or Opus) before starting |
| User explicitly asks for a careful answer / important decision / nuanced judgment | escalate to Sonnet or Opus |

Don't escalate to Opus for routine chat, simple lookups, format conversions, or tasks Sonnet/MiniMax handled correctly. Opus burns Pro/Max quota the fastest — use it deliberately.

## Model identifiers

| Model | Identifier | Auth | Context | Notes |
|---|---|---|---|---|
| MiniMax-M2.7 (default) | `MiniMax-M2.7` | base_url (paid per-token) | 200K tokens | Bulk-friendly, no Pro/Max quota touched |
| Sonnet 4.6 (auto-fallback) | `anthropic/claude-sonnet-4-6` | Claude Pro/Max OAuth (`~/.claude/.credentials.json`) | 1M tokens | Best general balance; bills Pro/Max |
| Opus 4.6 (manual only) | `anthropic/claude-opus-4-6` | same OAuth | 1M tokens | Smartest; Pro/Max quota burns fastest |

## Switching mechanics

- `/model <name>` — switch **for this session only** (lives in `_session_model_overrides` in memory; lost on restart).
- `/model <name> --global` — write to `~/.hermes/config.yaml`. **But:** the entrypoint re-applies `HERMES_ENFORCED_MODEL` on every restart, so `--global` only sticks until next container start. To make a permanent change, the user has to update the Railway env var.
- `/model --provider anthropic` — auto-pick best Anthropic model. Useful if the user just wants a provider switch.
- The `/model` command refuses to switch while an agent loop is running (see `gateway/run.py`). Wait until the current request finishes.

**Always use the dashed form** for Anthropic identifiers (`claude-opus-4-6`, not `claude-opus-4.6`). The dotted form silently routes to an older variant — Anthropic's API treats `4.6` as a decimal that gets truncated to `4`, landing on `claude-opus-4-20250514` (Opus 4.0) instead of the intended `claude-opus-4-6` (Opus 4.6). Same applies to Sonnet.

## Mid-conversation switch — what to expect

Switching from a large-context model (Sonnet/Opus, 1M) down to MiniMax (200K) only triggers compression if the conversation has actually grown past ~170K tokens. For everyday chat, that's never. When it does fire, the compressor uses **Gemini 3 Flash** to summarize middle turns, preserving system prompt + first 3 turns + last ~20 messages. Older middle detail gets a lossy summary, not deletion.

Switching **upward** in context is free — no compression needed.

## When a fallback fires

You won't see it directly — the agent loop swallows the failure and retries with Sonnet. If you suspect MiniMax is down or rate-limited, check `/opt/data/logs/agent.log` for `fallback_activated` lines. Don't apologize to the user about a fallback that worked; only surface it if the *fallback also failed* and the request errored visibly.
