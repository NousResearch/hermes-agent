---
name: model-selection
description: "Guide for picking which AI model Hermes should use. Default + automatic fallback are configured in environment/config; this skill is only for cases where the user explicitly asks for a different model."
version: 2.0.0
author: Gordon Rouse
license: MIT
metadata:
  hermes:
    tags: [hermes, models, routing]
---

# model-selection

## Default behavior — no skill action needed

On every container start, the entrypoint enforces `model.default = openai-codex/gpt-5.5` (set via `HERMES_ENFORCED_MODEL` Railway env var). GPT-5.5 is served via the OpenAI Codex backend (`https://chatgpt.com/backend-api/codex`) using OAuth credentials tied to the user's ChatGPT subscription — no per-token API billing.

`fallback_providers` in `~/.hermes/config.yaml` is a single-entry chain: **MiniMax-M2.7** (paid per-token via the MiniMax base URL). If GPT-5.5 fails — quota exhaustion, network error, or detected model substitution (see below) — Hermes falls through to MiniMax silently.

Fallbacks are silent — no user-visible message, just keeps answering. Switch happens inside `agent/run_agent.py`'s fallback loop.

`auxiliary.compression.{provider,model}` is pinned to `gemini` / `gemini-3-flash` (using `GEMINI_API_KEY`). Compaction uses Gemini's large context window, so summarization is robust regardless of which main model is active.

Don't invoke this skill for routine work. The default + auto-fallback handles it.

## Strict model verification — prevents silent OpenAI downgrades

The Codex backend can server-side substitute a smaller model (e.g. `gpt-5.5-mini`) when the requested tier is rate-limited. To prevent this from happening invisibly, `run_agent.py` checks the `model` field on every Codex response and treats any non-matching family as a failure — triggering the explicit MiniMax fallback instead of accepting the downgrade.

Match rule: served model must equal requested model, OR match `{requested}-DIGIT...` (date-stamped versions like `gpt-5.5-2026-01-15`). Anything alphabetic after the family name (`-mini`, `-nano`, etc.) is treated as substitution.

Controlled by env var `HERMES_CODEX_STRICT_MODEL` (default `1`). Set to `0` to disable if it ever causes false positives.

### Cooldown + Telegram notifications

To avoid wasting one Codex call per turn during a sustained quota outage, detection triggers a per-model cooldown (`HERMES_CODEX_COOLDOWN_S`, default 3600 = 1 hour). While the cooldown is active, `_run_codex_stream` returns `None` immediately without hitting the API, routing to the fallback chain.

A Telegram message goes to `$TELEGRAM_HOME_CHANNEL` once per cooldown entry:
- **On substitution detected:** `⚠️ Codex model substitution detected — Routing to MiniMax for the next ~60 min, then probing GPT again.`
- **On successful recovery probe** (first matching response after the cooldown elapses): `✅ Codex back online — Resuming primary routing.`

If the post-cooldown probe still substitutes, the cooldown re-arms and another "off" notification fires. Steady-state during outage: ~1 wasted Codex call + 1 notification per cooldown period.

Logs to grep when debugging:
```
/opt/data/logs/agent.log
  WARNING - Codex model substitution detected: requested=gpt-5.5 served=gpt-5.5-mini — routing to fallback.
  DEBUG   - Codex cooldown active for gpt-5.5 (until 1778303456); skipping API call.
```

## When to manually switch

Use `/model <name>` (session-scoped) when:

| Situation | Action |
|---|---|
| User says "use MiniMax" / "use the cheap model" | `/model minimax/MiniMax-M2.7` |
| User says "go back to default" / "use GPT" | `/model openai-codex/gpt-5.5` |
| User asks for Claude / Sonnet / Opus | Tell them Anthropic models are not currently configured. The fork sunset Anthropic to consolidate on the ChatGPT subscription + MiniMax. Re-adding would require a separate Claude Pro/Max OAuth via `hermes auth add anthropic`. |
| User asks for "the smartest" / "think harder" | GPT-5.5 IS the top tier here. If they want more reasoning depth, suggest framing the request to invoke `xhigh` reasoning effort via prompt (the model supports low/medium/high/xhigh internally; default is `medium`). |

## Model identifiers

| Model | Identifier | Auth | Context | Notes |
|---|---|---|---|---|
| GPT-5.5 (default) | `openai-codex/gpt-5.5` | ChatGPT subscription OAuth (`/opt/data/auth.json`) | 272K tokens | Reasoning levels: low/medium/high/xhigh (default medium); image input supported |
| MiniMax-M2.7 (auto-fallback) | `minimax/MiniMax-M2.7` | API key (env: `MINIMAX_API_KEY`) | 200K tokens | Paid per-token; ~$10/mo subscription with generous quota |

## Switching mechanics

- `/model <name>` — switch **for this session only** (lives in `_session_model_overrides` in memory; lost on restart).
- `/model <name> --global` — write to `~/.hermes/config.yaml`. **But:** the entrypoint re-applies `HERMES_ENFORCED_MODEL` on every restart, so `--global` only sticks until next container start. To make a permanent change, update the Railway env var.
- The `/model` command refuses to switch while an agent loop is running (see `gateway/run.py`). Wait until the current request finishes.

## Verifying the current model

Logs (`/opt/data/logs/agent.log`) will show `API Response received - Model: <name>` per turn when verbose logging is on. The `model` field on Codex responses tells you exactly which variant the backend served. If you need to verify the current default without sending a turn, the entrypoint logs `[entrypoint] Enforced model.default = openai-codex/gpt-5.5` on every boot.

If the user asks "what model are you on?" mid-conversation, be honest: you can't reliably introspect from inside the agent loop. Describe the behavioral tell and suggest they grep the log.

## Mid-conversation switch — what to expect

Switching from GPT-5.5 (272K) down to MiniMax (200K) only triggers compression if the conversation has actually grown past ~170K tokens. For everyday chat, that's never. When it does fire, the compressor uses **Gemini 3 Flash** to summarize middle turns, preserving system prompt + first 3 turns + last ~20 messages.

Switching **upward** in context is free — no compression needed.

## When a fallback fires

You won't see it directly — the agent loop swallows the failure and retries with MiniMax. If you suspect GPT-5.5 is having issues, check `/opt/data/logs/agent.log` for `fallback_activated` lines or `Codex model substitution detected` warnings. Don't apologize to the user about a fallback that worked; only surface it if the *fallback also failed* and the request errored visibly.
