# PHASE-0 PROBE — the "fable→opus flip" is a REFUSAL-triggered client-side failover

**Date:** 2026-07-07 · **Owner:** Apollo · **Status:** RESOLVED — P4 replaced by reason-surfacing.

## What I first got wrong (and Ace correctly challenged)

Initial P4 hypothesis: the fable→opus "flip" is a SILENT server-side safety reroute, detectable
only by comparing requested vs served `response.model`. I probed `response.model` on benign +
keyword-flagged prompts, saw it always echo `claude-fable-5`, and wrongly concluded "un-buildable,
the reroute is invisible." That was **testing the wrong signal** — I never actually triggered a flip.

## The real mechanism (ground-truthed against the live subscription boxes + harness + logs)

1. fable **genuinely refuses** hard-blocked content — direct probe to a subscription box
   (`:18801/anthropic`) with a Pentagon/bioweapon prompt returned **`stop_reason: refusal`** (HTTP 200,
   empty content). (Keyword-flagged-but-answerable prompts like "keylogger" return `max_tokens`, which
   is why my first probes missed it.)
2. The harness maps that refusal to `finish_reason == "content_filter"` (conversation_loop.py:1615)
   and calls **`_try_activate_fallback()`** — the CLIENT-SIDE failover — to retry on the next
   configured model (opus). That IS the "flip."
3. Live logs confirm it: `agent.chat_completion_helpers: Fallback activated: claude-fable-5 →
   claude-opus-4-8 (claude-api-proxy-f1)` fired repeatedly on a real session this morning.

So the flip is a **real, observable client-side failover triggered by a safety refusal** — already
announced in-session via `_emit_fallback_announce` (morning spec, `announce_route_change`, default-on).
There is NO silent server-side reroute. P4-as-a-new-detector was solving a non-problem.

## What actually ships (replaces P4): reason-surfacing

The failover announce said `🔄 Model fallback: fable → opus` with NO reason — indistinguishable from a
rate-limit failover. The fix threads the `FailoverReason` into `_emit_fallback_announce` so the line
reads `🔄 Model fallback (safety refusal): claude-app/claude-fable-5 → claude-api-proxy-f1/claude-opus-4-8`
vs `(rate limit)` / `(provider overloaded)` / etc.

## Cross-method coverage (Ace's architecture question)

The refusal handler is WIRE-AGNOSTIC — both transports normalize a refusal to the same
`finish_reason == "content_filter"` branch, so the reason-surfacing covers every harness-driven method:

| Method | Wire | Refusal normalization | Reason-surfaced |
|---|---|---|---|
| claude-app (:18810) | anthropic | `stop_reason:refusal` → `content_filter` (transports/anthropic.py:239) | ✅ |
| claude-api-proxy-fN | anthropic | same | ✅ |
| claude-bpp (:18811) | openai | `message.refusal` / `content_filter` → `content_filter` (chat_completions.py:699) | ✅ |
| claude-bridge-fN | openai | same | ✅ |
| claude-code-relay | (Claude Code CLI) | `claude -p` runs its OWN agent loop; a fable refusal there is handled by Claude Code, NOT this harness failover | ➖ N/A |

Live refusal-passthrough probe verified: `:18801` (anthropic) and `:18810` (claude-app) both return
`stop_reason: refusal` untouched; the openai-wire path maps `message.refusal`/`content_filter` the same.
