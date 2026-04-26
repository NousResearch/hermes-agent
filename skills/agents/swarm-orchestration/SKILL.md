---
name: swarm-orchestration
description: Decide when to fan a request across parallel sub-agents (a "swarm") vs solve it solo or via a single delegation. Hermes is the orchestrator — runs in-process by default, escalates to a connected Telegram bot swarm when the user wants visible activity.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [orchestration, multi-agent, parallelism, delegation, telegram-swarm]
---

# Swarm Orchestration

You are the **leader**. When a user asks you to do something complex, you choose between three execution modes:

| Mode | When | How |
|---|---|---|
| **Solve solo** | Single straightforward question, lookup, single-source answer | Just answer / use tools yourself |
| **Single delegation** | One bigger task you want a fresh subagent to handle in isolation | `delegate_task` |
| **Swarm (fan-out)** | The request decomposes into 2–8 *independent* angles | `hermes_swarm` (default) or `telegram_orchestrate_swarm` (if user wants live Telegram visibility) |

This skill teaches you how to recognise which mode fits, and how to decompose well when swarming.

## When to swarm — the deciding question

> *"Can I split this into 2+ sub-questions whose answers don't depend on each other?"*

If yes → swarm. Examples:
- "Research X across legal / market / technical angles" → 3 parallel angles
- "Draft 4 variants of this email" → 4 parallel drafts
- "Verify these 3 claims independently" → 3 parallel fact-checkers
- "Monitor these 5 feeds for X" → 5 parallel monitors
- "Compare A, B, C on dimensions D1, D2, D3" → 3 parallel comparisons (or 9 if cross-cells matter)

If the next step **needs the previous step's answer**, do not swarm — that's sequential reasoning, do it solo or via a single `delegate_task`.

## Default: `hermes_swarm` (zero setup)

`hermes_swarm` runs in-process. No Telegram, no manager bot, no roster — just decompose, fan out, get structured results back. **This is your default for any swarm-shaped request.**

```
hermes_swarm(
    objective="<the user's goal in 1-3 sentences>",
    subtasks=[
        {"goal": "<atomic angle 1>", "persona": "<short role hint>"},
        {"goal": "<atomic angle 2>", "persona": "<short role hint>"},
        ...
    ],
)
```

Per worker you can also pass `context` (extra grounding) and `toolsets` (allowed tool whitelist).

## Escalation: `telegram_orchestrate_swarm` (requires approval)

Use **only** when the user explicitly wants visible Telegram bot activity ("send me updates from each agent", "I want to watch them work in Telegram", or they've already been working with named fleet bots like `@research_legal_bot`). Requires a one-time manager-bot setup; if not configured, prefer `hermes_swarm`.

> **One-time fleet setup (operator side).** The user runs `hermes fleet setup` once to register a manager bot, then either:
> * `hermes fleet add <username>` — creates a NEW child bot through the Managed Bots deep-link flow (one tap to confirm).
> * `hermes fleet adopt --token <bot_token>` — adopts a bot the user **already** created in BotFather.  Use this when `add` says "username already taken."
>
> Both paths land the bot in the roster as `active`, ready for `telegram_orchestrate_swarm`.

**This variant has visible side-effects** (named bots posting into chats), so it requires explicit user consent — same pattern as `terminal_tool`'s dangerous-command flow. Two paths to run:

### A. Approval flow (default)

```
1. Call telegram_orchestrate_swarm(objective, subtasks)  ← no user_approved
   → returns {"status": "approval_required", "plan": {...}}
2. Surface the plan via the `clarify` tool:
     clarify(
        question="Run this Telegram swarm? <one-line summary of plan>",
        choices=["Yes, run as proposed", "Adjust the plan", "Cancel"],
     )
3a. User says yes → re-call with user_approved=true.
3b. User wants changes → adjust subtasks, re-call without user_approved
    (this loops back to step 2).
3c. User cancels → solve solo or use hermes_swarm.
```

### B. By-name request (skips approval)

If the user named specific bots in their request ("use @legal_bot and @market_bot"), pin every subtask to a `bot_username`. The tool treats this as the user already requesting by name — no approval prompt.

### Operator override

Power users can set `telegram_fleet.auto_approve: true` in `~/.hermes/config.yaml` (or `TELEGRAM_FLEET_AUTO_APPROVE=1`) to disable the prompt entirely. Default is **off** — never run a Telegram swarm without consent.

The Telegram variant adds:
- Each subtask runs as a named child bot (visible identity).
- A `report_chat_id` argument streams live status as each bot.
- Same `objective` / `subtasks` shape as `hermes_swarm`.

## How to decompose well

1. **Atomic, independent subtasks.** Each goal should stand alone. If two goals reference each other, merge them.
2. **3–5 workers is the production sweet spot.** Coordination overhead dominates beyond ~7. The cap is 16 — treat that as a ceiling, not a target.
3. **Balance the work.** Aim for similar effort per subtask. Wall-clock = slowest worker; one outlier worker means no speedup.
4. **Use personas / roles.** Each worker behaves better when its persona is concrete: "skeptical legal analyst", "market sizing specialist", "fact-checker focused on dates and numbers".
5. **Spawn a verifier.** For high-stakes answers, add a final subtask `{"goal": "Cross-check the other workers' answers for contradictions and unsupported claims", "persona": "skeptic / fact-checker"}` — a well-known pattern in multi-agent research; it works.
6. **Distil, don't dump.** Each worker returns one focused answer; you synthesise. Don't ask workers to "report everything they did."

> Prior art: these patterns trace back to hierarchical reinforcement learning (Sutton's Options framework, Feudal Networks), recent multi-agent scaling research, and Moonshot AI's published PARL methodology for Kimi K2.5/K2.6.  We adopt them as orchestration heuristics, not as a training regime.

## Anti-patterns — when NOT to swarm

- **Tightly-coupled / sequential**: "Read this file then patch it then commit it." → solve solo or use `delegate_task` once.
- **Trivial questions**: "What's the time?" → answer directly.
- **Fewer than 2 angles**: `hermes_swarm` will reject single-subtask plans (it would just be `delegate_task` with extra steps).
- **Over-parallelism**: 12 subtasks for a 3-angle problem dilutes specialisation. Match worker count to natural angles, not max capacity.
- **Round-tripping**: don't swarm to swarm — workers run as leaf delegates; they can't recurse back into more swarms.

## Reading the result

Both swarm tools return:

```json
{
    "objective": "...",
    "results": [{"goal": "...", "persona": "...", "response": "...", "duration_seconds": 1.2, "error": null}, ...],
    "metrics": {
        "workers": 4,
        "failures": 0,
        "critical_path_seconds": 12.3,
        "total_serial_seconds": 41.0,
        "parallel_speedup": 3.3,
        "wall_clock_seconds": 12.4
    },
    "summary": "...human-readable rollup..."
}
```

`critical_path_seconds` is what the user actually waited; `parallel_speedup` tells you whether the fan-out paid off. If speedup < 1.5×, you over-parallelised — next time use fewer workers or solve sequentially.

## After the workers return — your job is synthesis

You are the leader. The structured `results` array is for **you** to read and synthesise into the final user-facing answer. Don't paste raw worker outputs back to the user; weave them into a coherent reply that answers the original question, calling out contradictions surfaced by any verifier.

## Procedure (the short version)

1. Read the user's request.
2. Ask: are there 2+ independent angles? If no, answer solo or use `delegate_task`.
3. If yes, decompose into 2–8 atomic subtasks with clear personas.
4. (Optional) Add a verifier subtask if facts/numbers matter.
5. Pick the variant:
   - Default → `hermes_swarm` (invisible in-process, just runs).
   - Telegram → `telegram_orchestrate_swarm`. **First call returns `approval_required`**; surface the plan via `clarify`, get user consent, re-call with `user_approved=true`. Skip approval only if user pinned specific bots by name.
6. Synthesise the structured results into your final answer.

## Verification

You did this right when:
- The user's request decomposes naturally into your subtasks.
- Each worker has a concrete, distinct persona.
- `parallel_speedup` ≥ 2× (otherwise fewer workers next time).
- Your final answer reads as one coherent response, not a transcript of N workers.
