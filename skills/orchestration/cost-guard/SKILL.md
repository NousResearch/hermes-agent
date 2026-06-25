---
name: cost-guard
description: Catch a task that's ballooning in tool calls or subagent count and force an explicit continue/decompose/stop decision. Trigger when a task has run past ~30 tool calls or when about to delegate more than 5 subagents.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, cost, budget, safety]
    related_skills: [model-selection-router, orchestration-patterns, task-router]
---

# Cost Guard

> Long-running tasks drift. This skill is the periodic gut-check that keeps a runaway loop from quietly spending the user's budget.

## Trigger

Apply when **either**:

- A single task has been running for **more than ~30 tool calls**, or
- You're about to **delegate more than 5 subagents** in one batch.

Don't wait for a perfect count — if it *feels* like this task has gone long, run the check.

## Four-tier check

| Tool calls | Action |
|------------|--------|
| **10–20** | Quiet self-check: is this still on track? Could a **cheaper model** finish the rest? (See `model-selection-router` — drop to cheap for mechanical tail work.) |
| **20–30** | Hard re-evaluation. Should this be **decomposed differently** — fanned out, or handed to a better-fit profile? Are you repeating yourself? |
| **30–40** | **Surface a summary to the user**: what's done, what's left, current approach. Ask if they want to continue as-is. |
| **40+** | **Stop.** Explain what's been accomplished and what remains. Don't push past this silently — get an explicit go-ahead. |

## Delegation cost multiplier

Use this to reason about fan-out:

- Each child agent at `deepseek/deepseek-v4-flash` ≈ **5% of a `gpt-oss-120b` turn**.
- So batching **10 cheap agents ≈ 0.5 expensive turns** — fan-out on the cheap model is genuinely cheap, *if the children are cheap-tier work*.
- The trap is the inverse: 10 children that each escalate to the full model is 10 full turns, not 0.5. Keep delegated work on `delegation.model` unless a child truly needs reasoning.

The takeaway: **wide cheap fan-out is fine; deep or expensive fan-out is not.** Five+ children is the point to consciously confirm they're all cheap-tier.

## Red flags (stop and reassess immediately, regardless of count)

- **Looping on the same tool call** with the same or near-same arguments.
- **Reading files repeatedly without progress** — re-reading the same file you already have in context.
- **Retrying a failed operation more than 3 times** without changing the approach. Three identical failures mean the approach is wrong, not unlucky.
- **Spawning children to escape your own confusion** — if you can't specify a child's task crisply, delegating won't fix it.

When a red flag fires, the fix is almost never "try harder on the same path." Step back, state the blocker plainly, and either change approach or ask the user.
