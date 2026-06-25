---
name: model-selection-router
description: Pick the right model tier before starting a task so cheap work runs on the cheap model and only genuinely hard work burns the full model. Trigger this reflexively at the start of any substantial task, before the first real tool call.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, model-selection, cost, routing]
    related_skills: [task-router, cost-guard, orchestration-patterns]
---

# Model Selection Router

> Choose the model tier that matches the work. Defaulting everything to the full model is the single biggest avoidable cost on this install.

## Trigger

Apply this skill **before any substantial task begins** — the moment you've understood what the user wants and before you fire the first implementation tool call. It's a two-second classification, not a ceremony.

Skip it only for a single-turn answer you can give from context with no tools.

## The three tiers on this install

| Tier | Model | Use for |
|------|-------|---------|
| **Cheap** | `deepseek/deepseek-v4-flash` | Status checks, simple lookups, short completions, single-file edits under ~50 lines, summarizing something already in context, mechanical reformatting |
| **Mid** | `qwen/qwen3-32b` | Reasoning tasks, moderate code changes, multi-step plans under ~5 steps, comparing a few options, writing a focused doc |
| **Full** | `openai/gpt-oss-120b` | Architecture decisions, complex/multi-service debugging, new MCP server development, multi-file refactors, anything where a wrong call is expensive to unwind |

The cheap model is also the `delegation.model` default and the smart-routing `cheap_model`, so cheap-tier work and most delegated subtasks already land there for free.

## Concrete examples

**Cheap (`deepseek/deepseek-v4-flash`):**
- "What's the disk usage on hermes?" → one lookup, cheap.
- "Bump the version string in `pyproject.toml`." → trivial single-file edit.
- "Summarize this error log I pasted." → already in context.
- "List the profiles configured here." → status check.

**Mid (`qwen/qwen3-32b`):**
- "Add input validation to this endpoint and explain the edge cases." → moderate code + reasoning.
- "Draft a 4-step migration plan for this table." → bounded multi-step plan.
- "Which of these three caching strategies fits our QPS?" → small comparison.

**Full (`openai/gpt-oss-120b`):**
- "Design the orchestration layer for the new agent fleet." → architecture.
- "This race condition only shows up under load across two services — find it." → complex debugging.
- "Build a new MCP server that wraps the Fastmail API." → new server dev.
- "Refactor the dispatcher to support goal-mode across all worker profiles." → multi-file refactor.

## How to apply

- **Launching a fresh run / delegating:** pass the flag, e.g. `--model deepseek/deepseek-v4-flash` (or the mid/full id).
- **Mid-conversation:** `allow_self_model_switch` is `true` on this install, so call the `model_switch` tool to drop down to cheap for a stretch of mechanical work, or step up to full when a task turns out harder than it looked. Switch back down when the hard part is done.
- **Delegated subtasks** inherit `delegation.model` (cheap) by default — only override to mid/full when the child task genuinely needs the reasoning.

## Anti-patterns

- **Defaulting to the full model for everything.** This is the failure mode this skill exists to prevent. A status check on `gpt-oss-120b` costs ~10x what it should.
- **Staying on the full model after the hard part is done.** If you escalated to debug something and you're now just writing the one-line fix, switch back down.
- **Escalating preemptively "just in case."** Start at the tier the task looks like; step up with `model_switch` only when you actually hit the wall.
- **Overriding `delegation.model` to full for cheap fan-out.** Ten cheap children are the whole point of cheap delegation — don't make them expensive by reflex.
