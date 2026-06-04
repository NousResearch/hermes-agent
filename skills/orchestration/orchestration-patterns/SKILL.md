---
name: orchestration-patterns
description: Decision tree and patterns for when (and when not) to spawn subagents, plus fan-out and sequential templates. Trigger when deciding whether a task warrants multi-agent orchestration vs. doing it yourself.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, multi-agent, delegation, patterns]
    related_skills: [task-router, model-selection-router, cost-guard]
---

# Orchestration Patterns

> The whole question is: spawn a subagent, or do it yourself? Get that right first, then pick the shape.

## Trigger

Apply when you're holding a task that *might* be multi-agent work — anything that smells like "several things at once," "a different specialist," or "this'll take a while on its own."

## Spawn a subagent when

- The task needs a **different toolset** than your current profile (route it — see `task-router`).
- The task is **parallelizable** — independent lanes that can run at the same time.
- The task needs **>20 turns** to complete independently, so it deserves its own context budget.
- The task requires a **specialized profile** you aren't currently running as.

## Do it yourself when

- It's a **single-tool lookup**.
- It's a **quick edit under ~50 lines** your toolset already covers.
- The **answer is in context**.
- The task is **under ~5 steps** and on-profile.

If none of the spawn conditions hold, doing it inline is both cheaper and faster.

## Fan-out pattern (parallel)

When a goal decomposes into N **independent** subtasks:

1. Decompose into N self-contained subtasks, each with its own profile and acceptance criteria.
2. `delegate_task` them in **batch** so the dispatcher fans them out concurrently.
3. **Collect** the results as children complete.
4. **Synthesize** the collected outputs into the final answer yourself (or in one synthesis child).

Use this for "research these 4 angles," "check config across these 3 services," etc. Keep children unlinked so nothing serializes them.

## Sequential pattern (gated)

When task A produces the **input** for task B:

1. `delegate_task` A.
2. **Await** A's result.
3. `delegate_task` B with A's output threaded into its prompt.

Use this only for true data dependencies. Don't serialize lanes that could run in parallel.

## Anti-patterns

- **Spawning agents for trivial work.** A child agent costs roughly **5x** an inline turn. A one-line edit does not deserve a subagent — see `cost-guard` for the math.
- **Deep nesting.** This install caps `max_spawn_depth` at **2**. A child spawning a child spawning a child will hit the wall — keep your trees shallow and synthesize at depth 1.
- **Forgetting to set the profile.** A child with no `profile` inherits the parent's toolset, so it loads tools irrelevant to its task and wastes tokens. Always set `profile` on the child to match its work.
- **Serializing parallelizable lanes.** If B doesn't actually need A's output, run them as a fan-out batch, not a sequential chain.
- **Fan-out with no synthesis step.** Collecting 5 child outputs and dumping them raw on the user is not orchestration — you still owe a synthesis.

## Depth limit

`max_spawn_depth = 2` is configured on this install. Plan your task graph to fit: you (depth 0) can spawn children (depth 1) who can spawn grandchildren (depth 2), and that's the floor. If a problem seems to need deeper nesting, flatten it — decompose more at the top instead.
