---
name: parallel-orchestration
description: "Decompose a raw high-level goal into parallel subagent workers, assign toolsets, and synthesize results into a unified output."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, orchestration, parallel, multi-agent, synthesis, fan-out]
    requires_toolsets: [delegation]
    related_skills: [subagent-driven-development]
---

# Parallel Orchestration

**Core principle:** One raw goal → N independent workers → one synthesized answer.

## When to Use

Use when a goal splits into two or more **independent** sub-goals too large to handle serially in your own context.

Use `subagent-driven-development` instead when you already have a plan file.  
Call `delegate_task` directly when you already know the task array.  
Handle it yourself when only 1 sub-goal exists — no parallelism, only overhead.

## Prerequisites

Requires the `delegation` toolset. Grandchild workers (workers spawning their own sub-workers) require `delegation.max_spawn_depth: 2` in `config.yaml`; default is `1`. Workers passed `role="orchestrator"` silently degrade to `"leaf"` when the depth limit is reached.

## Decompose

Write the breakdown out before calling anything so the user can redirect.

```
Goal: Compare Pinecone, Weaviate, and Qdrant on cost, latency, and DX
→ Worker A: Pinecone
→ Worker B: Weaviate
→ Worker C: Qdrant
→ You: reconcile and synthesize
```

## Assign Toolsets

| Task type | Toolsets |
|---|---|
| Web research, competitive analysis | `["web"]` |
| Code implementation (write + run) | `["terminal", "file"]` |
| Code audit / read-only review | `["file"]` |
| Browser-dependent tasks (JS-heavy, login flows) | `["browser"]` |
| Sandboxed code execution or testing | `["code_execution"]` |
| Image or document visual analysis | `["file", "vision"]` |
| Pure reasoning / synthesis | `[]` |

## Invoke

```python
delegate_task(
    tasks=[
        {
            "goal": "Research Pinecone: pricing, query latency, and developer experience. Return a structured summary with specific numbers.",
            "context": "User wants a vector DB comparison. Your scope is Pinecone only.",
            "toolsets": ["web"]
        },
        {
            "goal": "Research Weaviate: pricing, query latency, and developer experience. Return a structured summary with specific numbers.",
            "context": "User wants a vector DB comparison. Your scope is Weaviate only.",
            "toolsets": ["web"]
        },
        {
            "goal": "Research Qdrant: pricing, query latency, and developer experience. Return a structured summary with specific numbers.",
            "context": "User wants a vector DB comparison. Your scope is Qdrant only.",
            "toolsets": ["web"]
        },
    ]
)
```

Each result has `status` (`"completed"` / `"failed"` / `"interrupted"` / `"timeout"` / `"error"`). Gate synthesis on `status == "completed"`; only completed workers have usable `summary` output.

Pass workers exactly: the original user goal, the worker's specific scope, and prior phase output for sequential steps. Per-task `toolsets` override the top-level parameter.

## Synthesize

Structure the output around the user's original goal, not the worker breakdown.

- **Reconcile conflicts** — prefer: more recent source > specific measurement > primary source. When unresolvable, present both and mark the uncertainty.
- **Note every gap** — if a worker failed, say so: *"A and B covered; C was unavailable."*
- **Don't list workers** — synthesize into the user's framing, not "Worker A found..."

## Never

- Spawn a single worker when you can handle it yourself
- Use `role="orchestrator"` without `delegation.max_spawn_depth: 2` in config — silently degrades, no error
- Silently drop failed workers
- Present a partial result as complete
