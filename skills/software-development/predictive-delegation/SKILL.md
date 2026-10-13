---
name: predictive-delegation
description: "Analyze message patterns and suggest proactive subagent delegation for parallelizable work."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, subagent, parallel, optimization, proactive, workflow, efficiency]
    related_skills: [spike, subagent-driven-development]
---

# Predictive Delegation

Use this skill when you detect **parallelizable work patterns** in the user's request that would benefit from proactive subagent delegation — before the user explicitly asks for it. This is a pure edge capability: it suggests delegation opportunities but never auto-delegates.

Load this when analyzing multi-part requests, batch operations, or tasks with independent subtasks that could run concurrently.

## When NOT to use this

- Single focused task with no independent subtasks
- Tasks with strict sequential dependencies
- User explicitly said "do this yourself" or "don't delegate"
- Work requires interactive back-and-forth (subagents can't ask questions)
- Task is trivially small (< 30 seconds expected runtime)

## Trigger Conditions

Activate this analysis when ANY of these patterns are detected:

1. **Batch file operations**: User mentions multiple files/directories to process independently
   - "Update all README files in these 5 repos"
   - "Run tests on modules A, B, and C"
   - "Refactor these 4 similar functions"

2. **Multi-repo/multi-project work**: References to separate codebases or projects
   - "Sync changes across frontend, backend, and docs"
   - "Apply this security patch to all services"

3. **Research + implementation split**: Clear separation between investigation and coding
   - "Figure out the best API approach and then implement it"
   - "Research X while setting up Y"

4. **Independent test/validation suites**: Multiple test groups with no shared state
   - "Run unit tests, integration tests, and e2e tests"
   - "Validate against staging and production configs"

5. **Parallel content generation**: Multiple outputs from the same input
   - "Generate changelogs for v1, v2, and v3"
   - "Write summaries for each of these papers"

## Decision Framework

Before suggesting delegation, verify ALL of these:

```
[ ] Subtasks are truly independent (no shared mutable state)
[ ] Each subtask has clear inputs and expected outputs
[ ] Subtasks don't require user interaction mid-flight
[ ] Total parallel time < sequential time (overhead justified)
[ ] No ordering constraints between subtasks
[ ] Error in one subtask shouldn't block others
```

If any check fails, do NOT suggest delegation.

## Suggestion Format

When suggesting delegation, present it as an **option**, not a directive:

```markdown
I notice this work has N independent parts that could run in parallel:

| # | Subtask | Est. Time | Delegate? |
|---|---------|-----------|-----------|
| 1 | [description] | ~Xm | Yes/No |
| 2 | [description] | ~Ym | Yes/No |

**Parallel estimate**: ~max(X,Y)m vs sequential ~sum(X+Y)m
**Trade-off**: Subagents can't ask clarifying questions; results need integration.

Want me to delegate these, or handle them sequentially?
```

## Integration with delegate_tool

When the user accepts delegation:

1. Use `delegate_task()` from `tools/delegate_tool.py` for each subtask
2. Provide complete context in each delegation prompt (subagents have no conversation history)
3. Specify expected output format for easy reintegration
4. Monitor completion via returned agent IDs
5. Synthesize results into a unified response

Example delegation call pattern:
```python
from tools.delegate_tool import delegate_task

agent_id = delegate_task(
    task_description="[Complete self-contained task description]",
    context={"files": [...], "requirements": [...]},
    timeout_seconds=300,
)
```

## Anti-Patterns

- **Never auto-delegate without user confirmation** — this is advisory only
- **Don't delegate dependent tasks** — if B needs A's output, keep sequential
- **Don't over-delegate** — 2 subtasks isn't worth the overhead; aim for 3+
- **Don't delegate interactive work** — subagents can't ask follow-up questions
- **Don't hide failures** — report subagent errors transparently

## Metrics

Track delegation effectiveness (optional, for self-improvement):
- Suggestion acceptance rate
- Time saved vs sequential execution
- Reintegration complexity (low/medium/high)
- User satisfaction signals (implicit or explicit)

This data feeds back into trigger condition refinement but is never persisted without explicit opt-in.