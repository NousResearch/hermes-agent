---
name: swarm-orchestration
description: Use when a task is broad, long-running, high-stakes, or decomposes into independent workstreams. Coordinates delegate_task swarms with orchestrator subagents, scout/reviewer lanes, one-writer discipline, and bounded synthesis.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [swarm, delegation, orchestration, subagents, autonomy, review]
    related_skills: [claude-code, codex, subagent-driven-development, dispatching-parallel-agents]
---

# Swarm Orchestration

Use this skill when a task is large enough that one agent doing everything serially would be slow, brittle, or context-heavy.

Agent rule: initiate this yourself when it fits. If a task is broad, long-running, high-stakes, or naturally parallel, load this skill without waiting for the user to say "swarm" or to issue `/swarm`. The `/swarm` command is only a status/control surface, not the activation path.

## Trigger

Use Swarm when any of these are true:

- The task naturally splits into 3+ independent workstreams.
- The work will likely take more than 10 minutes.
- The task is high-stakes and benefits from independent review.
- Codebase exploration, research, QA, or debugging can be done in parallel.
- The parent agent needs to preserve context for synthesis instead of reading every intermediate result.

Do not use Swarm for:

- One or two direct tool calls.
- A single-file edit or tight serial debugging loop.
- Work where several agents would edit the same files without isolated worktrees.
- A sequential pipeline where step B depends on step A.
- Anything needing user interaction inside a child; subagents cannot clarify.

## Operating Model

Parent agent = controller.

The controller decomposes, assigns, monitors, and synthesizes. It should avoid doing all the work itself when parallel workers can produce better evidence faster.

Default topology:

```text
controller
  ├─ scout / reviewer agents: read-only investigation, design critique, QA
  ├─ writer agent: one implementation lane per branch/worktree
  └─ synthesis: parent combines results and chooses the next slice
```

Use role="orchestrator" only when a child must further decompose its own domain:

```text
controller
  └─ domain orchestrator, role="orchestrator"
       ├─ leaf scout A
       ├─ leaf scout B
       └─ leaf reviewer C
```

Leaf children cannot delegate further. Orchestrator children may spawn their own workers if the profile's delegation depth allows it. All children remain barred from user interaction and shared side-effect tools such as clarify, memory, send_message, and execute_code.

## How to Dispatch

For independent first-level work, use batch delegation:

```python
delegate_task(tasks=[
    {
        "goal": "Inspect the auth subsystem for likely failure points.",
        "context": "Repo path: /path/to/repo. Read-only. Return files inspected, findings, and confidence.",
        "toolsets": ["terminal", "file"],
        "role": "leaf",
    },
    {
        "goal": "Review the current diff for security and regression risks.",
        "context": "Repo path: /path/to/repo. Do not edit. Return critical/important/minor findings.",
        "toolsets": ["terminal", "file"],
        "role": "leaf",
    },
])
```

For a domain that needs its own fan-out, explicitly grant orchestrator role:

```python
delegate_task(
    goal="Coordinate backend API investigation across routes, data layer, and tests.",
    context="Repo path: /path/to/repo. Spawn leaves only for independent read-only investigation. Synthesize before reporting.",
    toolsets=["terminal", "file"],
    role="orchestrator",
)
```

The context must be self-contained. Include repo path, branch/worktree, constraints, exact verification commands, files to avoid, and expected output format.

## Coding Discipline

Use one writer lane per branch or worktree.

- Scouts and reviewers are read-only by policy. Hermes does not hard-enforce read-only lanes, so instruct those children not to edit and keep their tool/context scoped accordingly.
- Writers own the actual edit path.
- If two writers are needed, isolate them in separate git worktrees.
- Reviewers check spec compliance first, then code quality/security.
- Parent re-reads files changed by workers before editing.

Preferred coding pattern:

1. Scout agents inspect separate domains.
2. Parent chooses the smallest executable slice.
3. One writer implements with TDD where practical.
4. Two independent reviewers inspect the diff.
5. Parent runs verification and synthesizes next steps.

## Cost and Fanout Guardrails

- Start with 2-4 children. Use 6 only when workstreams are truly independent and the active profile's delegation.max_concurrent_children allows it.
- Use role="orchestrator" sparingly; most children should be leaves.
- Depth 2 is the normal ceiling when the active profile allows it. Many Hermes installs default lower, so check delegation.max_spawn_depth before assuming nested orchestration is available.
- Do not re-delegate the whole task to one child. Decompose or do it yourself.
- Prefer fast/cheap scouts for broad inspection and stronger models for synthesis/review when configured.
- Stop spawning when evidence converges.

## Output Contract for Workers

Ask each child to return:

- What it inspected or changed.
- Findings, ranked by severity.
- Exact files touched, if any.
- Verification commands run and results.
- Blockers or uncertainty.
- A one-line recommendation.

## Telegram / Long-Running Work

For long work from Telegram:

- Use Swarm when it materially reduces wall time or increases confidence.
- Keep status terse.
- If work continues beyond the current turn, make continuation durable with a background process or cron job.
- Do not restart the active gateway mid-task just to apply config; defer restarts until progress is durable.
- Nested orchestrator telemetry may be less complete than first-level child telemetry on some Hermes builds, so treat parent synthesis and explicit worker summaries as the source of truth.

## Verification

Before claiming completion:

- Run the relevant tests or verification command yourself.
- Do not trust worker summaries without checking git status/diff/tests.
- Use independent Claude review for hardening/security when changing autonomy behavior.
- Report actual evidence, not intent.
