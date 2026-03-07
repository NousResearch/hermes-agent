---
name: hermes-acp-orchestrator
description: Orchestrate ACP-style delegation with explicit target routing to hermes, codex, or claude-code, including timeout-aware and output-bounded execution patterns.
version: 1.0.0
author: Hermes Agent (Nous Research)
license: MIT
metadata:
  hermes:
    tags: [acp, delegation, orchestration, codex, claude-code]
    related_skills: [codex, claude-code, hermes-agent]
---

# Hermes ACP Orchestrator

Delegate complex work through a single ACP-style orchestration pattern while keeping context isolated and output bounded.

Supported delegation targets:
- `hermes` (internal subagent, default)
- `codex` (external Codex CLI)
- `claude-code` (external Claude Code CLI)

## When to Use

Use this skill when you need one or more of the following:
- Explicit routing to a specific agent (`agent="codex"`, `agent="claude-code"`, `agent="hermes"`)
- Mixed-agent batch delegation (`tasks[].agent`)
- Safer external delegation with timeout and output-size controls
- Clear role split across implementation, review, and synthesis tasks

Prefer direct tool calls when the task is a single straightforward action.

## Core Patterns

### 1) Single-target delegation

```python
delegate_task(
    goal="Implement the fix and run the focused test suite",
    context="Keep changes minimal and list modified files.",
    agent="codex"
)
```

### 2) Mixed-agent parallel batch

```python
delegate_task(tasks=[
    {
        "goal": "Audit for regressions and edge cases",
        "agent": "claude-code",
        "toolsets": ["file"]
    },
    {
        "goal": "Apply patch and validate tests",
        "agent": "codex",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Summarize outcomes and open questions",
        "agent": "hermes",
        "toolsets": ["file"]
    }
])
```

### 3) Guardrail-first delegation

```python
delegate_task(
    goal="Refactor auth middleware for consistency",
    context="Return concise summary: changes, tests run, blockers.",
    agent="claude-code",
    max_iterations=30
)
```

## Execution Guidelines

1. Always include concrete context: paths, constraints, acceptance criteria.
2. Keep each delegated task narrow and testable.
3. Use per-task `agent` selection based on specialization.
4. Request concise output to reduce context pressure.
5. For external agents, ensure explicit completion criteria in the goal.

## Recommended Config

```yaml
delegation:
  max_iterations: 50
  default_toolsets: ["terminal", "file", "web"]
  external_timeout_seconds: 900
  external_max_output_chars: 24000
```

## Troubleshooting

- **Unsupported agent**: use only `hermes`, `codex`, or `claude-code`.
- **Timeouts on external targets**: increase `external_timeout_seconds` or split the task.
- **Output too large**: tighten expected output format in the goal/context.
- **Weak results**: provide stronger constraints, exact files, and pass/fail criteria.
