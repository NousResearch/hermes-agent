---
name: taskmaster
description: Orchestrate complex tasks with automatic model routing.
version: 1.0.0
author: Rafael Zendron (rafaumeu)
license: MIT
metadata:
  hermes:
    tags: [orchestration, delegation, model-routing, task-management]
    related_skills: [kanban-orchestrator, kanban-worker]
    requires_toolsets: [delegation]
---

# TaskMaster Skill

Orchestrate complex multi-step tasks by automatically decomposing them into subtasks and routing each subtask to the most appropriate LLM. Uses `delegate_task` with per-task model overrides so a single orchestration call can leverage different models for different work types — fast/cheap models for simple tasks, capable models for complex reasoning.

Does NOT implement a new runtime or framework. It is a structured prompt + procedure that teaches the agent HOW to decompose and route tasks using existing Hermes primitives.

## When to Use

- User asks to "orchestrate", "coordinate", or "break down" a complex task
- User wants to run multiple subtasks in parallel with different models
- User wants cost-efficient execution by routing simple work to cheap models
- User asks for "taskmaster" by name

## Prerequisites

- Hermes config must have at least 2 providers configured (e.g., `openrouter`, `google`)
- `delegate_task` tool must be available (delegation toolset enabled)
- Optional: `model_switch` tool for mid-session adaptation

## Quick Reference

| Complexity | Model Family | Cost | Best For |
|---|---|---|---|
| LOW | gemini-2.0-flash, gpt-4o-mini | ~$0 | Formatting, extraction, simple lookups |
| MEDIUM | gemini-2.5-flash, gpt-4o, glm-4 | ~$0 | Summarization, categorization, drafting |
| HIGH | claude-sonnet-4, gemini-2.5-pro | paid | Reasoning, code generation, analysis |
| VOTE | 2-3 MEDIUM models | ~$0 | Quality gate via multi-model consensus |

## Procedure

### Step 1: Classify the task

Read the user's request and classify each identified subtask:

- **LOW**: Simple formatting, data extraction, list generation, translation, file I/O
- **MEDIUM**: Summarization, drafting, categorization, comparison, search synthesis
- **HIGH**: Complex reasoning, code generation, architectural decisions, multi-step analysis
- **VOTE**: Any subtask where accuracy is critical and the answer is uncertain

### Step 2: Build the task list

Construct a `delegate_task` call with per-task `model` overrides. Use the routing table above. Example:

```json
{
  "tool": "delegate_task",
  "input": {
    "tasks": [
      {
        "goal": "Extract key metrics from this report",
        "model": "google/gemini-2.0-flash-001",
        "toolsets": ["web", "terminal"]
      },
      {
        "goal": "Analyze trends and write executive summary",
        "model": "anthropic/claude-sonnet-4",
        "toolsets": ["web"]
      },
      {
        "goal": "Validate analysis with independent review",
        "model": "openai/gpt-4o",
        "toolsets": ["web"]
      }
    ]
  }
}
```

### Step 3: Fallback chain

If a model fails or is unavailable:

1. Try same-tier alternative (e.g., `gemini-2.0-flash` → `gpt-4o-mini`)
2. Try next tier up (e.g., `gemini-2.0-flash` → `gemini-2.5-flash`)
3. Fall back to parent agent model (inherited default)

### Step 4: Quality gate (optional)

For HIGH-complexity or critical tasks, run a VOTE pass:

- Send the same task to 2-3 MEDIUM models
- Compare results — if all agree, accept
- If they disagree, flag for user review

### Step 5: Synthesize

Collect all subtask results and present a unified response. If any subtask failed, report which ones and why.

## Model Routing Script

Use `scripts/route_model.py` to resolve the best model for a given complexity tier and available providers:

```
terminal: python3 optional-skills/autonomous-ai-agents/taskmaster/scripts/route_model.py --tier HIGH --provider openrouter
```

## Pitfalls

- **Don't over-route**: If a task has 2-3 simple subtasks, just run them all on the same model. Routing overhead isn't worth it for fewer than 3 subtasks.
- **Free model rate limits**: Some free-tier providers (OpenRouter free, Google) have rate limits. If you get 429 errors, fall back to a different provider.
- **Provider detection**: The `provider` field is optional in per-task overrides. If you specify just `model`, Hermes auto-detects the provider. Only specify `provider` when the model name is ambiguous across providers.
- **Context window**: Delegated subtasks start fresh — they don't inherit the parent's conversation. Pass all needed context in the `goal` and `context` fields.
- **Maximum 3 concurrent subtasks**: `delegate_task` runs up to 3 tasks in parallel. For more subtasks, batch them in groups of 3.

## Verification

1. All subtasks returned results (no failures)
2. Each subtask used the expected model (check tool output)
3. Synthesized response addresses the original user request completely
4. Total token cost stayed within expected range for the task complexity
