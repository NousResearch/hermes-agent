# Swarm Router & Roles

## Overview

Two modules that handle model routing and agent role definitions for the swarm system.

## `swarm/roles.py` — Agent Role Definitions

Defines `AgentRole`, a frozen dataclass with fields:

| Field                | Type         | Description                          |
|----------------------|--------------|--------------------------------------|
| `name`               | `str`        | Role identifier (e.g. `"planner"`)   |
| `description`        | `str`        | What this role does                  |
| `default_model`      | `str`        | Model to use when no router present  |
| `system_prompt_suffix` | `str`      | Appended to system prompt            |
| `tools_allowed`      | `list[str]`  | Which tools this role may invoke     |
| `max_tokens`         | `int`        | Output token limit                   |
| `temperature`        | `float`      | Sampling temperature                 |

### Predefined Roles

| Constant     | Model (default)  | Purpose                                       |
|--------------|------------------|-----------------------------------------------|
| `PLANNER`    | `claude-opus`    | Decomposes goals into subtask graphs          |
| `EXECUTOR`   | `gemini-flash`   | Runs tools, writes code, produces artifacts   |
| `CRITIC`     | `claude-opus`    | Reviews output for correctness and style      |
| `VERIFIER`   | `claude-sonnet`  | Pass/fail validation against acceptance criteria |
| `MERGER`     | `claude-sonnet`  | Resolves conflicts across multi-agent outputs |
| `RESEARCHER` | `gemini-flash`   | Web search and document gathering             |

### API

```python
from swarm.roles import get_role, list_roles, PLANNER

role = get_role("executor")   # returns EXECUTOR
all_roles = list_roles()      # returns all six roles
```

## `swarm/router.py` — Budget-Aware Model Routing

`ModelRouter` selects the best model for a given role and task type while respecting a USD budget.

### Constructor

```python
from swarm.types import SwarmConfig
from swarm.router import ModelRouter

config = SwarmConfig(budget_limit_usd=5.00)
router = ModelRouter(config)
```

Set `budget_limit_usd=0` (default) to disable budget enforcement.

### Methods

#### `select_model(role, task_type=None, budget_remaining=None) -> str`

Picks the best model. Resolution order:
1. If `task_type` has a routing entry, those candidates come first.
2. Role-based candidates are appended.
3. If a budget is active, models whose estimated probe cost exceeds the remaining budget are filtered out.
4. First surviving candidate wins.

#### `estimate_cost(model, input_tokens, output_tokens) -> float`

Returns estimated USD cost using built-in per-model pricing (per 1M tokens).

#### `log_usage(model, input_tokens, output_tokens, cost, role="", task_type="")`

Records a completed request for spend tracking.

#### `get_spend_summary() -> dict`

Returns a dict with keys: `total_cost_usd`, `budget_limit_usd`, `budget_remaining_usd`, `total_input_tokens`, `total_output_tokens`, `num_requests`, `by_model`, `by_role`, `by_task_type`.

#### `is_within_budget(estimated_cost) -> bool`

Returns `True` if adding `estimated_cost` would stay within the configured limit. Always `True` when no limit is set.

### Default Routing Rules

| Role       | Model preference (first wins)     |
|------------|-----------------------------------|
| planner    | claude-opus, gpt-4o              |
| critic     | claude-opus, gpt-4o              |
| executor   | gemini-flash, gpt-4o-mini        |
| verifier   | claude-sonnet                    |
| merger     | claude-sonnet, claude-opus       |
| researcher | gemini-flash, gpt-4o-mini        |
| local      | qwen3.5-9b (zero cost)           |

### Pricing Table

| Model          | Input ($/1M) | Output ($/1M) |
|----------------|-------------|---------------|
| claude-opus    | 15.00       | 75.00         |
| claude-sonnet  | 3.00        | 15.00         |
| claude-haiku   | 0.25        | 1.25          |
| gpt-4o         | 2.50        | 10.00         |
| gpt-4o-mini    | 0.15        | 0.60          |
| gemini-flash   | 0.10        | 0.40          |
| qwen3.5-9b     | 0.00        | 0.00          |
