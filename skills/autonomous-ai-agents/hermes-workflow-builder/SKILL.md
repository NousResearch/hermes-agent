---
name: hermes-workflow-builder
description: Use when creating, designing, improving, validating, or deploying Hermes Workflow graphs from a user goal.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [workflows, orchestration, kanban, planning, automation]
    related_skills: [hermes-agent, plan, subagent-driven-development]
---

# Hermes Workflow Builder

Use this skill when the user wants to create, design, improve, or deploy a Hermes Workflow graph.

## Core behavior

1. Clarify only if required. If the user gives enough goal/context, proceed.
2. Identify workflow inputs, cells, assigned profiles, routing decisions, waits, joins, and terminal outputs.
3. Prefer text-first `agent_task.prompt` strings. Use `${ input.foo }` and `${ node.cell.output.field }` placeholders for runtime context.
4. Include an explicit JSON output contract inside each agent-task prompt.
5. Keep workflow YAML human-readable. Do not require users to write raw JSON.
6. Validate the YAML with `hermes workflow validate <path>` before claiming it is ready.
7. Deploy only when the user asked for deployment or explicitly approves it.

## Workflow design checklist

- Workflow id: lowercase kebab/snake style, stable across versions.
- Triggers: start with manual unless schedule/event is explicitly needed.
- Inputs: list the fields the user must provide at run time.
- Cells:
  - `pass` for normalization and final summaries.
  - `agent_task` for work assigned to profiles.
  - `switch` for branch decisions.
  - `parallel` + `join` for independent workstreams.
  - `wait` only for intentional delays.
  - `fail` for terminal rejection states.
- Every `agent_task` must define:
  - `profile`
  - `title`
  - text `prompt`
  - output JSON contract in the prompt
- Edges must use `switch.case_name` or `parallel.branch_name` when leaving branch nodes.

## Authoring workflow

1. Draft a compact cell table:

| Cell id | Type | Profile | Objective | Output contract | Next |
| --- | --- | --- | --- | --- | --- |

2. Write YAML under `.hermes/workflows/<workflow-id>.yaml` unless the user specifies another path.
3. Validate:

```bash
hermes workflow validate .hermes/workflows/<workflow-id>.yaml
```

4. If validation fails, fix the YAML and validate again.
5. If the user asked to deploy:

```bash
hermes workflow deploy .hermes/workflows/<workflow-id>.yaml
hermes workflow show <workflow-id> --json
```

6. If the user asked to test a manual run, create an input JSON file and run:

```bash
hermes workflow run <workflow-id> --input /path/to/input.json --json
hermes workflow tick --limit 10 --json
hermes workflow executions show <execution-id> --json
```

## Agent-task prompt template

Use this structure for each cell prompt:

```text
You are the `<profile>` profile executing workflow cell `<cell_id>`.

Workflow goal:
<one sentence>

Cell objective:
<what this cell alone must do>

Available workflow context:
- Input: ${ input.<field> }
- Upstream output: ${ node.<previous_cell>.output.<field> }

Constraints:
- Stay inside this cell objective.
- Do not defer required work.
- Return JSON only.

Return JSON matching this contract:
{
  "summary": "string",
  "status": "string"
}
```

## Common pitfalls

- Mixed text prompts require runtime interpolation support; if `${ input.foo }` appears literally in a Kanban task body, the workflow runtime is stale.
- Do not use JSON as the dashboard-facing editing format; use text prompts and YAML definitions.
- Do not create switch edges from `switch` directly for named cases; use `switch.case_name` or `switch.default`.
- Do not claim a workflow is deployed until `hermes workflow show <id> --json` returns it.
