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

Use this skill when the user wants to create, design, improve, validate, or deploy a Hermes Workflow from a plain-language goal.

## Core behavior

1. Clarify only if required. If the user gives enough goal/context, proceed.
2. Prefer dashboard/API `workflow_draft` from the user's plain-language goal whenever available.
3. Use dashboard/API `workflow_refine` for corrections, missing steps, profile changes, routing changes, and prompt improvements.
4. Review the generated spec with the user when approval is needed; otherwise continue with the requested validate/deploy/test flow.
5. Always validate before deploy.
6. Deploy only when the user asked for deployment or explicitly approves it.
7. After deploy, verify deployed state with `hermes workflow show <workflow-id> --json` or the dashboard definition view.
8. Only write YAML by hand when the user asks for advanced/manual authoring or when `workflow_draft` / `workflow_refine` tooling is unavailable.

## Review checklist for generated specs

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

## Prompt and cell guidance

Use the prompt/cell checklist to review and refine generated specs, not as a default reason to hand-author YAML first.

Useful cell-table view:

| Cell id | Type | Profile | Objective | Output contract | Next |
| --- | --- | --- | --- | --- | --- |

Prefer text-first `agent_task.prompt` strings. Use `${ input.foo }` and `${ node.cell.output.field }` placeholders for runtime context. Include an explicit JSON output contract inside each agent-task prompt.

Agent-task prompt pattern:

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

## Manual YAML fallback

Only write YAML by hand for advanced/manual requests or when draft/refine tooling is unavailable. Keep workflow YAML human-readable and do not require users to write raw JSON.

If manual authoring is needed:

1. Draft a compact cell table.
2. Write YAML under `.hermes/workflows/<workflow-id>.yaml` unless the user specifies another path.
3. Validate and fix until green:

```bash
hermes workflow validate .hermes/workflows/<workflow-id>.yaml
```

4. If the user asked to deploy, deploy and verify:

```bash
hermes workflow deploy .hermes/workflows/<workflow-id>.yaml
hermes workflow show <workflow-id> --json
```

5. If the user asked to test a manual run, create an input JSON file and run:

```bash
hermes workflow run <workflow-id> --input /path/to/input.json --json
hermes workflow tick --limit 10 --json
hermes workflow executions show <execution-id> --json
```

## Common pitfalls

- Do not default to raw YAML authoring when `workflow_draft` / `workflow_refine` can produce and revise the spec.
- Mixed text prompts require runtime interpolation support; if `${ input.foo }` appears literally in a Kanban task body, the workflow runtime is stale.
- Do not use JSON as the dashboard-facing editing format; use plain-language draft/refine and text prompts, with YAML/JSON as advanced import/export/debug formats.
- Do not create switch edges from `switch` directly for named cases; use `switch.case_name` or `switch.default`.
- Do not claim a workflow is deployed until `hermes workflow show <workflow-id> --json` returns it.
