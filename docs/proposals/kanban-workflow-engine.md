# Workflow Engine — Design Specification

> This document defines the Kanban workflow engine proposed in the companion PR.
> It is not an implementation; it serves as the design contract for review before code is written.

## Overview

Add an opt-in workflow engine to `dispatch_once` that automatically creates
downstream tasks when upstream tasks complete, based on a declarative YAML
workflow definition stored per board.

## Workflow definition schema

```yaml
# .hermes/kanban/boards/{board}/workflow.yaml
version: 1
enabled: true

# Global defaults applied to all auto-created tasks
defaults:
  workspace_kind: dir        # resolved from the completed task's workspace

rules:
  # Each rule matches on: assignee + optional mode field from task metadata
  # and produces downstream task(s) based on outcome (pass / fail)

  - trigger:
      assignee: product-manager
      mode: execute
    outcomes:
      pass:
        - title: "Review PM output (UX perspective)"
          assignee: ux-designer
          mode: review
          body_template: |
            Review docs/PRD-draft.md from a UX research perspective.
            Dimensions: feature completeness, user scenario coverage,
            state definitions, error handling.
            Pass → kanban_complete; Fail → kanban_block with itemized issues.

  - trigger:
      assignee: ux-designer
      mode: review
    outcomes:
      pass:
        - title: "UX research execution"
          assignee: ux-designer
          mode: execute
      fail:
        - title: "PM rework (UX feedback)"
          assignee: product-manager
          mode: rework
          body_template: |
            UX review did not pass. Revise docs/PRD-draft.md.
            Feedback is in the upstream task's comment.
```

### Schema fields

| Field | Type | Required | Description |
|---|---|---|---|
| `version` | int | yes | Schema version. Must be `1`. |
| `enabled` | bool | yes | Master toggle. `false` = skip all workflow processing. |
| `defaults` | object | no | Defaults merged into every auto-created task. |
| `rules` | list | yes | Ordered list of trigger → outcome mappings. |
| `rules[].trigger` | object | yes | Match criteria. |
| `rules[].trigger.assignee` | string | yes | Must match the completed task's assignee. |
| `rules[].trigger.mode` | string | no | Must match `metadata.mode` on the completed task. |
| `rules[].outcomes` | object | yes | Keys: `pass`, `fail`. At least one required. |
| `rules[].outcomes.pass` | list | no | Tasks to create on pass. |
| `rules[].outcomes.fail` | list | no | Tasks to create on fail. |
| `outcomes[][].title` | string | yes | Task title. |
| `outcomes[][].assignee` | string | yes | Profile to assign. |
| `outcomes[][].mode` | string | no | Stored in task metadata for downstream matching. |
| `outcomes[][].body_template` | string | no | Jinja2-like template for task body. Variables TBD. |

## Integration point in dispatch_once

```python
def dispatch_once(conn, ...):
    # ... existing reclaim / promote / spawn logic ...

    # NEW: workflow advance (after spawn, same tick)
    workflow_created = 0
    if workflow_enabled(board_slug):
        workflow_created = advance_workflow(conn, recently_completed_ids)
    result.workflow_created = workflow_created
    return result
```

## advance_workflow() algorithm

```
1. Filter recently_completed_ids to tasks whose status = done
2. For each completed task:
   a. Read workflow.yaml from board directory
   b. If not enabled → skip
   c. Match first rule where trigger.assignee == task.assignee
      and (trigger.mode is unset or trigger.mode == task.metadata.mode)
   d. Determine outcome:
      - If task was blocked/gave_up → outcome = "fail"
      - Else scan task's last comment + result for pass/fail keywords
      - Default: pass (task completed normally without block)
   e. Look up outcomes[outcome] in the matched rule
   f. For each task spec in the outcome list:
      - Create task via same code path as kanban_create
      - Set parent = completed task id
      - Merge defaults from workflow.yaml
   g. Record created task ids in result
3. Enforce max_chain_depth: count ancestors of each new task,
   abort if depth exceeds configured limit
4. Return count of created tasks
```

## Outcome detection

### MVP: keyword scan (Option A)

```yaml
kanban:
  workflow:
    outcome_pass_keywords: ["通过", "pass", "approved"]
    outcome_fail_keywords: ["不通过", "fail", "blocked"]
```

Scan the task's `result` field and the latest `comment` text. If any pass
keyword is found → pass. If any fail keyword is found → fail. If neither →
default to pass (task completed normally).

### Future: structured metadata (Option B)

Workers set `metadata.outcome = "pass"` or `"fail"` in their `kanban_complete`
call. Cleaner but requires worker cooperation. Can be added later without
breaking Option A.

## Configuration

```yaml
# config.yaml
kanban:
  workflow:
    enabled: false                  # global kill switch
    outcome_pass_keywords: ["通过", "pass", "approved", "done"]
    outcome_fail_keywords: ["不通过", "fail", "blocked", "block"]
    max_chain_depth: 20             # safety limit against infinite loops
```

Per-board override via `workflow.yaml`'s `enabled` field.

## Safety

- `max_chain_depth` prevents runaway chain creation (default 20)
- Only runs when `workflow.yaml` exists AND `enabled: true`
- One rule matched per completed task (first match wins)
- Fail path is explicit — no implicit rework loop
- All created tasks are visible via `kanban list` and the dashboard

## What this is NOT

- Not a general DAG engine (no parallel fan-out, no complex conditionals)
- Not a replacement for auto_decompose (that splits tasks; this chains stages)
- Not requiring changes to any existing tool or worker behavior

## Related

- Issue #25020: broader auto-orchestration vision. This is a narrower, 
  independently useful subset focused on deterministic chaining.
