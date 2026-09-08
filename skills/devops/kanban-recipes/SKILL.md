---
name: kanban-recipes
description: Instantiate reusable Kanban task graphs.
version: 1.0.0
author: Victor Kyriazakos (@victor-kyriazakos), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, workflows, recipes]
---

# Kanban Recipes Skill

Recipes create fresh native tasks from a portable JSON graph. They reuse the
existing dispatcher and task lifecycle, without custom statuses or a gate DSL.

## When to Use

Use a recipe for repeated research, writing, or implementation graphs with
explicit all-parent dependencies. Use board export/import to move execution
history, not to start another copy of a workflow.

## Prerequisites

- The operator has authorized starting the work. A successful run makes roots
  immediately dispatchable.
- Every role binds to an installed local profile. Check with `terminal` using
  `hermes profile list`. Do not invent profile names or modify credentials.
- Worktree tasks require an existing native project binding with a primary repo.
- Choose the board explicitly. Do not override a worker's inherited board pin.

## How to Run

Read `templates/brief.json` with `read_file`. Use `write_file` to create a local
copy and separate invocation files. For example, inputs can be
`{"topic":"SQLite transactions"}` and bindings can be
`{"roles":{"researcher":"default","writer":"default"}}`.

Run through `terminal`:

```bash
hermes kanban --board default recipe validate brief.json --inputs inputs.json --bindings bindings.json --json
hermes kanban --board default recipe run brief.json --inputs inputs.json --bindings bindings.json --key brief-request-001 --json
hermes kanban --board default recipe show <instance-id> --json
hermes kanban --board default recipe export <instance-id> --output portable.json
```

## Quick Reference

| Command | Effect |
|---|---|
| validate | Read-only parse, binding resolution and effective plan |
| run | Atomic receipt, native tasks and dependency links |
| show | Original invocation plus current task state |
| export | Portable definition only, no invocation data |

The run result contains `instance_id`, `replayed`, both digests, and a mapping
from node keys to new task IDs. Retain it as the delivery receipt.

## Procedure

1. Read the recipe with `read_file`. Verify every task body and effect against
   the operator's authorization. Parsing is not a sandbox for task instructions.
2. Bind all roles and supply declared inputs. Use a new key for new work, and
   retain the exact key for retries of that invocation.
3. Validate. Inspect the effective profiles, workspace policy and task controls.
4. Run once. If output fails or storage reports an uncertain outcome, inspect
   or replay the same key instead of choosing a new one.
5. Follow task state through native Kanban inspection and lifecycle tools.
   A recipe stage is a node key, not a status. Same-card review remains native.

## Pitfalls

- JSON only. Duplicate keys, unsupported versions/fields, cycles and malformed
  interpolation fail without a partial graph.
- Only title/body interpolate `{{input.NAME}}`. `{{{{` escapes literal `{{`.
  Input text is never evaluated again. Objects/arrays appear as untrusted JSON
  in worker context, not as substituted instructions.
- Same key plus changed definition, input or explicit binding is a conflict.
  Changed ambient defaults alone do not alter an existing instance.
- Archived tasks do not release a recipe key. Native parent readiness accepts
  archived parents, which is not proof of successful acceptance.
- Imported recipe history is fenced in triage. Export its definition and use
  a new key with explicit local bindings. There is no v1 resume/rebind command.
- Missing bound profiles block recipe tasks. Repair the profile or reassign
  deliberately, then use the normal operator unblock path.
- Export refuses an existing file unless `--overwrite` is explicit.
- Never put credentials in recipe inputs. Receipts retain those inputs, and
  board archives carry the historical invocation.

## Verification

Read back `recipe show` and compare its mapping with the run receipt. Confirm
that roots are ready and descendants wait for parents. A repeated identical key
must return the same instance and tasks with `replayed=true`. For another run,
use a distinct key and verify the IDs differ.
