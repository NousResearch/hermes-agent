# Loop Engineering Development Pipeline

Status: reference workflow.
Owner: Loop owner / orchestrator.
Scope: non-trivial development work unless a human owner explicitly approves a bypass.

## Operating rule

```text
No SPEC_APPROVED → no coding.
No evidence → not done.
Merge ≠ product done.
```

Default chain:

```text
intake → analyst → spec-reviewer → coder → reviewer → QA → release/observe → learning loop
```

## Roles

| Role | Responsibility | Output |
|---|---|---|
| Orchestrator / Loop owner | route work, enforce gates, verify evidence | task graph, status, final evidence |
| analyst | turn messy request into testable spec | story map, `.feature`, examples, domain rules, questions |
| spec-reviewer | approve/reject spec before build | `SPEC_APPROVED` / change requests / blockers |
| coder | implement approved slices only | diff, tests, commands, handoff |
| reviewer | spec compliance + code quality | verdict, required fixes |
| QA | executable BDD/regression + evidence | coverage matrix, test output, report |
| release/observe | validate impact after release | metric/result/learning |

## Gate statuses

### Spec

```text
SPEC_DRAFTED
SPEC_APPROVED
REQUEST_SPEC_CHANGES
BLOCKED_ON_DOMAIN_QUESTION
REJECTED_NOT_WORTH_BUILDING
```

### Development

```text
IMPLEMENTED
PARTIAL
BLOCKED_ON_SPEC_CONFLICT
BLOCKED_ON_ENV
```

### Review

```text
SPEC_COMPLIANT
APPROVED
REQUEST_CHANGES
BLOCKED_ON_TECH_RISK
```

### QA

```text
QA_APPROVED
FAILED
PARTIAL
BLOCKED_FOR_ENV
BLOCKED_FOR_SPEC
```

### Product

```text
RELEASED
OBSERVING
VALIDATED
NEEDS_ITERATION
FAILED_HYPOTHESIS
```

## Handoff contracts

Every handoff must include concrete evidence:

```text
Outcome:
Artifacts / files:
Commands run:
Actual output:
Diff / commit / report path:
Open blockers:
Next gate:
```

## Default Kanban dependencies

```text
analyst → spec-reviewer → coder → reviewer → QA → release/observe
```

- `coder` depends on `SPEC_APPROVED`.
- `reviewer` depends on coder `IMPLEMENTED`.
- `QA` depends on reviewer `APPROVED`.
- `release` depends on `QA_APPROVED` or explicit `RELEASE_WITH_KNOWN_RISK`.
- `observe` depends on `RELEASED`.

## Exceptions

Tiny docs/typo/read-only tasks may use a lightweight path.

Emergency bypass must be explicit:

```text
HOTFIX_BYPASS
Reason:
Risk owner:
Rollback:
Follow-up spec/regression deadline:
```

Then create missing spec/regression work immediately afterward.

## Generator

Use the reusable generator to create the full Kanban chain:

```bash
scripts/create_loop_kanban_chain.py \
  --title '<feature/task title>' \
  --workspace dir:/path/to/repo \
  --request '<raw request/context>' \
  --artifact-root features
```

Default mode is dry-run and prints exact `hermes kanban create ...` commands.

Create real cards:

```bash
scripts/create_loop_kanban_chain.py \
  --title '<feature/task title>' \
  --workspace dir:/path/to/repo \
  --request-file /path/to/request.md \
  --execute
```

Useful options:

```text
--board <slug>          target a board
--test-first-qa         insert QA planning before coder
--no-observe            skip observe card
--dispatch              create cards and run one dispatcher pass
--idempotency-prefix X  stable dedup key for repeated runs
```

Verified smoke test on 2026-06-27 created and inspected this graph on a temporary board:

```text
analyst → spec-reviewer → coder → reviewer → QA
```

Smoke board and test cards were archived afterward.

## Canonical skill

Use `loop-engineering-development-pipeline` for future development orchestration.
