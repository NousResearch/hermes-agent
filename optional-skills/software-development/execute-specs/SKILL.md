---
name: execute-specs
license: MIT
version: 1.4.0
author: afonsoft
platforms: [linux, macos, windows]
description: "Implement approved SPEC SDDs test-driven, slice by slice."
metadata:
  hermes:
    tags: [Spec-Driven, TDD, Implementation, SDD]
  version: "1.4.0"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Execute TDD from SPEC

Test-driven development guided by the approved SPEC SDD. The SPEC is the single source of truth. Every test is derived from a numbered requirement or acceptance criterion.

Internal reasoning and commands are in English. All questions and explanations to the user are in **Portuguese (pt-BR)**.

## When to Use

- A `.specs/SPEC-{YYYYMMDD}-{feature}.md` exists with `Status: Approved`.
- The user asks to implement a feature, change, or bugfix.
- The user asks to execute or run a SPEC (e.g., "execute the spec", "run the spec", "run TDD from the spec").
- O usuário pede para executar ou rodar um SPEC (ex.: "executar o spec", "rodar o spec", "fazer TDD a partir do spec").
- Before writing implementation code.

## When NOT to Use

- Do not use when the SPEC is missing or still in `Draft`.
- Do not use when the user only wants a code review or fix without tests.
- Do not use when the SPEC explicitly declines a requirement (marked `[A DEFINIR]`).

## Core Principles

1. **The SPEC is the oracle.** Every test must map to a requirement or acceptance criterion.
2. **One vertical slice at a time.** One failing test, one passing test, one refactor.
3. **No speculative code.** Write only the minimum code to make the current test pass.
4. **No refactor in RED.** Refactor only when the suite is green.
5. **Re-validate after every change.** Run the full suite before moving to the next slice.

## Process

### 1. Read the SPEC

Load `.specs/SPEC-{YYYYMMDD}-{feature}.md` and identify:

- Section 4 — numbered requirements (`RF-001`, `RF-002`, ...).
- Section 6 — acceptance criteria in BDD `Given...when...then`.
- Section 7 — task plan and validation strategy.
- Section 3 — files to create or modify.
- Section 0 — the `Ticket` field: resolve the linked GitHub Issue number (e.g. `#123`) if present.

If the SPEC is not approved, stop and invoke `/write-specs`.

### 2. Slice the work

Order the work by the task plan (Section 7). Each slice is one requirement or one acceptance criterion. Never write all tests upfront.

When the first slice starts, set the SPEC `Status` to `In implementation` and sync the linked Issue to `in_progress` (see GitHub Issue Status Sync).

### 3. Red-Green-Refactor loop

For each slice:

```text
RED    → Write one test that exercises a public seam and fails.
GREEN  → Write the minimum code that passes the test.
REFACTOR → Clean duplication, improve names, respect SOLID while green.
```

### 4. Test at the public seam

Test behavior through public interfaces (API, function, component), not internal implementation. Tests must survive internal refactors. See [references/tests.md](references/tests.md) for examples and [references/mocking.md](references/mocking.md) for mock rules.

### 5. Mapping to the SPEC

For each test, include a comment or description pointing to the source requirement:

```python
# Covers RF-003: order total must include tax
# Covers AC-02: Given an empty cart, when checking out, then the system returns 422
```

### 6. Re-validate and report

After every slice:

1. Run the full test suite for the affected stack.
2. Run lint / type check.
3. Confirm the SPEC requirement is satisfied.

Report progress to the user in Portuguese and continue automatically to the next slice. Do not ask for approval between slices; the approved SPEC is the source of truth.

```text
Slice concluido: [RF-XXX / AC-YYY]
- Teste: [PASS/FAIL]
- Build: [PASS/FAIL]
- Lint: [PASS/FAIL]

Avançando para o proximo slice: [PROXIMO].
```

### 7. Closure

When all slices are green:

- Run the full suite (unit, integration, relevant E2E).
- Run the validation strategy from the SPEC (Section 7.1).
- Update `docs/qa/test-plan-<feature>.md` or equivalent if it exists.
- Sync the linked Issue to `in_review` (see GitHub Issue Status Sync).
- Invoke `/qa-analyst` for the mandatory pre-PR review.
- Do not open the PR until QA approves.
- When the PR opens, sync the Issue to `in_pullrequest`. When the PR merges and delivery is finalized, sync to `done`, set the SPEC `Status` to `Done`, and close the Issue if GitHub did not close it automatically.

## GitHub Issue Status Sync

When the SPEC's `Ticket` field points to a GitHub Issue, keep its status label in sync with execution, per the Label Contract in `create-issues`. Status labels are exclusive — always remove the previous one when applying the next. If `gh` is unavailable or the SPEC has no linked Issue, skip the sync and continue — never let label management block TDD.

| Moment | Action |
| --- | --- |
| Implementation starts (first slice) | Swap current status label → `in_progress`; set SPEC `Status: In implementation` |
| All slices green (before the QA gate) | Swap → `in_review` |
| PR opened | Swap → `in_pullrequest` |
| PR merged and delivery finalized | Swap → `done`; set SPEC `Status: Done`; close the Issue if still open |
| SPEC canceled at any point | Swap → `canceled` + pt-BR comment with the reason; set SPEC `Status: Canceled` |

**Blockers** — PR merge conflict, failing CI that cannot be resolved in the slice, missing dependency, unanswered question, or any other impediment: keep the current status label, add `blocked`, and post a pt-BR comment describing the blocker. When the blocker clears, remove `blocked` and comment the resolution.

```bash
# status transition (exclusive)
gh issue edit 123 --remove-label todo --add-label in_progress

# flag a blocker
gh issue edit 123 --add-label blocked
gh issue comment 123 --body "Bloqueio: conflito de merge no PR #45 em src/foo.ts. Aguardando resolução."

# blocker cleared
gh issue edit 123 --remove-label blocked
gh issue comment 123 --body "Bloqueio resolvido: conflito do PR #45 resolvido no merge com develop."
```

After flagging a blocker, report it to the user in Portuguese and pause that work item — do not mark the slice as done while `blocked`.

## Anti-Patterns

- ❌ Writing all tests before any implementation.
- ❌ Testing private methods or internal state.
- ❌ Mismatched expectations in mocks (see [references/mocking.md](references/mocking.md)).
- ❌ Refactoring while a test is red.
- ❌ Skipping the validation strategy from the SPEC.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Implementing before the test | Write the failing test first. |
| The test does not map to the SPEC | Link every test to `RF-###` or `AC-###`. |
| Slice too large | Break the slice until one behavior is tested. |
| Not running the full suite after a green test | Re-validate before moving on. |
| Leaving the Issue status label stale | Sync the label at every transition (`in_progress` → `in_review` → `in_pullrequest` → `done`). |
| Hiding a blocker | Add `blocked` + a pt-BR comment on the Issue and report to the user; never silently stall. |

## References

- `write-specs` — for producing the SPEC SDD
- `qa-analyst` — for the mandatory pre-PR review
- `diagnose` — when a test fails unexpectedly and the cause is unclear
- `references/tests.md` — test examples and patterns
- `references/mocking.md` — mocking rules
- `references/refactoring.md` — refactoring guidance
- `references/deep-modules.md` — deep module design
- `references/interface-design.md` — interface design patterns
