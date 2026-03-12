# Subagent-Driven Delivery System (Manager Mode)

## Purpose
This is the operating playbook for running Hermes as an execution manager that delegates implementation to subagents while enforcing:
- clean branches
- clean commits
- spec compliance
- code quality
- test reliability
- merge-ready handoff

If you tell Hermes **"start subagent-driven development"**, this is the default system it should execute.

---

## Operating Contract

### What you provide
At minimum:
1. product objective (what we are building)
2. constraints (stack, deadlines, non-goals)
3. acceptance criteria

### What Hermes owns end-to-end
1. decomposes work into task packets
2. creates/upgrades plan + todo tracking
3. delegates implementation to subagents
4. runs two-stage reviews per task (spec, then quality)
5. enforces branch + commit hygiene
6. validates tests and integration
7. prepares PR-ready summary and next actions

---

## Standard Workflow

## Phase 0 — Intake + Planning
1. Confirm objective, constraints, and acceptance criteria.
2. Produce/update a plan doc in `docs/plans/`.
3. Create todo items from the plan.
4. Define branch strategy and expected commit groups.

Output artifacts:
- `docs/plans/<date>-<initiative>-execution-plan.md`
- todo list with explicit task IDs

## Phase 1 — Task Packeting
For each task, Hermes creates a packet with:
- scope in/out
- files expected to change
- tests to add/run
- commit message target
- risks and rollback notes

Template: `docs/templates/subagent-task-packet.md`

## Phase 2 — Implementation Delegation
For each packet:
1. Dispatch implementer subagent with full context.
2. Require TDD or test-first where practical.
3. Require command transcript summary (tests run, files changed, commit made).

## Phase 3 — Two-Stage Review (mandatory)
After implementation, run both:
1. **Spec Compliance Review** — exact requirements met, no scope creep.
2. **Code Quality Review** — maintainability, safety, style, edge cases, test quality.

Do not proceed until both pass.

## Phase 4 — Integration + Stabilization
1. Rebase/sync branch as needed.
2. Run targeted tests for touched components.
3. Run broader regression set (or full suite when risk is high).
4. Resolve integration conflicts and re-run critical tests.

## Phase 5 — Delivery/Handoff
1. Prepare PR summary with:
   - what changed
   - why
   - evidence (tests/logs)
   - risks
   - rollback path
2. Provide final merge recommendation.
3. Queue next sprint tasks.

---

## Required Quality Gates (must pass)

1. **Gate A: Plan Clarity**
   - acceptance criteria are testable
   - each task has clear done-definition

2. **Gate B: Task Completion**
   - subagent output includes files/tests/commit hash
   - no unresolved questions in task packet

3. **Gate C: Review Pass**
   - spec review: PASS
   - quality review: APPROVED

4. **Gate D: Test Evidence**
   - required tests pass locally
   - no unexplained failures in adjacent suites

5. **Gate E: Hygiene**
   - branch naming follows policy
   - commit messages follow policy
   - no temp/debug artifacts

Reference checklist: `docs/templates/manager-review-gates.md`

---

## Manager Command Pattern (what to tell Hermes)

Use this prompt shape for best results:

```text
Start subagent-driven development for <initiative>.
Use docs/playbooks/subagent-driven-delivery-system.md as the operating policy.
Enforce docs/policies/branch-commit-pr-hygiene.md.
Generate plan + task packets, execute in phases, and keep clean commits/branches.
Run mandatory review gates and provide merge-ready summary.
```

Optional add-ons:
- `reasoning=high` for deep architecture or refactor work
- "parallelize independent tasks" when low file overlap

---

## Parallelization Rules

Safe to parallelize only if tasks do **not** overlap in files/modules.

- Parallel allowed: docs vs backend API, independent adapters, isolated tests.
- Sequential required: shared core modules, migrations, routing policy, common schemas.

If overlap risk is uncertain, execute sequentially.

---

## Stop Conditions (escalate instead of guessing)
Hermes must pause and ask for direction if:
1. product requirement conflict is discovered
2. security/data-loss risk appears
3. acceptance criteria become mutually incompatible
4. external dependency blocks progress > defined threshold

---

## Definition of Done (project-level)
A project cycle is done when:
- all planned tasks are completed or explicitly deferred
- all mandatory gates pass
- tests and verification evidence are recorded
- docs are updated for operations and future maintenance
- handoff includes clear next-step queue
