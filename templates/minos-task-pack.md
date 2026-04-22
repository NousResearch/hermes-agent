# Minos task pack

Task id: <minos-task-id>
Run id: <run-id>
Minos owner: <named-owner>
Builder: <builder-name>
Human sponsor: <human-name-or-handle>

## Objective
- <one clear outcome>

## Scope
- In scope: <allowed work>
- Out of scope: <explicit non-goals>
- Expected artifacts (see `docs/minos-run-artifacts.md` for canonical names):
  - `builder-summary.md`
  - `gate-summary.md` when validation runs
  - `minos-decision.md` is produced later by Minos

## Workspace
- Repo/workdir: <absolute path>
- Allowed paths: <paths builder may modify>
- Forbidden paths: <paths builder must not modify>
- GitHub/remote policy: private only; unknown visibility = stop and escalate to Minos

## Inputs
- Required docs/specs:
  - <path or URL>
- Existing files to inspect first:
  - <path>

## Constraints
- Success criteria:
  - <observable requirement>
  - <observable requirement>
- Escalation triggers:
  - missing or conflicting context
  - change required outside allowed paths
  - unknown or public GitHub remote visibility

## Verifier commands
Replace these examples with task-specific commands before execution. Run them before stopping and record the result:
- Example: `pytest tests/test_target.py`
- Example: `python -m pytest -q`
- Example: `ruff check <path>`
- Example: `git diff --stat`

## Stop conditions
- Acceptance criteria met and verifier commands completed.
- Required context is missing or conflicting.
- A policy or safety issue is found, including unknown or public GitHub remote visibility.
- A change outside allowed paths is required.

## Handoff back to Minos
- Status: done | partial | blocked
- Files changed: <list>
- Verifier summary: <pass/fail with command output summary>
- Blockers or follow-ups: <list or none>
