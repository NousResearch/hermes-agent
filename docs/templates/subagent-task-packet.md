# Subagent Task Packet Template

Use one packet per implementation task.

---

## Task ID
`TASK-<number>`

## Title
Short imperative title.

## Objective
What outcome must exist when this task is complete.

## Scope In
- explicit items included

## Scope Out
- explicit non-goals

## Target Files
- `path/to/file_a.py`
- `path/to/file_b.py`

## Required Tests
- tests to add/update
- tests to run

Example commands:
```bash
source .venv/bin/activate
python -m pytest tests/<target> -q
```

## Implementation Notes
- constraints
- architecture decisions
- compatibility requirements

## Acceptance Criteria
- [ ] criterion 1
- [ ] criterion 2
- [ ] criterion 3

## Commit Target
Preferred commit message:
`type(scope): summary`

## Deliverable Format (required from implementer)
1. Summary of code changes
2. Files changed
3. Tests run + results
4. Risks/known limitations
5. Commit hash

## Review Instructions
### Spec Compliance Reviewer checks
- all acceptance criteria satisfied
- no scope creep
- file paths and interfaces match plan

### Code Quality Reviewer checks
- readability and maintainability
- edge-case handling
- test quality and coverage relevance
- security/safety concerns

## Rollback Note
How to safely undo this task if needed.
