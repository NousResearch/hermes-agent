# Manager Review Gates Checklist

Use this checklist after each task and before final merge.

---

## Gate 1 — Spec Compliance (PASS/FAIL)
- [ ] Task objective achieved exactly as defined
- [ ] Acceptance criteria all satisfied
- [ ] No missing required files/changes
- [ ] No unauthorized scope expansion

Outcome:
- `PASS` or
- `FAIL` with explicit remediation list

---

## Gate 2 — Code Quality (APPROVED/CHANGES REQUESTED)
- [ ] Code follows project patterns and conventions
- [ ] Naming and structure are clear
- [ ] Error handling is appropriate
- [ ] Security and safety concerns considered
- [ ] Tests are meaningful and non-flaky

Outcome:
- `APPROVED` or
- `REQUEST_CHANGES` with severity labels

---

## Gate 3 — Verification Evidence
- [ ] Required targeted tests passed
- [ ] Integration/regression checks passed (as required)
- [ ] No unexplained failures remain
- [ ] Syntax/compile checks pass for touched files

---

## Gate 4 — Hygiene
- [ ] Branch name follows policy
- [ ] Commit messages follow policy
- [ ] No secrets/temp/backup artifacts
- [ ] Diffs are scoped to intended changes

---

## Gate 5 — Delivery Readiness
- [ ] Operator docs updated (if behavior changed)
- [ ] PR summary includes test evidence and risks
- [ ] Rollback path documented
- [ ] Deferred items captured as explicit follow-ups

---

## Final Decision
- `MERGE_READY`
- `MERGE_READY_WITH_FOLLOWUPS`
- `NOT_READY`
