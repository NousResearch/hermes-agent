# Subagent Manager Operations Runbook

## Purpose
Day-to-day operating procedure for running Hermes as a development manager over subagents.

Related docs:
- `docs/playbooks/subagent-driven-delivery-system.md`
- `docs/policies/branch-commit-pr-hygiene.md`
- `docs/templates/subagent-task-packet.md`
- `docs/templates/manager-review-gates.md`

---

## 1) Kickoff Command (copy/paste)

```text
Start subagent-driven development for <initiative>.
Follow docs/playbooks/subagent-driven-delivery-system.md and enforce docs/policies/branch-commit-pr-hygiene.md.
Create/update a plan in docs/plans, generate task packets, execute with subagents, and run mandatory review gates.
Keep branches and commits clean and provide a merge-ready handoff.
```

For deeper planning/review quality:
```text
/ask reasoning=high <same request>
```

---

## 2) Expected Hermes Execution Behavior

1. Build or refresh plan in `docs/plans/`.
2. Open/refresh todo list for all tasks.
3. Execute tasks via delegate subagents.
4. Apply two-stage review per task.
5. Fix findings before moving forward.
6. Run integration verification.
7. Prepare final delivery summary with risks + follow-ups.

---

## 3) Clean Branch + Commit Flow

For each initiative:
1. create branch following naming policy
2. group commits by logical units
3. run tests before each commit
4. keep docs/tests in same branch as feature changes
5. no mixed unrelated changes

Recommended commit grouping:
1. implementation
2. tests
3. docs/runbooks
4. final polish/fixes

---

## 4) Required Delivery Output

At cycle end, Hermes should provide:
1. concise changelog
2. files/modules touched
3. test commands and results
4. risk + rollback note
5. pending/deferred queue
6. recommended next command to continue

---

## 5) Escalation Triggers

Hermes should explicitly pause and ask for decision when:
- requirements conflict
- security/privacy risk discovered
- migration/data model impact is unclear
- major architecture fork has tradeoffs

---

## 6) Quality Bar Summary

No merge recommendation unless all are true:
- spec compliance PASS
- quality review APPROVED
- verification evidence present
- branch/commit hygiene clean
- operator-facing docs updated
