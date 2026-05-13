# IT Automation Lab Workflow

## Add a New Automation Procedure

1. Define the problem and target environment.
2. Create a runbook in `docs/runbooks/` using `docs/runbooks/template.md`.
3. Start with read-only discovery commands.
4. Identify risk level and required approvals.
5. Add or update a script only after the manual procedure is clear.
6. Add `--dry-run` for state-changing scripts.
7. Add tests or validation commands.
8. Update `scripts/README.md` if a new script is added.
9. Review the diff before committing.
10. Push only after approval.

## Script Design Rules

Scripts should be:

- idempotent when practical
- explicit about inputs and targets
- safe by default
- verbose enough for auditability
- quiet about secrets
- easy to run with `--help`
- testable without production credentials

## Runbook Design Rules

Runbooks should include:

- purpose
- scope
- prerequisites
- inputs
- risk level
- step-by-step procedure
- verification
- rollback or recovery
- troubleshooting
- related scripts

## Commit Guidance

Use conventional commits for lab-only documentation or scripts:

```bash
git add docs scripts README.md AGENTS.md
git commit -m "docs: add IT automation lab guide"
```

Do not mix unrelated code behavior changes with lab documentation unless the plan explicitly calls for it.
