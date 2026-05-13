# Runbooks

Runbooks describe repeatable IT operations before those operations are automated in scripts or delegated to an agent.

## Catalog

- `template.md` — copy this for new runbooks
- `system-inventory.md` — collect read-only host inventory
- `log-triage.md` — gather and inspect logs
- `service-health-check.md` — verify a service is healthy
- `backup-verification.md` — verify backups exist and are restore-ready

## Authoring Rules

Every runbook should document:

- purpose and scope
- risk level
- prerequisites
- exact commands or script references
- verification steps
- rollback or recovery steps
- known failure modes

Prefer read-only checks first. If a procedure changes state, document approval requirements and dry-run behavior.
