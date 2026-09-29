# Employee surface exclusions

- Status: active
- Scope: `agent/employee_policy.py`, native command/skill/service boundaries, dashboard routes and controls
- Introduced: employee surface audit

## Downstream intent

Skills, Kanban, legacy personality controls and native schedule authoring are
absent from the employee product, including indirect loading and background
services. Native cron inspection/execution remains. Hindsight is the sole
memory provider; personal memory is per person, never global USER.md. Doctor
reports employee directories and per-person memory; it does not recreate skills
or SOUL.md.

## Reconciliation

Keep upstream implementations intact and preserve the small policy checks at
native entry points. New command aliases, loaders, UI routes or workers must
respect the same exclusions. Do not hide controls without also closing their
backend path. Preserve native project/environment prompting, adapters and
execution mechanics. Hindsight recalls against the current message before replying, including the
first substantive turn (`recall_sync: true`); retention batching is independent.

## Validation

Run employee surface/dashboard tests and Hindsight timing tests through
`scripts/run_tests.sh`; run dashboard checks. See
[implementation map](../../specs/implementation.md).
