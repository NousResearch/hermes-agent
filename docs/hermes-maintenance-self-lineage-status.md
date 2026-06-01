# Hermes Maintenance Self/Lineage status notifications

Hermes Maintenance profile/project notifications must use generic two-line status output instead of role-specific labels.

Canonical Matrix/profile notification shape:

```text
Self status: WORKING|WAITING|BLOCKED|DORMANT — <short status for the speaking profile itself>
Lineage status: WORKING|WAITING|BLOCKED|DORMANT — <short aggregate status for structural descendants>
```

Do not use role-specific coordinator/PA/responsible status labels for these notifications. Do not use legacy blocker-only or negated-blocker wording. `BLOCKED` remains reserved for a concrete human-action dependency; non-human pauses render as `WAITING`, and taskless/no-active-assignment profiles render as `DORMANT`.

Lineage is structural responsibility (for example `pa_yunuen -> coordinator_hermes_maintenance -> responsible_*`), not Kanban task dependency. The machine-readable lineage registry lives at `docs/hermes-maintenance-lineage.json`; its `dependencies` key is intentionally separate.

Deterministic renderer:

```bash
python scripts/hermes-maintenance-status.py --profile coordinator_hermes_maintenance --board hermes-maintenance
```

For tests, dry-runs, or sample notifications, pass fixture task rows instead of reading the live Kanban DB:

```bash
python scripts/hermes-maintenance-status.py \
  --profile coordinator_hermes_maintenance \
  --tasks-json /path/to/tasks.json
```

Machine-readable output is available with `--json`.

Kanban compatibility mapping used by the renderer:

- `ready`, `queue`, `running`, `in_progress`, `review` -> `WORKING`
- `todo`, `scheduled` -> `WAITING`
- `blocked` -> `BLOCKED`
- `triage` -> `DORMANT`
- `done`, `archived` -> completion/history only, not live status

Aggregation default for lineage descendants:

1. `WORKING` if any descendant is working or immediately spawnable.
2. `BLOCKED` if none are working and at least one descendant is human-blocked.
3. `WAITING` if none are working/blocked and at least one descendant is waiting on non-human/system continuation.
4. `DORMANT` if no descendant has active/queued/waiting/blocked work.

The text includes concise counts/evidence so a `WORKING` aggregate does not hide branch-local blockers.
