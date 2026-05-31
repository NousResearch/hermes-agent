# HAD-1161 backup-boundary reconciliation after HAD-1147

Issue: HAD-1161, Reconcile backup boundary after HAD-1147 path-contract inventory merges.

Manager verification for this reconciliation: HAD-1147 / PR #132 merged on
`origin/main` as `409081b533546af629d00e4c11b28fa787a58bb3`.

This artifact is planning-only. It does not execute migration commands, mutate
host disk state, change systemd or Docker configuration, update live backup
jobs, or inspect secrets.

## Source-of-truth files inspected

- `notes/data-migration/path-contract-inventory.md`: merged HAD-1147 path
  contract inventory from PR #132.
- `notes/data-migration/backup-boundary-restore-readiness-closeout.md`: HAD-1155
  backup boundary, restore order, holdback gate, and follow-up issue list.
- `scripts/data_migration_dry_run.py`: dry-run planner source for managed
  `/data/hermes` roots.
- `tests/scripts/test_data_migration_dry_run.py`: test coverage proving the dry
  run enumerates the four managed roots and treats Docker named volumes as
  inventory-only.
- `tests/scripts/test_data_migration_path_contract_inventory.py`: test coverage
  for the HAD-1147 inventory sections, required path classes, and decisions.

## Reconciliation result

No new `/data/hermes` subvolume is required after HAD-1147. Existing backup
boundary remains the four HAD-1155 roots:

- `/data/hermes/profile-default` for `/home/david/.hermes`
- `/data/hermes/stacks` for `/home/david/stacks`
- `/data/hermes/ctx-data` for `/home/david/.ctx-data`
- `/data/hermes/codex-home` for `/home/david/.codex`

The HAD-1155 restore-order assumptions still match the merged HAD-1147
inventory: restore the four `/data/hermes` roots, recreate the four logical
path contracts, verify Hermes/stacks/ctx/Codex state while writers are still
stopped, then restore service-specific Docker named volumes through their own
backup plan before starting dependent containers.

## Inventory reconciliation table

| HAD-1147 path or class | Backup-boundary classification | HAD-1161 decision |
|---|---|---|
| `/home/david/.hermes` and subpaths including `config.yaml`, `.env`, `state.db`, `logs`, `cron`, `skills`, `plugins`, `codex`, `ctx`, `vm-workers`, `backlog`, `implementation_delegate`, `notes`, `self_improvement`, `cache`, and `home` | Bind-mount backup through `/data/hermes/profile-default`; critical, important, and operational-history subtrees remain in scope; cache subtrees remain selective unless explicitly promoted by inventory. | No boundary change. This is already covered by the HAD-1155 `profile-default` root and restore plan. |
| `/home/david/stacks`, `/home/david/stacks/hermes-agent`, `/home/david/stacks/hermes-agent/venv`, and `/home/david/stacks/hermes-agent/.venv` | Bind-mount backup through `/data/hermes/stacks`; project source and WIP are durable, while virtualenvs and dependency caches are rebuildable unless cheaper to retain. | No boundary change. This is already covered by the HAD-1155 `stacks` root and restore plan. |
| `/home/david/stacks/smb-ontology-platform/ops` | Bind-mount backup through `/data/hermes/stacks`; generated ops feed files are important/live app data under the stacks parent. | No boundary change. Restore with stacks before starting containers that bind this path. |
| `/home/david/stacks/hadto-decision-room-data/decision-room` | Bind-mount backup through `/data/hermes/stacks`; mutable OWB data under the stacks parent is important source/application data. | No boundary change. Restore with stacks before starting containers that bind this path. |
| `/home/david/.ctx-data` | Bind-mount backup through `/data/hermes/ctx-data`; critical ctx worktree/session metadata. | No boundary change. This is already covered by the HAD-1155 `ctx-data` root and restore plan. |
| `/home/david/.codex` | Bind-mount backup through `/data/hermes/codex-home`; critical secret-sensitive Codex metadata. | No boundary change. This is already covered by the HAD-1155 `codex-home` root and restore plan. |
| `/home/david/.ops-agent/ops_heartbeat.sh` | Host path discovered by HAD-1147 but outside the current `/data/hermes` managed roots. If it moves in the same data-disk migration, it needs a separate bind-mount/root decision; if it stays in place, protect it through the existing host backup path. | Explicitly out of the HAD-1161 `/data/hermes` boundary. Track separately before any migration that changes ops automation paths. |
| Docker named volumes under `/var/lib/docker/volumes/...`: `hadto-pipeline_pipeline-data`, `ontology-platform_archivebox-data`, `ontology-platform_oxigraph-data`, and `ontology-platform_source-materials-minio-data` | Docker named-volume handling, not bind-mount backup through `/data/hermes`. | Outside this issue's boundary and assigned to HAD-1158. They must be backed up/restored before disk work or accepted as an out-of-boundary risk, as HAD-1155 already states. |
| `/home/david/.local/share/uv/python` | Explicit exclusion / no-move for this `/data/hermes` migration; rebuildable toolchain mounted read-only into the Docker sandbox. | No boundary change. Restore by existing home/toolchain backup or rebuild after host recovery. |
| `/home/david/.config/gh` and `/home/david/.config/git` | Explicit exclusion / no-move for this `/data/hermes` migration; secret-sensitive tool configuration outside the requested move set. | No boundary change. Preserve through non-Hermes host backup/secret handling before any home-wide operation. |
| `/opt/data`, `/home/pn/.codex`, and `/workspace` | Explicit exclusion / no-move derived container targets. Their host sources are `/home/david/.hermes`, `/home/david/.codex`, and the configured workspace bind mount. | No boundary change. Restore the host sources, then preserve the same container target contracts. |
| `/tmp/hermes-results` and `{env.get_temp_dir()}/hermes-results` | Explicit exclusion / no-move ephemeral tool output spill paths. | No boundary change. Do not add to recurring durable backup. |

## Restore-plan consequence

The HAD-1147 inventory adds no fifth `/data/hermes` restore root. The restore
plan should continue to restore `/data/hermes/profile-default`,
`/data/hermes/stacks`, `/data/hermes/ctx-data`, and
`/data/hermes/codex-home`, then recreate the logical paths
`/home/david/.hermes`, `/home/david/stacks`, `/home/david/.ctx-data`, and
`/home/david/.codex`.

The only newly explicit host-path action is classification, not a restore-order
change:

- Stacks-adjacent service data discovered by HAD-1147 is covered by
  `/data/hermes/stacks`.
- Docker named volumes remain outside bind-mount backup coverage and belong to
  HAD-1158.
- `/home/david/.ops-agent/ops_heartbeat.sh`, `/home/david/.local/share/uv/python`,
  `/home/david/.config/gh`, and `/home/david/.config/git` do not create new
  `/data/hermes` roots.

## Remaining unknowns and follow-ups

- HAD-1158 must still define backup and restore handling for
  `hadto-pipeline_pipeline-data`, `ontology-platform_archivebox-data`,
  `ontology-platform_oxigraph-data`, and
  `ontology-platform_source-materials-minio-data`. These are critical per
  service but outside this issue's bind-mount boundary.
- If a later migration expands from Hermes-managed paths to broader home/ops
  automation paths, decide whether `/home/david/.ops-agent/ops_heartbeat.sh`
  stays in place, receives its own backup assertion, or moves behind a separate
  bind mount.
- `/home/david/.config/gh`, `/home/david/.config/git`, and
  `/home/david/.local/share/uv/python` may be needed for operator ergonomics
  after restore, but HAD-1147 classifies them as outside the requested
  `/data/hermes` move set.
- Historical duplicate Compose files under `/home/david/stacks` should not be
  treated as active services without current container or systemd evidence.

## Verification

Run:

```bash
python -m pytest tests/scripts/test_data_migration_backup_reconciliation.py
```

This verifies that the HAD-1161 reconciliation artifact exists, cites
HAD-1147 / PR #132 and merge commit
`409081b533546af629d00e4c11b28fa787a58bb3`, names the relevant
`/data/hermes` roots, and classifies the HAD-1147 paths as bind-mount backup,
Docker named-volume handling, or explicit exclusion.
