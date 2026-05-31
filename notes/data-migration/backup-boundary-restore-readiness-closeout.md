# HAD-1155 backup boundary and restore-readiness closeout

Issue: HAD-1155, Data migration: backup-boundary and restore-readiness closeout.

This artifact closes the planning gap between the data-disk dry-run planner and
the operational migration runbooks. It defines the backup boundary for the
planned `/data/hermes` layout, the retention and exclusion decisions around
holdbacks and caches, and the restore-order assumptions after bare OS recovery.

This document is planning-only. It does not execute migration commands, mutate
host disk state, change systemd or Docker configuration, or update live backup
jobs.

## Dependencies and current PR state

The backup boundary depends on these related migration artifacts:

- HAD-1147 / PR #132, path-contract inventory: expected to define the final
  path contracts and any additional roots that must be reconciled before the
  backup boundary is treated as complete.
- HAD-1150 / PR #130, maintenance-window and writer-quiescence runbook: expected
  to define the writer stop/start order and the pre-copy quiet-state gates.
- HAD-1148 / PR #131, dry-run planner already on `main`: currently proposes the
  managed roots `/data/hermes/profile-default`, `/data/hermes/stacks`,
  `/data/hermes/ctx-data`, and `/data/hermes/codex-home`.

Until HAD-1147 and HAD-1150 merge, treat this closeout as the backup and restore
policy for the known dry-run roots plus a reconciliation checklist for any later
inventory additions.

## Backup classes

Use these classes in backup manifests, restore checklists, and follow-up Linear
issues.

| Backup class | Meaning | Restore target |
|---|---|---|
| `critical-durable` | Required to resume Hermes, Codex, ctx, gateway, cron, or active project work. Must be backed up and restore-tested. | Restore before services start. |
| `important-durable` | High-value history, notes, research, source materials, provenance, or generated records. Must be backed up unless a manager accepts a documented omission. | Restore before normal operator work resumes. |
| `operational-history` | Logs, cron output, reports, and evidence that improve auditability but are not required for boot. | Restore after critical state, before holdback cleanup. |
| `rebuildable` | Virtualenvs, dependency caches, build outputs, and downloaded toolchains that can be recreated from source plus credentials. | Restore only if cheaper than rebuild. |
| `cache-excluded` | Temporary caches, socket files, runtime locks, WAL/SHM residue after clean shutdown, and tool spill files. | Excluded from durable backup. |
| `holdback-retained` | Local rollback copy created during migration. Not the new source of truth and not a recurring backup source. | Retain only until backup verification passes. |

## `/data/hermes` backup boundary

The known `/data/hermes` subvolumes are the dry-run planner targets. If HAD-1147
adds another subvolume or path contract, add it to this table before migration.

| `/data/hermes` path | Logical path preserved | Backup class | Back up | Retain | Exclude | High-value data and notes |
|---|---|---|---|---|---|---|
| `/data/hermes/profile-default` | `/home/david/.hermes` | `critical-durable` with `operational-history` subtrees | Yes | Yes | Select caches only | Hermes config, state DB, sessions, gateway state, cron jobs, cron output, skills, plugins, notes, backlog, provenance records, implementation delegate data, self-improvement records, VM-worker metadata, and audit/evidence files. Secrets under `.env` must be backed up through the secret-safe backup path and never printed in logs. |
| `/data/hermes/stacks` | `/home/david/stacks` | `critical-durable` and `important-durable` | Yes | Yes | Per-repo build artifacts and dependency caches | Project repos, WIP branches, `.git` history, source materials, ontology research, decision-room data under repo trees, generated ops material, and Hermes source/runtime checkout. Dirty repo state must be inventoried before backup acceptance. |
| `/data/hermes/ctx-data` | `/home/david/.ctx-data` | `critical-durable` | Yes | Yes | None by default | ctx data, worktrees, workspace metadata, and session/runtime state needed by Hadto and Hermes coordination. Treat missing ctx data as restore-blocking if ctx-backed sessions are in active use. |
| `/data/hermes/codex-home` | `/home/david/.codex` | `critical-durable` | Yes | Yes | None by default | Codex metadata, auth/config state, run metadata, and local agent state. This is secret-sensitive and must be encrypted or otherwise protected in backup storage. |
| `/data/hermes/*/cache`, where present | Same as parent | `cache-excluded` unless promoted by inventory | No, except explicitly named durable subtrees | Local cache may remain | Yes | Exclude media/tool caches, temporary downloads, package caches, and runtime spill files unless HAD-1147 identifies a cache subtree as the only copy of source material. |
| `/data/hermes/*/logs`, where present | Same as parent | `operational-history` | Yes with bounded retention | Yes with retention limit | Old rotated logs after retention window | Logs are useful for audit and regression diagnosis. They do not block service boot if current state and config are restored. |
| `/data/hermes/.migration-evidence` or equivalent evidence directory, if created | Evidence-only | `operational-history` | Yes | Yes | No | Store manifests, checksums, dry-run output, writer-quiescence evidence, backup verification reports, and restore rehearsal reports. |

## High-value data checklist

The backup is not acceptable until these data families are accounted for in a
manifest or explicitly named as out of scope with owner signoff.

| Data family | Expected location | Class | Restore concern |
|---|---|---|---|
| Provenance records | `/data/hermes/profile-default/notes`, backlog, implementation delegate, self-improvement, Hadto plugin state, and project repo records | `important-durable` | Required for auditability and continuity of prior decisions. |
| Ontology research | `/data/hermes/stacks`, especially ontology and decision-room repos or data trees | `important-durable` | May include research materials not reproducible from public source. |
| Session state | `/data/hermes/profile-default/state.db`, sessions, gateway state, and related SQLite files | `critical-durable` | Must pass SQLite verification after restore before Hermes gateway starts. |
| Cron output | `/data/hermes/profile-default/cron/jobs.json` and `/data/hermes/profile-default/cron/output` | Jobs are `critical-durable`; output is `operational-history` | Jobs resume automation; output preserves report history and context chaining. |
| Project repos | `/data/hermes/stacks` | `critical-durable` | Restore repo contents and `.git` metadata before services or agents use those paths. |
| Source materials | `/data/hermes/stacks` and Docker service storage named in follow-ups | `important-durable` | Some source materials may live outside `/data/hermes` in Docker named volumes and must not be silently assumed covered. |
| ctx data | `/data/hermes/ctx-data` and `/data/hermes/profile-default/ctx` | `critical-durable` | Restore before ctx-backed sessions, Hadto coordination, or workspace hygiene run. |
| Codex metadata | `/data/hermes/codex-home` and repo-local `.git/hermes-codex` under stacks | `critical-durable` | Restore before Codex or delegated workers resume; protect as secret-sensitive metadata. |

## Retention and exclusion decisions

Retention decisions:

- Keep all four managed subvolumes in recurring backup scope:
  `/data/hermes/profile-default`, `/data/hermes/stacks`,
  `/data/hermes/ctx-data`, and `/data/hermes/codex-home`.
- Keep operational-history data with bounded retention: Hermes logs, cron
  output, data-migration evidence, backup verification reports, and restore
  rehearsal reports.
- Keep local migration holdbacks only as temporary rollback material until the
  holdback gate below passes.

Exclusion decisions:

- Exclude temporary tool spill paths such as `/tmp/hermes-results` and backend
  temp directories. They are not part of `/data/hermes`.
- Exclude package caches, media caches, node/python build caches, virtualenvs,
  and generated build outputs unless a path-contract inventory explicitly
  promotes a subtree to durable data.
- Exclude stale SQLite WAL/SHM residue only after writer quiescence and database
  verification have proven the database is clean. Active WAL/SHM files are a
  migration blocker, not disposable backup noise.
- Exclude holdback directories from recurring backup jobs after the migration
  source of truth is accepted. They are retained locally for rollback only.

## Holdback gate

Expected holdback directories follow the HAD-1148 dry-run naming shape:

- `/home/david/.hermes.pre-data-migration-holdback.${MIGRATION_ID}`
- `/home/david/stacks.pre-data-migration-holdback.${MIGRATION_ID}`
- `/home/david/.ctx-data.pre-data-migration-holdback.${MIGRATION_ID}`
- `/home/david/.codex.pre-data-migration-holdback.${MIGRATION_ID}`

Holdback directories can be removed only after all of these are true:

- The current recurring backup includes every backed-up `/data/hermes`
  subvolume listed above.
- A readback or restore rehearsal from backup storage proves that backup
  artifacts contain backup classes, restore order notes, exclusion decisions,
  retention decisions, and the holdback gate itself.
- SQLite/session verification passes for restored Hermes state.
- Repo readback for `/data/hermes/stacks` proves `.git` metadata and working
  tree state were captured or intentionally excluded with evidence.
- ctx and Codex metadata readback passes without exposing secrets in logs.
- Docker named-volume coverage is either completed through a follow-up issue or
  recorded as an accepted out-of-boundary risk.
- The manager explicitly accepts the backup verification evidence and closes the
  rollback window.

If any item fails, keep holdbacks and treat deletion as blocked.

## Restore-order assumptions after bare OS recovery

These assumptions define order, not exact commands.

1. Recreate the `david` user, base packages, backup client, Git, Python, Docker
   or container runtime, systemd user-session support, and any credential access
   needed to read encrypted backups.
2. Mount the data disk at `/data` before enabling Hermes, Hadto, ctx, Codex, or
   Docker services that depend on restored paths.
3. Restore `/data/hermes/profile-default`, `/data/hermes/stacks`,
   `/data/hermes/ctx-data`, and `/data/hermes/codex-home` from backup storage
   with ownership, modes, symlinks, xattrs, and secret protections preserved.
4. Recreate the logical path contracts so `/home/david/.hermes`,
   `/home/david/stacks`, `/home/david/.ctx-data`, and `/home/david/.codex`
   resolve to the restored data-disk content before any writer starts.
5. Verify Hermes state while services are still stopped: config readability,
   secret-file presence without printing contents, SQLite integrity, session
   visibility, cron job visibility, and gateway state readability.
6. Verify `/home/david/stacks`: expected repos are present, `.git` metadata is
   readable, dirty-state evidence is available, and the Hermes checkout can
   provide the runtime expected by systemd.
7. Verify ctx and Codex metadata before resuming delegated or VM worker flows.
8. Restore service-specific Docker named volumes and bind-mounted app data from
   their own backup plan before starting containers that depend on them.
9. Start lower-level storage consumers before top-level intake: Docker services
   that have verified storage, user timers/cron, and Hermes gateway last. This
   order is expected to be finalized by HAD-1150.
10. Run application smoke checks and a backup readback check before deleting any
    holdback or declaring the host fully recovered.

## Follow-up issues to create or confirm

These are named follow-ups for the manager or Linear owner to create or map to
existing issues. They are not created by this delegated run.

| Proposed issue title | Scope | Why it remains uncovered |
|---|---|---|
| `HAD-1158: Docker named-volume backup and restore plan for ontology and Hadto services` | Inventory and backup/restore Docker named volumes including `hadto-pipeline_pipeline-data`, `ontology-platform_archivebox-data`, `ontology-platform_oxigraph-data`, and `ontology-platform_source-materials-minio-data`. | Docker named volumes are outside `/data/hermes` and are not covered by home-path bind mounts. |
| `HAD-1159: Restore rehearsal for /data/hermes on a scratch host` | Prove a bare-OS restore can reconstruct logical paths, read Hermes state, inspect repos, and start services in the documented order. | This closeout defines assumptions but does not execute a restore. |
| `HAD-1160: Backup verifier manifest for /data/hermes backup classes` | Implement or document the recurring backup readback that proves critical-durable, important-durable, operational-history, retention, exclusion, and holdback-gate coverage. | Holdback cleanup depends on a repeatable verification artifact. |
| `HAD-1161: Reconcile backup boundary after HAD-1147 path-contract inventory merges` | Add any newly discovered `/data/hermes` subvolumes or host paths to the backup boundary and restore plan. | PR #132 is still open while this closeout is written. |
| `HAD-1162: Classify service-specific source-material stores outside /data/hermes` | Identify ontology/source-material stores not under `/data/hermes`, including Docker-backed object/blob stores and repo-adjacent data directories. | Source materials are high-value and may not all live under the four dry-run subvolumes. |

## Closeout checklist

- Backup classes are defined.
- `/data/hermes` subvolumes are classified as backed up, retained, or excluded.
- High-value data families are named: provenance, ontology research, session
  state, cron output, project repos, source materials, ctx data, and Codex
  metadata.
- Restore order after bare OS recovery is documented.
- Holdback deletion is gated on backup verification and manager acceptance.
- Uncovered Docker services and future subvolumes have named follow-up issues.
- No migration commands were executed to produce this artifact.
