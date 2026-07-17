# PostgreSQL State Migration Runbook

This runbook migrates one Hermes profile's durable session state from its
SQLite `state.db` to an explicit PostgreSQL schema. It is an offline,
fail-closed migration: Hermes does not fall back to SQLite when
`sessions.state.backend` is `postgres`.

## Prerequisites

- Install the optional backend dependency: `hermes-agent[postgres-state]`.
- Provision a PostgreSQL 16+ database reachable by the Hermes runtime. The
  migration target role needs permission to create its dedicated state schema,
  a temporary staging schema, and the `pg_trgm` extension.
- Keep the DSN only in the runtime environment, for example
  `HERMES_STATE_POSTGRES_DSN`. Do not put a DSN in `config.yaml`, shell
  history, tickets, or logs.
- Choose an explicit lowercase schema name, unique to this profile, such as
  `hermes_default_state`. The target schema must not already exist; Hermes
  creates and owns it during publication. Do not share a state schema across
  profiles.
- Retain a tested SQLite backup and confirm PostgreSQL backup/PITR coverage
  before beginning. The migration itself creates a temporary SQLite backup
  through SQLite's backup API; it never copies WAL or SHM sidecars.

## Preflight And Dry Run

1. Stop or drain every Hermes writer for the target profile. This includes
   CLI sessions, gateways, cron runs, compression work, and delegated tasks.
2. Confirm no compression lease, pending handoff, or undelivered delegation is
   active. The migration rejects these states rather than copying live work.
3. Export the DSN in the process environment and run a dry run:

```bash
hermes migrate state-postgres \
  --schema hermes_default_state \
  --run-id state-postgres-20260716
```

4. Check the JSON report. It must finish in `dry_run`, identify only the
   intended schema and DSN environment-variable name, and expose no
   credentials.
5. Verify the target schema does not exist. The migration refuses every
   pre-existing target namespace, including an empty schema, rather than
   trusting inherited ownership, ACLs, defaults, routines, or types.

## Apply And Validate

Run the same command with `--apply` while writers remain stopped:

```bash
hermes migrate state-postgres \
  --apply \
  --schema hermes_default_state \
  --run-id state-postgres-20260716
```

The migration holds a SQLite writer fence through snapshot, verification,
publication, config cutover, and report persistence. It streams rows in
batches, verifies per-table row counts and SHA-256 digests, publishes the
schema atomically, then switches only `sessions.state` in `config.yaml`.
The previous raw configuration is retained beside `config.yaml` as
`config.yaml.bak-pre-state-postgres-<timestamp>`.

Before restarting writers, validate all of the following:

- The JSON report has `phase: "complete"` and each durable table has equal
  source/target count and digest values.
- The generated configuration contains `backend: postgres`, the configured
  `dsn_env`, and the exact intended schema, but no DSN value.
- A read-only Hermes startup against the configured schema succeeds before a
  writable gateway or router is started.
- A representative session, its messages, title lookup, and message search
  are visible in PostgreSQL.

Resume the normal Hermes deployment only after these checks pass.

## Recovery And Rollback

If publication completed but config cutover or report persistence was
interrupted, keep writers stopped and rerun the exact `--apply` command with
the same `--run-id`. Hermes reads the durable publish marker and verifies the
current fenced SQLite snapshot against the published counts and digests before
finishing cutover. Recovery also revalidates the published schema's ownership,
ACLs, and executable objects; any post-publication namespace change blocks
cutover for investigation. A changed SQLite source requires a new migration
run ID; do not force a stale run through.

If the PostgreSQL backend is unhealthy after cutover:

1. Stop Hermes writers and preserve the PostgreSQL schema and migration report
   for investigation. Do not delete the target schema as part of rollback.
2. Restore the `config.yaml.bak-pre-state-postgres-<timestamp>` file created
   by the successful cutover.
3. Confirm the retained SQLite snapshot/state is intact, then restart Hermes
   on SQLite deliberately.
4. Investigate and run a new migration only after the original failure is
   understood. There is no automatic SQLite fallback because silent split
   writes would corrupt recovery evidence.

The migration intentionally has no destructive `--replace` mode. Use a new,
empty PostgreSQL schema for a new attempt rather than overwriting evidence.
