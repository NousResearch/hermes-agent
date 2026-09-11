# Holographic Operations (R6)

Local-only, deterministic, zero-LLM production operations for stock Holographic.
All entry points live in `plugins/memory/holographic/operations.py` and take a
`MemoryStore` (e.g. `provider._store`). No new CLI, no new config keys, no
network, no auto-wiring (startup/session hooks are untouched by design).

## Health (read-only)

```python
from plugins.memory.holographic import operations as ops
rep = ops.health_check(store)          # never writes
rep["status"]  # HEALTHY | DEGRADED | NEEDS_REPAIR | BLOCKED
```

11 sections: database, schema, indexes, lineage, temporal, trust, security,
cache, config, migration, storage. Every report carries `llm_calls: 0`.

Explicit startup entry point (wire it yourself; not auto-called):

```python
ops.startup_check(store, config={"db_path": "..."})
```

## Repair

```python
ops.repair(store, mode="CHECK")            # default, read-only
ops.repair(store, mode="REPAIR_DERIVED")   # rebuild missing HRR banks only
ops.repair(store, mode="REPAIR_SAFE")      # + incremental_vacuum
ops.repair(store, mode="REPORT")           # CHECK + one-line summary
```

Never deletes facts/history. Authoritative-data issues (bad trust, orphans,
suspicious content) are reported, not silently rewritten (human boundary).

## Backup / verify / restore / rotation

```python
meta = ops.create_backup(store, "/backups/dir")   # snapshot + .meta.json sidecar
ops.verify_backup(meta["path"])                   # checksum + schema + integrity
ops.restore_to_scratch(meta["path"], "/tmp/scratch.db")  # NEVER overwrites prod
ops.rotate_backups("/backups/dir", keep=5)        # never deletes last good backup
```

Backups are verbatim snapshots (fidelity first). They may contain user-stored
secrets — there is no local encryption facility, so protect backup files with
filesystem permissions. Metadata, events, stats and metrics never carry fact
content (tested). Note: `create_backup` is an explicit-mutation path — it
ensures ops tables and records a `BACKUP_CREATED` event in the production DB
(counts only, no content).

## Maintenance

```python
ops.run_maintenance(store, mode="NORMAL")   # health + stats + event prune
ops.run_maintenance(store, mode="LIGHT")    # + migration ensure + bank repair
ops.run_maintenance(store, mode="DEEP")     # + integrity + FTS consistency
ops.run_maintenance(store, mode="DEEP", budget_ms=60000)  # bounded; rerun resumes
```

Idempotent: crash mid-run, just rerun. Budget exceeded → `BUDGET_EXCEEDED` with
completed steps, no partial derived state (bank rebuilds are single writes).

## Migration / circuit / events / metrics

```python
ops.migration_status(store)     # NOT_REQUIRED|READY|RUNNING|COMPLETE|FAILED|BLOCKED
ops.ensure_migration(store)     # additive, idempotent, transactional
ops.circuit_state(store, "maintenance-DEEP")  # CLOSED|OPEN (3 failures)
ops.recent_events(store, limit=50)            # bounded at 200, content-free
ops.get_stats(store)            # redacted counts/sizes
ops.get_metrics(store)          # incl. llm_calls == 0
ops.hrr_available()             # False without numpy (bank checks vacuous)
ops.network_dependencies()      # [] always
```

## Production profile (existing keys only)

```yaml
memory:
  provider: holographic
  # auto_extract: false     # default; no LLM anywhere in this plugin
  # hrr_dim: 1024  hrr_weight: 0.3  min_trust_threshold: 0.3
```

Destructive actions (delete history, reset DB, cross-project merge, disabling
protection) are human-only — this module exposes no API for them.
