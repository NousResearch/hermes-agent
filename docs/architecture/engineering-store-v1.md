# Engineering Store V1 Persistence Contract

## Ownership

`EngineeringStore` owns durable engineering workflow facts. Hermes `SessionDB`
continues to own conversation and session persistence; neither store wraps or
duplicates the other.

`FileEngineeringStore` receives its root directory explicitly. It does not
select a project directory or discover `.hermes-engineering` itself.

## Disk layout and authority

```text
<root>/runs/<workflow_run_id>/
├── workflow.json
├── evidence.jsonl
├── verifications/attempt-<attempt>.json
└── reviews/attempt-<attempt>.json
```

- `workflow.json` is the current authoritative `WorkflowRun` snapshot. It is
  neither transition history nor an event log.
- `evidence.jsonl` is the append-ordered Evidence stream.
- Verification and Review files are immutable results, one per workflow
  attempt.
- Files matching temporary-file names are never read as authoritative records.

All canonical reads and writes cross the explicit conversion functions in
`engineering.store.records`. Persisted enums use their string values,
timestamps use timezone-aware ISO-8601 strings, and top-level records use
`schema_version = 1`. Verification and Review verdicts are omitted and
recomputed by their domain constructors.

## Create, update, and conflict semantics

- `create_workflow` creates a run directory and rejects an existing run.
- `save_workflow` replaces only an existing workflow snapshot.
- Evidence requires an existing parent workflow. Before append, the store
  linearly scans all run evidence and rejects an `evidence_id` already present
  anywhere under the configured root.
- Verification and Review require an existing parent workflow and reject an
  existing result for the same `workflow_run_id + attempt`.

These uniqueness checks are deterministic for the supported single-writer
model. Concurrent writers are unsupported; check-then-write races are not
prevented by locks in V1.

## Crash-safer snapshot replacement

Workflow, Verification, and Review snapshots are serialized to a uniquely
named temporary file in the canonical file's directory. The implementation
then flushes Python's stream buffer, calls `fsync` on the temporary file, closes
it, and invokes `os.replace`.

An exception before `os.replace` leaves an existing canonical snapshot
unchanged. The temporary file is removed on handled success or failure. A hard
process or machine crash can leave an orphan temporary file, but normal reads
ignore it and startup never promotes it automatically.

This is crash-safer atomic replacement, not full power-loss durability. The
implementation does not `fsync` the containing directory, so durability of the
directory entry across sudden power loss is not guaranteed.

## Evidence append and corruption

Each successful append writes one compact UTF-8 JSON record plus one newline,
then flushes and `fsync`s the evidence file before returning. A hard crash may
leave a truncated final line. Subsequent reads treat that line as corruption;
they do not skip it or return a partial Evidence sequence.

JSON decoding failures, unsupported schemas, and domain reconstruction failures
are wrapped as `EngineeringStoreCorruption` with the underlying cause retained.
Not-found errors remain distinct. Read methods construct and validate complete
domain objects before returning them.

## Path boundary

Workflow identifiers accept only a bounded safe character set without path
separators. Resolved run and record paths must remain under the configured
root, preventing traversal and absolute-path escape through an identifier.

## Deferred limitations

- thread/process-safe concurrent writers and file locking;
- recovery or automatic removal of hard-crash temporary files;
- truncated JSONL recovery;
- directory `fsync` and stronger power-loss durability;
- event history, migrations, indexes, artifact/raw-log storage, retention,
  cleanup, remote sync, and garbage collection.
