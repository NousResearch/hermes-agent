# Authoritative worker-session provenance schema

Status: contract blueprint; no implementation

Ground truth: `origin/main` at `a61183b56fdb45b9d2a0f2f6b8482e665ccf702f` (fetched 2026-07-24)

Parent architecture: Kanban task `t_71bc4ffd`

## Scope and invariants

This contract gives later transcript maintenance a positive, persisted answer to one question: which exact Kanban board/task/run and profile state database own a Hermes session?

The answer is authoritative only when all of the following are true:

1. The board row is `attached`.
2. The active board's live `board_instance_id` equals the row's `board_instance_id`.
3. The active state database's live `state_db_instance_id` equals the row's `state_db_instance_id`.
4. A state-DB row exists for the same `session_id`, the referenced `sessions` row exists, and all seven columns match byte-for-byte.
5. The state-DB row is also `attached`.
6. The board task/run join still agrees that `run_id` belongs to `task_id`.

Anything else is unlinked or partial provenance and is ineligible for destructive maintenance. In particular, there is no legacy backfill from `sessions.cwd`, `sessions.source`, `parent_session_id`, timestamps, PIDs, prompts, comments, completion metadata, or filesystem paths.

Instance identifiers are 128-bit random lowercase hex strings generated with `secrets.token_hex(16)`. They are opaque identities, not secrets and not hashes of mutable names or paths.

## Current schema facts being preserved

At the pinned `origin/main` revision:

- `tasks.session_id` is documented as the originating chat/agent session (`hermes_cli/kanban_db.py:1205-1210`). It is not a worker transcript pointer.
- `task_runs.profile` already records the selected worker profile (`hermes_cli/kanban_db.py:1256-1260`).
- `sessions.profile_name` exists, but `sessions` has no board/task/run provenance (`hermes_state.py:1056-1104`).
- The primary writable `SessionDB` connection enables `PRAGMA foreign_keys=ON` before schema initialization (`hermes_state.py:1654-1671`).
- Both Kanban connection paths enable `PRAGMA foreign_keys=ON` (`hermes_cli/kanban_db.py:2100-2115`, `2126-2159`).

## Board identity

Each board database owns one immutable `board_instance_id` in a singleton `board_meta` row. A singleton table is preferred over a generic pragma or untyped key/value row because it gives the worker relation a real foreign-key target and makes a second identity structurally impossible.

```sql
CREATE TABLE IF NOT EXISTS board_meta (
    singleton         INTEGER NOT NULL PRIMARY KEY CHECK (singleton = 1),
    board_instance_id TEXT NOT NULL UNIQUE
        CHECK (
            length(board_instance_id) = 32
            AND board_instance_id NOT GLOB '*[^0-9a-f]*'
        )
) WITHOUT ROWID;
```

A backup restored as the same logical board preserves this ID. A copied DB
with the identity intact is still the same logical board. Creating a distinct
board from that copy requires an exclusive clone/rekey operation that first
revokes and removes the copied worker-link rows, then replaces the singleton
ID; merely changing the board slug or path does not create a new identity.

### First-upgrade algorithm

The board identity migration runs on every uncached initialization path after connection pragmas are set. Its correctness does not depend on the best-effort cross-process init flock.

1. Generate a candidate ID in application code.
2. Start `BEGIN IMMEDIATE` through an init-only transaction helper with the
   same busy/retry policy as Kanban writes. Do not route schema initialization
   through the public `write_txn` delegated-child mutation guard.
3. Create `board_meta` and the worker-link schema with `IF NOT EXISTS`.
4. Execute:

   ```sql
   INSERT INTO board_meta (singleton, board_instance_id)
   VALUES (1, :candidate)
   ON CONFLICT(singleton) DO NOTHING;
   ```

5. In the same transaction, read:

   ```sql
   SELECT board_instance_id
   FROM board_meta
   WHERE singleton = 1;
   ```

6. Validate the exact lowercase-hex format. Missing or malformed data aborts initialization; it is never silently replaced.
7. Commit and return the selected value.

`BEGIN IMMEDIATE` serializes concurrent first-open writers. If two processes generated different candidates, one insert wins and both read the same committed value. A crash after table creation but before insert leaves an empty singleton table; the next open safely inserts. Re-running the migration is a no-op.

Authorization-sensitive callers must read the ID from the live connection inside their current transaction. A module-level cached ID may be used for display only, never to register or attach a session.

## State-database identity

Each Hermes state-database incarnation owns one immutable
`state_db_instance_id`. An explicit rename/copy rekey closes that incarnation
and starts a new one; ordinary opens never mutate the ID. It is not
`profile_name` and is not an absolute `HERMES_HOME` path. The state DB also
records the profile name under which the identity was issued as a rename/copy
guard; the name is an attribute, not the identity.

```sql
CREATE TABLE IF NOT EXISTS state_db_meta (
    singleton              INTEGER NOT NULL PRIMARY KEY CHECK (singleton = 1),
    state_db_instance_id   TEXT NOT NULL UNIQUE
        CHECK (
            length(state_db_instance_id) = 32
            AND state_db_instance_id NOT GLOB '*[^0-9a-f]*'
        ),
    profile_name_at_issue  TEXT NOT NULL CHECK (profile_name_at_issue <> '')
) WITHOUT ROWID;
```

This belongs in `hermes_state.py`'s declarative schema. The state schema version advances from 23 to 24 because this adds durable authorization state, even though the always-run reconciliation remains the correctness backstop.

After `_init_schema()` has created the table, writable `SessionDB` initialization runs an unconditional `BEGIN IMMEDIATE` ensure step equivalent to the board algorithm: insert `(1, candidate, active_profile_name)` on singleton conflict do nothing, then select and validate the live row before commit. It never updates an existing valid ID.

A read-only `SessionDB` must not seed identity. If `state_db_meta` or its singleton row is absent, read-only resolution returns `legacy_unlinked`; registration requires a writable initialized `SessionDB`.

### Registration verification

Every API that registers a reciprocal row accepts an expected identity, never authority by assertion. Inside the same state-DB write transaction that creates the session and reciprocal row, it must:

1. Read `state_db_instance_id` and `profile_name_at_issue` from the live active state DB.
2. Resolve the active profile name from the profile-scoped runtime, not from caller-provided data.
3. Require `expected_state_db_instance_id == live_state_db_instance_id`.
4. Require `expected_profile_name == active_profile_name == profile_name_at_issue`.
5. Reject before writing on any mismatch.

The board registration/attach API applies the same rule to `board_instance_id`: it reads the live singleton inside its transaction and compares it with the expected tuple. Neither API may accept a cached identity from startup as sufficient proof.

## Board-side `worker_session_links`

The relation has exactly the requested seven columns. There is deliberately no surrogate key.

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_task_runs_id_task_id
    ON task_runs(id, task_id);

CREATE TABLE IF NOT EXISTS worker_session_links (
    board_instance_id    TEXT NOT NULL,
    task_id               TEXT NOT NULL,
    run_id                INTEGER NOT NULL,
    profile_name          TEXT NOT NULL CHECK (profile_name <> ''),
    state_db_instance_id  TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    state                 TEXT NOT NULL
        CHECK (state IN ('allocated', 'attached', 'retired', 'orphaned')),

    PRIMARY KEY (state_db_instance_id, session_id),

    FOREIGN KEY (board_instance_id)
        REFERENCES board_meta(board_instance_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (run_id, task_id)
        REFERENCES task_runs(id, task_id)
        ON DELETE CASCADE
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_worker_session_links_run
    ON worker_session_links(board_instance_id, task_id, run_id);
```

The composite primary key enforces the required uniqueness within this board
and makes both key columns non-null under SQLite's `WITHOUT ROWID` semantics.
Because Hermes can host several independent board databases, the reciprocal
state table's `session_id` primary key is the final cross-board/fleet boundary:
a second board may reserve an `allocated` row, but it cannot create a second
local reciprocal attachment for the same physical session. Consequently two
boards can never both reach authoritative exact reciprocal `attached` state.

`run_id` is intentionally not unique. Multiple rows with the same board/task/run and different session IDs are required for compression and other explicit continuations.

The composite task-run foreign key prevents a row from pairing an existing run with the wrong task. Deleting a task/run cascades its board-side authority rows; this is authorization revocation, not evidence that the corresponding state sessions may be deleted. Any surviving state-side rows become reciprocal mismatches and therefore remain ineligible.

## Reciprocal state-DB `worker_session_links`

The state database uses the same relation name and exactly the same seven columns. `session_id` is the local primary key because one physical state DB can contain a session ID only once.

```sql
CREATE TABLE IF NOT EXISTS worker_session_links (
    board_instance_id    TEXT NOT NULL,
    task_id               TEXT NOT NULL,
    run_id                INTEGER NOT NULL,
    profile_name          TEXT NOT NULL CHECK (profile_name <> ''),
    state_db_instance_id  TEXT NOT NULL,
    session_id            TEXT NOT NULL PRIMARY KEY
        REFERENCES sessions(id) ON DELETE CASCADE,
    state                 TEXT NOT NULL
        CHECK (state IN ('allocated', 'attached', 'retired', 'orphaned')),

    FOREIGN KEY (state_db_instance_id)
        REFERENCES state_db_meta(state_db_instance_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) WITHOUT ROWID;
```

No state-side foreign key can target board tables because they live in another SQLite database. Exact reciprocal comparison is therefore an application-level invariant evaluated from live, locking connections to both databases.

## State meanings and transitions

- `allocated`: the board has reserved this session ID for the current run, but the exact local session/reciprocal commit has not yet been acknowledged by the board. Never eligible.
- `attached`: the session and local reciprocal row committed, and the board compare-and-swap acknowledged the same tuple while the run claim was current. This is the only authoritative state.
- `retired`: authority was explicitly revoked, for example because a retry superseded a transient session or maintenance deliberately retired the link. Never eligible.
- `orphaned`: allocation/attachment was abandoned or an explicit repair quarantined a partial/mismatched pair. Never eligible.

Allowed board transitions are monotonic:

```text
allocated -> attached
allocated -> orphaned
attached  -> retired
attached  -> orphaned
```

Terminal states do not transition back. Repair that wants a valid attachment creates a new allocation; it does not infer or resurrect authority from cwd, parentage, or timestamps.

Normal run closure must not retire every attached row: doing so would erase all provenance before retention-based maintenance could use it. Run closure instead marks any still-`allocated` rows `orphaned`; already-`attached` rows remain attached. A specific superseded retry session may be retired explicitly. The terminal `task_runs` row remains the source for run outcome and `ended_at`.

## Two-phase cross-DB attach

SQLite cannot atomically commit across the board and an independently opened profile state DB. The protocol is intentionally asymmetric and fail-closed:

1. **Board allocation transaction**
   - Re-read the live board ID.
   - Verify task status, `current_run_id`, claim lock/expiry, and selected profile.
   - Re-read the selected profile's live state DB ID.
   - Generate a fresh session ID.
   - Insert the board row as `allocated`.

2. **State registration transaction**
   - Re-read and verify the live state DB identity/profile binding.
   - Insert the `sessions` row and reciprocal row in one transaction.
   - The local reciprocal row is inserted as `attached`: locally, the session now exists and has committed provenance. If either insert fails, both roll back.

3. **Board attach transaction**
   - Re-read the live board ID and current task/run/claim/profile state.
   - Compare-and-swap the exact row from `allocated` to `attached`.
   - A zero-row update is a hard failure, never a fallback to an ordinary session.

A crash after step 1 leaves board `allocated` with no local row. A crash after step 2 leaves board `allocated` and local `attached`. Both shapes are ineligible because exact reciprocal `attached` rows do not exist. Automatic startup repair is forbidden.

Every continuation repeats all three steps with a fresh session ID. `parent_session_id` may describe transcript topology but never supplies provenance.

## Read-only resolver contract

The public resolver returns an authoritative link only after independently reading the live singleton IDs and exact rows from both locking read-only SQLite connections. It must not use `immutable=1` for live DBs.

The resolver compares all seven columns, verifies the state-side `sessions` row exists, and verifies the board task/run join. Suggested non-authoritative classifications are:

- `legacy_unlinked`
- `board_allocated`
- `board_terminal_state`
- `state_row_missing`
- `session_row_missing`
- `tuple_mismatch`
- `board_identity_mismatch`
- `state_identity_mismatch`
- `task_run_mismatch`
- `attached`

Only `attached` is eligible for later policy checks. Session age, `ended_at`, task/run terminality, maintenance locks, snapshots, and deletion are outside this schema contract and remain additional mandatory gates.

## Profile rename, copy, and restore

A profile rename or independent profile-home copy must not inherit the source state DB identity.

- Current managed `profile create --clone-all` already excludes `state.db`, `state.db-wal`, and `state.db-shm`, so its fresh target naturally receives a new ID (`hermes_cli/profiles.py:104-126`).
- Current `rename_profile` moves the directory without touching `state.db` (`hermes_cli/profiles.py:2150-2202`). The implementation must add an explicit identity-rekey step after quiescing the old profile and before the renamed profile may dispatch workers.
- Any import/copy intended to create an independent profile must run the same rekey step before worker registration. A restore explicitly declared to restore the same logical profile/database may preserve the ID.

Rekey runs under the profile-exclusive maintenance lock in one state-DB transaction:

1. Require no active worker registration/attach transaction.
2. Mark existing local worker links `orphaned`.
3. Update the singleton to a new random ID and new `profile_name_at_issue`; `ON UPDATE CASCADE` carries the new ID into the now-orphaned local rows.
4. Do not rewrite board rows. Their old identity no longer matches, so they fail closed.

A byte-for-byte raw copy is information-theoretically indistinguishable from restoration of the same logical DB using DB-local contents alone. Such a copy is unsupported as an independent profile until the explicit rekey operation runs. Registration must reject a profile-name binding mismatch and an expected/live state ID mismatch; it must never silently rekey on worker startup.

## Foreign-key pragma and cascade proof

SQLite foreign keys are per connection and default off. The DDL alone is insufficient.

Every writable board/state connection must execute `PRAGMA foreign_keys=ON` before any transaction and verify `PRAGMA foreign_keys` returns `1`. Setting it inside an already-open transaction is ineffective. Migration verification must run `PRAGMA foreign_key_check` and fail on any row.

Current primary connections satisfy this:

- Kanban `connect()` enables foreign keys on both fast and initialization paths.
- Writable `SessionDB` enables foreign keys before `_init_schema()`.
- Read-only `SessionDB` returns before setting the pragma, but it cannot delete and does not need cascades.

A live probe against the current `SessionDB` connection created a session plus a temporary `session_id REFERENCES sessions(id) ON DELETE CASCADE` row, then deleted via `SessionDB.delete_session()`. The observed result was:

```text
foreign_keys=1, link rows before=1, delete returned true, link rows after=0
```

If a future raw SQLite writer omits the pragma, deleting a session can leave a stale reciprocal row. The resolver must still join to `sessions`, so the stale row is ineligible rather than authoritative; however, the orphan is a schema-integrity failure and must be surfaced by `foreign_key_check`, not ignored.

## Explicit non-repurposing

`tasks.session_id` remains the originating chat/agent session that created the card. It is never written with a worker transcript ID and is never consulted for worker-session ownership.

`worker_session_id` in task-run completion metadata remains diagnostic only. It may help an operator investigate a run, but contradictory or missing metadata has no effect on authoritative resolution.

## Migration and rollout checklist

1. Fetch and pin current `origin/main`; do not implement against an installed stale checkout.
2. Add board singleton/relation DDL and the transactional always-run identity ensure.
3. Add state singleton/relation DDL, bump `SCHEMA_VERSION` to 24, and add the transactional always-run identity ensure.
4. Do not backfill worker links for existing sessions or runs.
5. Add live-read APIs for both IDs; do not expose an authorization cache.
6. Add transactional state registration and board allocation/attach CAS APIs.
7. Add the explicit rekey API and wire managed rename/copy/independent-restore paths before enabling worker registration there.
8. Restart long-lived gateways/dispatchers after rollout. Old processes cannot create authoritative rows; sessions they create remain safely unlinked.
9. Run schema checks on fresh and upgraded DBs, including `PRAGMA foreign_key_check`.
10. Only after this contract is merged may maintenance locking and transcript GC consume it.

## Required failure-mode tests

- Eight or more concurrent first opens converge on one board ID and one state DB ID.
- Interrupted migration with table present/row absent recovers idempotently.
- Invalid or duplicate singleton data fails initialization without regeneration.
- Two session IDs under one run succeed.
- A second run claiming the same `(state_db_instance_id, session_id)` fails without modifying the first row.
- A local session delete cascades the reciprocal state row with foreign keys enabled.
- The missing-pragma fixture demonstrates the stale row and resolver ineligibility, then `foreign_key_check` detects it.
- Board `allocated` without local row is ineligible.
- Board `allocated` plus local `attached` is ineligible.
- Exact reciprocal `attached` tuples resolve; a one-column mismatch does not.
- A same-cwd human CLI session with no rows is unlinked.
- Profile rename/copy rekeys before registration; replaying the old expected identity is rejected.
- Compression creates a second independently attached row for the same run.
- `tasks.session_id` and completion `worker_session_id` are ignored even when they contradict the reciprocal relation.

A local DDL probe of the schema in this note observed one identity across eight
concurrent board opens and, separately, eight concurrent state-DB opens. It
also allowed two sessions for one run, rejected a duplicate state/session key,
and cascaded the state reciprocal row on session deletion.