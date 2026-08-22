# Temporal Patterns Adoption — Architecture Reference

## Purpose

5 portable lifecycle patterns extracted from Temporal.io's design, adapted for single-process governed systems (no Temporal Server, no distributed infra). Proven in LAH Stack (259 tests, 15/15 acceptance scenarios).

## The 5 Patterns

### 1. Event-sourced Program History

**Core idea:** All state transitions are append-only events. Current state is rebuilt by deterministic replay.

**Schema:**
```
event_id (uuid), program_id, run_id, mission_id?, event_type,
previous_state, resulting_state, authority, timestamp,
payload (object), causation_id?, correlation_id, schema_version (1),
integrity_fingerprint (sha256 chain to previous event)
```

**Key exports:**
- `EventHistory` — JSONL-backed append-only store, duplicate detection, fingerprint chaining
- `ReplayEngine` — deterministic replay, fingerprint verification, checkpoint-relative partial replay
- `LegacyLedgerAdapter` — read-only translation from legacy formats to events

**Fail-closed paths:** corrupted JSONL, duplicate event IDs, fingerprint mismatch, invalid event types, out-of-order timestamps

### 2. Hierarchical Runtime State Machine (HSM)

**Core idea:** All program and mission transitions are defined in a single canonical registry with guards, side effects, and authority.

**Program states (6):** INITIALIZED, RUNNING, BLOCKED, OPERATOR_DECISION_REQUIRED, FAILED, COMPLETED
**Mission states (13):** PENDING, READY, ENQUEUED, CLAIMED, EXECUTING, VERIFYING, PROMOTING, PROMOTED, RETRYING, REPAIRING, BLOCKED, FAILED, CANCELLED

**Key exports:**
- `transition-registry.mjs` — `PROGRAM_TRANSITIONS[]`, `MISSION_TRANSITIONS[]`, `getProgramTransition()`, `getMissionTransition()`, `isValidProgramState()` — all frozen
- `StateMachine` — `transitionProgram()`, `transitionMission()`, `checkGuards()`, `executeSideEffects()`, `canTransition()`, `getAvailableEvents()`

**Each transition has:** source_state, event, destination_state, guards[], side_effects[], authority, allowed_retry, description

**Fail-closed paths:** unsupported transition (throws `InvalidTransitionError`), missing guard (throws `GuardCheckError`), terminal state reached

### 3. Program Ownership Lease

**Core idea:** Only one runtime may mutate a program at a time. Optimistic locking via lease_version prevents concurrent mutation.

**Lease schema (per-program file):**
```
program_id, runtime_owner_id, lease_version, acquired_at, renewed_at, expires_at, status (ACTIVE|EXPIRED|RELEASED|LOST)
```

**Key exports:**
- `LeaseManager` — `acquire()`, `release()`, `renew()`, `getLease()`, `isOwner()`, `isLeaseExpired()`, `forceRelease()`, `listActiveLeases()`

**Mechanism:** Filesystem atomic write (temp file → rename). Default lease duration: 60s. Default renew interval: 30s.

**Fail-closed paths:** `LEASE_OWNERSHIP_LOST` (different owner holds active lease), `STALE_LEASE_VERSION` (optimistic lock collision), expiration, concurrent resume without reacquire

**Read-only safe:** getLease, isOwner, isLeaseExpired, listActiveLeases — no ownership required.

### 4. Typed Task Categories

**Core idea:** All tasks belong to exactly one immutable category with defined behavior, retry policy, authority, and priority.

**9 categories (IDs 1-9, frozen, never renumbered):**

| ID | Name | Behavior | Default Timeout | Authority | Mutation |
|----|------|----------|----------------|-----------|----------|
| 1 | PLANNING | immediate | 30s | strategic_planner | READ_ONLY |
| 2 | EXECUTION | immediate | 60s | real_executor | READ_WRITE |
| 3 | VERIFICATION | immediate | 30s | verification_gate | READ_ONLY |
| 4 | PROMOTION | immediate | 30s | promotion_controller | READ_WRITE |
| 5 | RETRY | immediate | 60s | runtime | READ_WRITE |
| 6 | TIMER | scheduled | 120s | runtime | READ_WRITE |
| 7 | OPERATOR_SIGNAL | scheduled | 300s | operator | READ_WRITE |
| 8 | CLEANUP | immediate | 30s | runtime | READ_WRITE |
| 9 | ROLLBACK | immediate | 60s | runtime | READ_WRITE |

**Key exports:**
- `CATEGORY_IDS` — `{ PLANNING:1, EXECUTION:2, ... }`
- `CATEGORIES` — frozen array of category objects with retry_policy (frozen nested), allowed_authority, priority_class, mutation_permission
- `getCategory()`, `getCategoryByName()`, `isValidCategoryId()`, `getImmediateCategories()`, `getScheduledCategories()`

**Fail-closed paths:** invalid category ID/name (returns null), mutation of frozen objects (TypeError), persisted ID renumbering

### 5. Pluggable Persistence Factory

**Core idea:** All persistence goes through store interfaces. Filesystem is the default; SQLite or other backends can be added without changing consumers.

**Store interfaces (6):**
- `ProgramEventStore` — `append()`, `getAll()`, `getByProgram()`, `getByRun()`, `getById()`, `getByCorrelation()`, `getLast()`, `getRange()`, `count()`
- `ProgramLedgerStore` — `appendEntry()`, `getAllEntries()`, `getEntriesByProgram()`, `getEntry()`, `count()`
- `CheckpointStore` — `writeCheckpoint()`, `readCheckpoint()`, `listCheckpoints()`, `deleteCheckpoint()`
- `QueueStore` — `enqueue()`, `dequeue()`, `peek()`, `getAll()`, `count()`, `clear()`
- `ReceiptStore` — `writeReceipt()`, `readReceipt()`, `listReceipts()`, `deleteReceipt()`
- `LeaseStore` — `acquire()`, `release()`, `renew()`, `getLease()`, `isOwner()`, `isLeaseExpired()`, `forceRelease()`, `listActiveLeases()`

**Key exports:**
- `PersistenceFactory` — `getEventStore()`, `getLedgerStore()`, `getCheckpointStore()`, `getQueueStore()`, `getReceiptStore()`, `getLeaseStore()`, `getAllStores()`, `setBasePath()`
- Stores auto-create subdirectories on first access

**Runtime integration wiring:** `RuntimeIntegration` class composes all 5 patterns: acquires lease before mutations, validates HSM transitions before appending events, creates tasks with typed categories, supports replay from checkpoints.

## When to adopt these patterns

Add these patterns when your system needs:
- Crash-safe state recovery (event sourcing)
- Formalized lifecycle management (HSM)
- Concurrent runtime protection (lease)
- Structured task routing (categories)
- Storage backend flexibility (factory)

**Adopt incrementally** — each pattern is independently useful. The acceptance trial checklist validates all integrated paths.

## Reference implementation

LAH Stack implementation at `tools/temporal-patterns/` — 19 source modules (including runtime-integration bridge + persistence context), 8 test files, 259 unit tests + 15 acceptance trials. 

**LAH_TEMPORAL_PATTERNS_ADOPTION_ROADMAP_V1** (CERTIFIED 2026-07-15) — 5 patterns adopted, all isolated tests and acceptance trials pass.

**LAH_TEMPORAL_RUNTIME_INTEGRATION_V1** (CERTIFIED 2026-07-15) — patterns wired into real Progression Runtime:
- `RuntimeTemporalBridge` at `tools/temporal-patterns/runtime-temporal-bridge.mjs` — wraps all 5 patterns
- `PersistenceContext` at `tools/temporal-patterns/persistence/persistence-context.mjs` — canonical single-path authority for all stores
- Event schema expanded from 13 to 30 types including LEASE_ACQUIRED, MISSION_READY, MISSION_ENQUEUE_STARTED, VERIFICATION_STARTED, PROMOTION_DECIDED, CHECKPOINT_WRITTEN
- Real runtime imports bridge when persistence_context present in contract; fails closed on init/lease/path failures
- Call-site matrix at `tools/temporal-patterns/CALL-SITE-MATRIX.md`

### Fail-closed enforcement

When the contract carries a `persistence_context`, ALL temporal-pattern integration is mandatory:
- Bridge import failure → `TEMPORAL_RUNTIME_BRIDGE_UNAVAILABLE` (fatal, no fallback)
- Bridge init failure → `TEMPORAL_RUNTIME_BRIDGE_INITIALIZATION_FAILED` (fatal)
- Lease acquisition failure → `PROGRAM_OWNERSHIP_LOST` (fatal, no mutation without ownership)
- Path divergence → `PERSISTENCE_CONTEXT_MISMATCH` (fatal, no silent override)

See `governed-pilot-execution` skill → "Governed Integration Pattern: Fail-Closed Enforcement".
