# ADR 0001: Keep Kanban state row-oriented with an append-only event log

- Status: Accepted
- Date: 2026-05-31
- Context: Hermes Kanban v1

## Context

Hermes Kanban coordinates work across independent Hermes profiles by using a
shared SQLite board. The current implementation stores the operational view of a
card in row-oriented tables:

- `tasks` is the materialized task record used by dispatchers, CLIs, dashboard
  reads, and worker-context construction.
- `task_runs` is the materialized attempt record for every dispatch/worker run.
- `task_comments`, `task_links`, `task_attachments`, and
  `kanban_notify_subs` hold their own queryable projections.
- `task_events` is an append-only journal of important state transitions and
  side effects, including task creation, assignment, claims, completions,
  blocks, failures, and notifier-visible lifecycle events.

The board is written by many short-lived processes: gateway handlers, CLI
commands, dispatchers, and workers. Correctness depends on simple SQLite
transactions, WAL, `BEGIN IMMEDIATE`, and compare-and-swap updates over the
current `tasks` row. The dispatcher needs cheap queries such as "which ready task
can this profile claim now?" and "is this running task stale?" without replaying
an unbounded history stream on every tick.

As Kanban grows more event-heavy, we need to record whether `task_events` is the
source of truth or an audit/notification log alongside the materialized state.

## Decision

Kanban v1 will **not** make `task_events` the canonical event-sourced store.

The canonical state remains the normalized row set (`tasks`, `task_runs`, and
related tables). `task_events` remains an append-only audit, debugging, and
notification stream written in the same transaction as the state mutation that it
records.

In other words:

1. Reads that drive behavior MUST read the materialized tables, not reconstruct
   current state by replaying `task_events`.
2. Mutations MUST update the materialized table(s) and append the corresponding
   event in one SQLite write transaction when an event is part of the public
   lifecycle.
3. Event consumers MUST treat events as facts about what the kernel attempted or
   completed, not as the sole state representation.
4. Historical analysis, gateway notifications, dashboards, and worker retry
   context MAY use `task_events`, but they must tolerate older rows with missing
   payload fields and newer rows with additional fields.

## Rationale

### Dispatcher correctness is simpler with row-level CAS

The critical concurrency primitive is a single atomic claim:

- inspect a task's current status, dependencies, claim lock, and timestamps;
- update `tasks.status`, `claim_lock`, `claim_expires`, `worker_pid`, and
  `current_run_id` only if the row is still claimable;
- create the corresponding `task_runs` row and append the event in the same
  transaction.

This maps directly to SQLite's writer serialization and compare-and-swap update
semantics. A fully event-sourced board would have to rebuild current task state
from a stream before each claim or maintain projections that are themselves the
real concurrency boundary. At that point the projections are the state table in
all but name, with extra replay failure modes.

### The board is operational infrastructure, not an analytics ledger

Kanban's highest-value invariant is that profiles can keep working after process
crashes, gateway restarts, timeouts, and worker retries. Operational code needs a
bounded, inspectable representation of current truth. The existing rows make
manual recovery and debugging straightforward with ordinary SQLite queries.

The event log is still valuable, but for different jobs:

- explaining how a card reached its current state;
- grouping lifecycle events by `run_id`;
- notifying subscribed gateway threads without scanning full task rows;
- preserving failure history even after counters are reset;
- giving retried workers enough context to avoid repeating failed paths.

Those jobs do not require events to be the only source of truth.

### Backward compatibility matters

Existing boards may lack newer event payload fields or `run_id` on historical
rows. They may also include rows from earlier event-kind names that were migrated
in place. Keeping the materialized tables canonical lets additive migrations
remain cheap and forgiving: new columns can be added and backfilled where
possible without requiring a full event replay to recover board state.

## Consequences

### Positive

- Dispatcher hot paths stay simple and fast.
- Existing boards remain readable without replay migrations.
- Crash recovery can inspect and repair a bounded set of current-state columns.
- The dashboard and CLI can query current state directly while still exposing the
  event timeline for audit.
- Gateway notification watchers can tail `task_events` without owning business
  logic for state reconstruction.

### Negative

- There is duplicated information: some lifecycle facts appear both in a
  materialized row and as an event payload.
- Writers must preserve the invariant that state changes and their events are
  committed together.
- Bugs can create drift between a state row and its event history; tests should
  cover high-value transitions to catch that.
- Arbitrary temporal queries may need to combine event history with current rows
  rather than replaying a perfect append-only log.

## Alternatives considered

### Full event sourcing

All state changes would be represented only as events. Current `tasks` and
`task_runs` views would be rebuilt by replaying the stream or by maintaining
projections.

Rejected because it moves concurrency correctness from simple row-level updates
to projection freshness and replay ordering. It also makes dispatcher ticks and
manual recovery more complex while providing little benefit for the current
single-host SQLite deployment.

### Event log only for human-readable audit text

The board could store events as opaque strings and keep all structured data in
state rows.

Rejected because structured event payloads are useful for notifiers, dashboards,
worker retry context, and future analysis. The issue is not structured events;
the issue is making them the sole canonical state.

### Split operational state and immutable external ledger

Kanban could keep SQLite state as-is and additionally ship every event to an
external log system.

Deferred. This may be useful for hosted/multi-node deployments, but it is outside
Kanban v1's local-first scope. If added later, the external stream should be fed
from the same transactional event append path or an outbox table, not from ad hoc
post-commit hooks.

## Implementation guidelines

When adding or changing Kanban lifecycle behavior:

1. Treat the materialized tables as the canonical read model.
2. Put state mutation, run mutation, and `_append_event(...)` calls inside the
   same `write_txn(conn)` block for lifecycle transitions.
3. Include only stable, consumer-meaningful fields in event payloads. Do not
   require consumers to parse prose fields such as block reasons when a
   structured field is available.
4. Keep event payloads additive. New consumers must tolerate missing historical
   keys; old consumers must tolerate extra keys.
5. Prefer adding regression tests around transition helpers (`create_task`,
   `claim_task`, completion/block/failure paths, dependency promotion, and
   notifier-visible events) instead of testing raw SQL event counts alone.
6. If a future feature needs replayable history, first define the projection and
   drift-checking strategy in a new ADR before changing the source-of-truth
   contract.

## Related code and documents

- `hermes_cli/kanban_db.py` — SQLite schema, transaction helpers, task mutation
  helpers, run lifecycle helpers, and `_append_event(...)`.
- `tools/kanban_tools.py` — agent-facing Kanban tools.
- `docs/hermes-kanban-v1-spec.pdf` — original Kanban v1 design specification.
