# API Run Reconnect Design

Date: 2026-05-07

## Goal

Support resumable API-driven conversations for web clients that use Hermes'
structured run API. If a browser, reverse proxy, or SSE client disconnects while
an agent run is still in progress, the agent should continue running in the
background. The client can reconnect with the same `run_id` and resume receiving
missed lifecycle events from the last event it saw.

This design applies to `POST /v1/runs` and
`GET /v1/runs/{run_id}/events`. It intentionally does not change the existing
OpenAI-compatible streaming behavior for `/v1/chat/completions` or
`/v1/responses`, where a client disconnect currently interrupts the associated
agent task.

## Current Architecture

Hermes uses a synchronous core agent wrapped by async platform adapters:

- `run_agent.py` owns `AIAgent.run_conversation()`, the synchronous model/tool
  loop.
- `gateway/platforms/api_server.py` owns the OpenAI-compatible HTTP API and
  creates `AIAgent` instances from gateway runtime configuration.
- `hermes_state.py` owns durable conversation storage in SQLite.
- `ResponseStore` in `gateway/platforms/api_server.py` owns Responses API
  response chaining via `previous_response_id`.

The existing `/v1/runs` API already separates run submission from observation:

- `POST /v1/runs` returns a `run_id` immediately.
- `GET /v1/runs/{run_id}` returns pollable status.
- `GET /v1/runs/{run_id}/events` streams structured lifecycle events.
- `POST /v1/runs/{run_id}/stop` explicitly interrupts a run.

The missing piece is that the current event stream is backed by a single
in-memory queue. When the SSE connection ends, the queue entry is removed, so a
new connection cannot replay missed events or continue observing the same run.

## Selected Approach

Implement a resumable run event log in the API server:

- Keep the agent run independent from the SSE subscriber connection.
- Assign a monotonically increasing integer sequence to every run event.
- Store events in a lightweight SQLite-backed run event store.
- Fan out each committed live event to all connected subscribers for that
  `run_id`.
- On reconnect, attach and replay under a per-run sequencing discipline so
  there is no gap between stored replay and live fanout.
- Treat durable run metadata as the source of truth for public run status, with
  any in-memory status dictionary acting only as a cache.
- Retain terminal run status and event history for a bounded TTL.

This keeps the change local to the API server platform adapter and follows the
project's existing style: platform adapters orchestrate async transport concerns,
while `AIAgent` remains the synchronous execution engine.

### Architecture Fit

The approach fits the current Hermes architecture because it does not move
conversation execution out of `AIAgent` and does not change the message loop in
`run_agent.py`. The API server continues to create the agent and pass
`stream_delta_callback` and `tool_progress_callback`; the only change is that
those callbacks enqueue structured events into a resumable API-server-owned
broker instead of a single SSE queue.

This mirrors the existing `ResponseStore` pattern in `api_server.py`: small
SQLite persistence is acceptable at the platform adapter boundary when it
supports HTTP/API semantics such as response chaining or resumable observation.
It should not be pushed into `hermes_state.py`, which remains conversation
storage, nor into `AIAgent`, which should stay transport-agnostic.

## API Contract

`POST /v1/runs` returns the existing response plus enough metadata for clients to
subscribe:

```json
{
  "run_id": "run_...",
  "status": "started",
  "session_id": "run_...",
  "events_url": "/v1/runs/run_.../events"
}
```

`GET /v1/runs/{run_id}/events` accepts either resume mechanism:

- `Last-Event-ID: 42`
- `GET /v1/runs/{run_id}/events?after=42`

`after` and `Last-Event-ID` are exclusive sequence cursors: `after=42` means
"send events with `sequence > 42`". If both are supplied, the explicit `after`
query parameter takes precedence.

The SSE stream emits event IDs:

```text
id: 43
event: message.delta
data: {"event":"message.delta","run_id":"run_...","sequence":43,"delta":"..."}
```

Terminal events such as `run.completed`, `run.failed`, and `run.cancelled` are
also stored and replayable. If a client reconnects after a run has completed, it
receives any missing terminal event and the stream closes cleanly. If a terminal
run is requested with `after >= last_sequence`, there is nothing left to replay,
so the stream closes cleanly instead of staying open for events that can no
longer arrive.

## Components

### RunEventStore

Add a small SQLite-backed store in `gateway/platforms/api_server.py` or a nearby
module if the adapter becomes too large.

Responsibilities:

- Create tables for run events and run metadata.
- Initialize run metadata before the background task is created.
- Allocate the next sequence number per `run_id`.
- Append events transactionally and return the committed sequence.
- List events after a sequence.
- Store the public run status payload used by `GET /v1/runs/{run_id}`.
- Mark terminal runs and their terminal timestamp.
- On startup, mark any non-terminal persisted runs as failed/abandoned because
  the Python thread that was executing them cannot survive process restart.
- Prune old terminal runs and events after TTL.

Suggested tables:

```sql
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS run_meta (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    terminal_at REAL,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    status_data TEXT NOT NULL DEFAULT '{}'
);
```

`status_data` stores only JSON-serializable public status fields such as
`session_id`, `model`, `last_event`, `output`, `error`, and `usage`. It must not
contain live `AIAgent` objects, tasks, queues, or other process-local handles.

### Live Subscribers

Replace the single `_run_streams[run_id] = queue` model with per-run subscriber
sets:

- `_run_subscribers: Dict[str, set[asyncio.Queue]]`
- `_run_event_locks: Dict[str, asyncio.Lock]`
- `_run_statuses` may remain as a hot cache, but `RunEventStore` is the source
  of truth for `GET /v1/runs/{run_id}`.
- `_active_run_agents` and `_active_run_tasks` remain for explicit stop support.

When an event is produced:

1. Marshal the event onto the API server's event loop. Executor-thread
   callbacks must not write the store or subscriber queues directly.
2. Under the run's event lock, append it to `RunEventStore` and receive the
   committed sequence.
3. Update durable run metadata and the optional `_run_statuses` cache with
   `last_sequence` and `last_event`.
4. Put the committed sequenced event into every active subscriber queue.

All event types for one run, including terminal events, must pass through the
same append path. This preserves sequence ordering across text deltas, tool
events, reasoning events, and terminal events, even though the callbacks can be
triggered from a worker thread while terminal handling runs in the async task.

Terminal events are final. Once a terminal event has been committed, later
non-terminal callbacks for the same `run_id` are ignored and logged at debug
level rather than appended after completion.

When an SSE subscriber connects:

1. Parse the exclusive resume cursor from `after` or `Last-Event-ID`.
2. Validate that the run exists in `RunEventStore`.
3. Create a subscriber queue.
4. Under the run's event lock, add the queue to `_run_subscribers[run_id]`,
   capture current metadata, and list stored events with `sequence > after`.
5. Write the replayed events to the response.
6. Continue reading from the live queue, dropping any queued event whose
   sequence is less than or equal to the highest replayed sequence.

Registering the queue before replay, while using sequence-based de-duplication,
closes the race where a new event could otherwise be committed after the replay
query but before live subscription.

When an SSE subscriber disconnects:

1. Remove only that subscriber queue.
2. Do not interrupt the agent.
3. Do not cancel the run task.
4. Do not delete the event log.

### Run Lifecycle

`POST /v1/runs` should initialize run metadata before creating the background
task. The background task should emit:

- `run.started` once execution begins.
- `message.delta` for streamed assistant text.
- `tool.started`, `tool.completed`, and `reasoning.available` from the existing
  callbacks.
- exactly one terminal event: `run.completed`, `run.failed`, or `run.cancelled`.

The background task should not depend on a connected SSE response. It continues
until completion, failure, or an explicit stop request even if there are zero
active subscribers.

The existing `POST /v1/runs/{run_id}/stop` remains the explicit cancellation
path. It should still call `agent.interrupt("Stop requested via API")` and
cancel the task as it does today.

On adapter startup, `RunEventStore` should reconcile persisted non-terminal
runs from a previous process. Because in-flight Python execution cannot be
resurrected, each persisted `queued`, `running`, or `stopping` run should be
marked failed with a replayable terminal event such as:

```json
{
  "event": "run.failed",
  "run_id": "run_...",
  "error": "Run abandoned because the API server process restarted"
}
```

## Error Handling

- Unknown `run_id`: return `404 run_not_found`.
- Invalid `after` value: return `400 invalid_request_error`.
- `after` greater than current last sequence for a non-terminal run: attach
  live and keep the connection open until new events or terminal closure.
- `after` greater than or equal to the current last sequence for a terminal run:
  close the stream cleanly because no more events can arrive.
- Store write failure: log the error and fail the run with `run.failed` if the
  event log cannot preserve resumability.
- Subscriber send failure: remove that subscriber only.

## Security

The existing API key behavior remains unchanged. All run status, event, and stop
routes continue to call `_check_auth()`.

Events may contain tool previews and assistant deltas, so resumable event access
must remain protected by the same bearer token configuration as the existing run
endpoints.

## Testing Plan

Add or update tests under `tests/gateway/test_api_server_runs.py`:

- Client disconnect from `/events` does not call `agent.interrupt()` and does
  not cancel the run task.
- Reconnect with `Last-Event-ID` replays only events with a higher sequence.
- Reconnect with `?after=` behaves the same as `Last-Event-ID`.
- If both `Last-Event-ID` and `?after=` are provided, `?after=` wins.
- An event produced during reconnect replay is not lost and is not delivered
  twice.
- Completed runs can still replay their terminal event.
- Completed runs requested with `after >= last_sequence` close cleanly.
- Multiple subscribers can observe the same run without stealing events from
  one another.
- Events keep strictly increasing per-run sequence numbers across text, tool,
  reasoning, and terminal events.
- Late non-terminal callbacks after a terminal event are ignored.
- `POST /v1/runs/{run_id}/stop` still interrupts the active agent and emits a
  terminal cancellation/failure event.
- TTL cleanup removes old terminal run events but does not remove running runs.
- Adapter startup marks persisted non-terminal runs abandoned with a replayable
  terminal failure event.
- Auth requirements are preserved for run status, events, and stop routes.

## Documentation Needed For Upstream

Update project-facing documentation once the implementation lands:

- API server capability response should advertise resumable run events.
- Add a short user-facing example for:
  1. `POST /v1/runs`
  2. connect to `/v1/runs/{run_id}/events`
  3. reconnect with `Last-Event-ID`
  4. poll `/v1/runs/{run_id}` as fallback
- Mention that resumability applies to `/v1/runs`, not the OpenAI-compatible
  `/v1/chat/completions` streaming endpoint.

## Implementation Status

Implemented in `gateway/platforms/api_server.py`:

- `RunEventStore` for SQLite-backed run events and public run metadata.
- `/v1/runs` event delivery refactored from a single queue to replayable
  multi-subscriber streams.
- Atomic reconnect flow that registers a subscriber before replay and drops
  duplicate queued events by sequence.
- Startup reconciliation for persisted non-terminal runs.
- `/v1/capabilities` advertises resumable run events.
- Tests covering replay, completion replay, multiple subscribers, reconnect
  after disconnect, stop behavior, auth, and startup reconciliation.

Still useful before an upstream PR:

- Add user-facing docs or a release note explaining the resumable run stream
  behavior.
- Manually verify with an external SSE client that disconnects and reconnects
  using `Last-Event-ID`.

## Non-Goals

- Do not change `/v1/chat/completions` disconnect semantics.
- Do not change `/v1/responses` disconnect semantics in this PR.
- Do not attempt process-restart recovery of in-flight agent execution. Stored
  events and terminal status can survive process restart, but a Python thread
  that was running an agent cannot be resurrected after the process exits. On
  restart, persisted non-terminal runs are marked failed/abandoned rather than
  resumed.
- Do not rebuild the dashboard chat transcript in React. The dashboard chat tab
  remains PTY/TUI-backed.
