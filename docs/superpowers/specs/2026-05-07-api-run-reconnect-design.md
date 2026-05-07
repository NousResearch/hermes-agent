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
- Fan out each live event to all connected subscribers for that `run_id`.
- On reconnect, replay stored events after `Last-Event-ID` or `?after=<seq>`,
  then attach the subscriber to the live fanout.
- Retain terminal run status and event history for a bounded TTL.

This keeps the change local to the API server platform adapter and follows the
project's existing style: platform adapters orchestrate async transport concerns,
while `AIAgent` remains the synchronous execution engine.

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

The SSE stream emits event IDs:

```text
id: 43
event: message.delta
data: {"event":"message.delta","run_id":"run_...","sequence":43,"delta":"..."}
```

Terminal events such as `run.completed`, `run.failed`, and `run.cancelled` are
also stored and replayable. If a client reconnects after a run has completed, it
receives any missing terminal event and the stream closes cleanly.

## Components

### RunEventStore

Add a small SQLite-backed store in `gateway/platforms/api_server.py` or a nearby
module if the adapter becomes too large.

Responsibilities:

- Create tables for run events and run metadata.
- Allocate the next sequence number per `run_id`.
- Append events transactionally.
- List events after a sequence.
- Mark terminal runs.
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
    last_sequence INTEGER NOT NULL DEFAULT 0
);
```

### Live Subscribers

Replace the single `_run_streams[run_id] = queue` model with per-run subscriber
sets:

- `_run_subscribers: Dict[str, set[asyncio.Queue]]`
- `_run_statuses` remains the pollable status cache.
- `_active_run_agents` and `_active_run_tasks` remain for explicit stop support.

When an event is produced:

1. Append it to `RunEventStore`.
2. Update `_run_statuses[run_id]["last_sequence"]`.
3. Put the sequenced event into every active subscriber queue.

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

The existing `POST /v1/runs/{run_id}/stop` remains the explicit cancellation
path. It should still call `agent.interrupt("Stop requested via API")` and
cancel the task as it does today.

## Error Handling

- Unknown `run_id`: return `404 run_not_found`.
- Invalid `after` value: return `400 invalid_request_error`.
- `after` greater than current last sequence: attach live and keep the
  connection open until new events or terminal closure.
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
- Completed runs can still replay their terminal event.
- Multiple subscribers can observe the same run without stealing events from
  one another.
- `POST /v1/runs/{run_id}/stop` still interrupts the active agent and emits a
  terminal cancellation/failure event.
- TTL cleanup removes old terminal run events but does not remove running runs.
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

## Upstream Submission Checklist

Before opening an official Hermes PR, the change still needs:

- Implementation of `RunEventStore`.
- Refactor of `/v1/runs` event delivery from single queue to replayable
  multi-subscriber streams.
- Tests covering disconnect, replay, completion replay, stop behavior, auth,
  and cleanup.
- Capability metadata update in `/v1/capabilities`.
- Minimal docs or release note explaining the new resumable run stream behavior.
- Manual verification with an SSE client that disconnects and reconnects using
  `Last-Event-ID`.

## Non-Goals

- Do not change `/v1/chat/completions` disconnect semantics.
- Do not change `/v1/responses` disconnect semantics in this PR.
- Do not attempt process-restart recovery of in-flight agent execution. Stored
  events and terminal status can survive process restart, but a Python thread
  that was running an agent cannot be resurrected after the process exits.
- Do not rebuild the dashboard chat transcript in React. The dashboard chat tab
  remains PTY/TUI-backed.
