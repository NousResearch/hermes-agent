# HermesWeb Backend Contracts

This document defines the first production-facing Hermes Agent contracts that HermesWeb can consume without mocks.

All endpoints below use the same bearer token currently required by the existing `/api/sessions*` routes.

- HTTP auth: `Authorization: Bearer <token>`
- WebSocket auth: query param `?token=<token>` or `Authorization: Bearer <token>`
- Dates/times: Unix seconds for persisted message/session timestamps, ISO 8601 UTC strings for WebSocket event envelopes

## `GET /api/chat/history`

Purpose: return the persisted message history for one Hermes session.

Auth: required.

Query params:

- `session_id` (required): exact session ID or unique session ID prefix
- `sessionId` (optional alias): accepted for frontend compatibility, same semantics

Response shape:

```json
{
  "session_id": "20260423_103015_a1b2c3",
  "messages": [
    {
      "id": 101,
      "session_id": "20260423_103015_a1b2c3",
      "role": "user",
      "content": "Summarize the latest run.",
      "timestamp": 1776943815.415,
      "tool_name": null,
      "tool_calls": null,
      "tool_call_id": null,
      "finish_reason": null,
      "reasoning": null
    },
    {
      "id": 102,
      "session_id": "20260423_103015_a1b2c3",
      "role": "assistant",
      "content": "Here is the latest run summary...",
      "timestamp": 1776943829.208,
      "tool_name": null,
      "tool_calls": null,
      "tool_call_id": null,
      "finish_reason": "stop",
      "reasoning": null
    }
  ],
  "total": 2
}
```

Errors:

- `400` when `session_id` is missing
- `401` when auth is missing/invalid
- `404` when session does not exist or prefix is ambiguous
- `500` for unexpected server errors

Notes:

- Message shape intentionally mirrors the persisted session/message model used by `/api/sessions/{id}/messages`.
- No pagination yet. HermesWeb should expect full-session history in this first version.

## `POST /api/chat`

Purpose: execute one real Hermes chat turn, persist it to the session store, and return the resulting turn payload.

Auth: required.

Request body:

```json
{
  "content": "Continue the deployment checklist.",
  "session_id": "20260423_103015_a1b2c3"
}
```

Accepted body fields:

- `content` (required): user message text
- `session_id` (optional): existing session ID or unique prefix to continue
- `sessionId` (optional alias): accepted for frontend compatibility

Session behavior:

- If `session_id` is provided, Hermes resumes that persisted session and loads its prior message history from SQLite.
- If `session_id` is omitted, Hermes creates a fresh session through the normal runtime path and returns its generated `session_id`.
- If a provided `session_id` does not resolve, the endpoint returns `404` instead of silently creating a new session.

Response shape:

```json
{
  "session_id": "20260423_103015_a1b2c3",
  "session": {
    "id": "20260423_103015_a1b2c3",
    "title": "deployment checklist",
    "source": "web",
    "model": "gpt-5.4",
    "started_at": 1776943815.002,
    "ended_at": null,
    "end_reason": null,
    "message_count": 8,
    "tool_call_count": 2,
    "input_tokens": 1420,
    "output_tokens": 388,
    "last_active": 1776943829.208,
    "is_active": true
  },
  "messages": [
    {
      "id": 101,
      "session_id": "20260423_103015_a1b2c3",
      "role": "user",
      "content": "Continue the deployment checklist.",
      "timestamp": 1776943815.415,
      "tool_name": null,
      "tool_calls": null,
      "tool_call_id": null,
      "finish_reason": null,
      "reasoning": null
    },
    {
      "id": 102,
      "session_id": "20260423_103015_a1b2c3",
      "role": "assistant",
      "content": "Next we should validate the rollback path...",
      "timestamp": 1776943829.208,
      "tool_name": null,
      "tool_calls": null,
      "tool_call_id": null,
      "finish_reason": "stop",
      "reasoning": null
    }
  ],
  "user_message": {
    "id": 101,
    "session_id": "20260423_103015_a1b2c3",
    "role": "user",
    "content": "Continue the deployment checklist.",
    "timestamp": 1776943815.415,
    "tool_name": null,
    "tool_calls": null,
    "tool_call_id": null,
    "finish_reason": null,
    "reasoning": null
  },
  "assistant_message": {
    "id": 102,
    "session_id": "20260423_103015_a1b2c3",
    "role": "assistant",
    "content": "Next we should validate the rollback path...",
    "timestamp": 1776943829.208,
    "tool_name": null,
    "tool_calls": null,
    "tool_call_id": null,
    "finish_reason": "stop",
    "reasoning": null
  },
  "completed": true,
  "partial": false,
  "usage": {
    "input_tokens": 1420,
    "output_tokens": 388,
    "total_tokens": 1808
  }
}
```

Errors:

- `400` when `content` is empty
- `401` when auth is missing/invalid
- `404` when `session_id` does not resolve
- `503` when no runtime provider/model/API key is available
- `500` for unexpected server/runtime errors

Notes:

- This endpoint uses the real `AIAgent.run_conversation()` path, not a fake response generator.
- This first version is request/response only. It does not stream token deltas.
- The `messages` array contains only the new messages created by this turn.

## `WS /ws`

Purpose: maintain a live backend connection for HermesWeb connection state and small operational updates.

Auth:

- `ws://host/ws?token=<token>`
- or `Authorization: Bearer <token>` during websocket handshake

Optional query params:

- `session_id`: if present, the socket only receives session-scoped events for that session

Initial server event on connect:

```json
{
  "type": "hello",
  "connection_id": "f7a42bc913dd",
  "sent_at": "2026-04-23T19:30:15.221844+00:00",
  "session_id": "20260423_103015_a1b2c3",
  "status": {
    "version": "0.9.0",
    "release_date": "2026-04-20",
    "backend": "hermes-agent"
  }
}
```

Heartbeat:

- Server sends a `heartbeat` event every 15s of idle time.
- Client may send plain text `ping`; server replies with `pong`.

Event types currently emitted:

### `heartbeat`

```json
{
  "type": "heartbeat",
  "connection_id": "f7a42bc913dd",
  "sent_at": "2026-04-23T19:30:30.226190+00:00",
  "session_id": null
}
```

### `pong`

```json
{
  "type": "pong",
  "connection_id": "f7a42bc913dd",
  "sent_at": "2026-04-23T19:30:18.006192+00:00",
  "session_id": null
}
```

### `message.created`

Emitted after `POST /api/chat` persists each new message.

```json
{
  "type": "message.created",
  "sent_at": "2026-04-23T19:31:02.881015+00:00",
  "session_id": "20260423_103015_a1b2c3",
  "message": {
    "id": 102,
    "session_id": "20260423_103015_a1b2c3",
    "role": "assistant",
    "content": "Next we should validate the rollback path...",
    "timestamp": 1776943829.208,
    "tool_name": null,
    "tool_calls": null,
    "tool_call_id": null,
    "finish_reason": "stop",
    "reasoning": null
  }
}
```

### `session.updated`

Emitted after `POST /api/chat` completes and the session metadata is refreshed.

```json
{
  "type": "session.updated",
  "sent_at": "2026-04-23T19:31:02.882411+00:00",
  "session": {
    "id": "20260423_103015_a1b2c3",
    "title": "deployment checklist",
    "source": "web",
    "model": "gpt-5.4",
    "started_at": 1776943815.002,
    "ended_at": null,
    "end_reason": null,
    "message_count": 8,
    "tool_call_count": 2,
    "input_tokens": 1420,
    "output_tokens": 388,
    "last_active": 1776943829.208,
    "is_active": true
  }
}
```

Reconnect expectations:

- HermesWeb should treat the socket as best-effort operational telemetry, not authoritative streaming state.
- On reconnect, HermesWeb should re-fetch any active session history via HTTP if it cares about missed events.
- No event replay buffer exists yet.

Current non-goals:

- no token streaming
- no guaranteed delivery
- no server-side subscriptions beyond optional `session_id` filter
- no approval or agent events yet

## `GET /api/agents`

Purpose: return the list of agents currently observable by the backend. An "agent" in this context is a Hermes agent session — a running or recently active `AIAgent` instance tied to a session.

Auth: required.

Response shape:

```json
{
  "agents": [
    {
      "id": "agent-main",
      "session_id": "20260423_103015_a1b2c3",
      "name": "Hermes",
      "kind": "primary",
      "status": "idle",
      "model": "gpt-5.4",
      "platform": "web",
      "updated_at": "2026-04-23T19:31:02.882411+00:00"
    }
  ]
}
```

Field definitions:

- `id` (string): stable agent identifier derived from the session ID (`agent-<session_id>` or `"agent-main"` for the most recent active session)
- `session_id` (string): the Hermes session this agent is bound to
- `name` (string): display name — uses session title if available, otherwise `"Hermes"`
- `kind` (string): `"primary"` for top-level agents, `"subagent"` for delegated children
- `status` (string): one of `"idle"`, `"active"`, `"ended"`, `"error"`
  - `"active"`: session has no `ended_at` and was active within the last 5 minutes
  - `"idle"`: session has no `ended_at` but last activity is older than 5 minutes
  - `"ended"`: session has an `ended_at` timestamp
  - `"error"`: session ended with an error reason
- `model` (string): the model used by this agent session
- `platform` (string): where the agent is running (`"web"`, `"cli"`, `"telegram"`, etc.)
- `updated_at` (string): ISO 8601 UTC timestamp of last activity

Limitations:

- This first version derives agents from the persisted session store (SQLite). It does NOT track in-memory running agents that haven't written a session record yet.
- Subagent detection is based on `parent_session_id` in the sessions table.
- There is no live heartbeat or presence signal — `"active"` is inferred from recency.
- The list is capped at the most recent 50 sessions to keep the payload small.

Errors:

- `401` when auth is missing/invalid
- `500` for unexpected server errors

## `GET /api/approvals`

Purpose: return the list of pending (and optionally recently resolved) command approval requests from the internal approval queue.

Auth: required.

Query params:

- `status` (optional): filter by status. Accepted values: `"pending"`, `"approved"`, `"rejected"`. Default: `"pending"`.
- `limit` (optional): max number of approvals to return. Default: `50`.

Response shape:

```json
{
  "approvals": [
    {
      "id": "approval-abc123",
      "session_id": "20260423_103015_a1b2c3",
      "agent_id": "agent-main",
      "status": "pending",
      "title": "recursive delete",
      "kind": "command",
      "details": "Command matches dangerous pattern: recursive delete",
      "command": "rm -rf /tmp/old-build",
      "created_at": "2026-04-23T19:35:12.000000+00:00"
    }
  ],
  "total": 1
}
```

Field definitions:

- `id` (string): unique approval identifier (generated from a hash of session_key + queue position)
- `session_id` (string): the Hermes session this approval belongs to
- `agent_id` (string): agent identifier (same convention as `GET /api/agents`)
- `status` (string): `"pending"`, `"approved"`, or `"rejected"`
- `title` (string): short human-readable summary of the danger (the pattern description)
- `kind` (string): always `"command"` in this version (future: `"tool"`, `"patch"`, etc.)
- `details` (string): longer explanation of why approval was requested
- `command` (string): the actual command that triggered the approval request
- `created_at` (string): ISO 8601 UTC timestamp when the approval was created

Data source:

- Pending approvals come from the in-memory `_gateway_queues` in `tools/approval.py`. These are real, live approval requests that are blocking agent threads. A parallel copy is persisted to SQLite for crash recovery via `ApprovalDB.upsert_from_queue()`.
- Resolved (approved/rejected) approvals are persisted in SQLite via `ApprovalDB.resolve_approval()`.

Limitations:

- Server restarts clear pending approvals from `_gateway_queues`. The blocked agent threads receive an implicit "deny". Any pending approvals that were in-flight during a crash will appear in the API as `status=pending` but cannot be resolved (no associated thread to unblock).
- The `id` is derived from session key + queue index + timestamp hash, not a database primary key. It is stable within a server lifetime.
- See [approvals-persistence.md](approvals-persistence.md) for full architecture documentation.

Errors:

- `401` when auth is missing/invalid
- `500` for unexpected server errors

## `POST /api/approvals/{id}/approve`

Purpose: approve a pending command approval request, unblocking the waiting agent thread.

Auth: required.

Request body: empty (`{}`) or omitted.

Response shape:

```json
{
  "ok": true,
  "approval_id": "approval-abc123",
  "status": "approved"
}
```

Effect on runtime:

- Calls `resolve_gateway_approval(session_key, "session")` which sets the result on the `_ApprovalEntry` and signals the `threading.Event`, unblocking the agent thread.
- The agent continues execution as if the user had sent `/approve session` via a messaging platform.

Errors:

- `401` when auth is missing/invalid
- `404` when the approval ID does not match any pending approval
- `409` when the approval has already been resolved (approved or rejected)
- `500` for unexpected server errors

## `POST /api/approvals/{id}/reject`

Purpose: reject a pending command approval request, causing the agent to skip the dangerous command.

Auth: required.

Request body (optional):

```json
{
  "reason": "Too risky for production"
}
```

Response shape:

```json
{
  "ok": true,
  "approval_id": "approval-abc123",
  "status": "rejected"
}
```

Effect on runtime:

- Calls `resolve_gateway_approval(session_key, "deny")` which sets the result on the `_ApprovalEntry` and signals the `threading.Event`.
- The agent receives a denial response and will skip executing the dangerous command.

Errors:

- `401` when auth is missing/invalid
- `404` when the approval ID does not match any pending approval
- `409` when the approval has already been resolved (approved or rejected)
- `500` for unexpected server errors

## `GET /api/tasks`

Purpose: return the list of administrative tasks.

Auth: required.

Query params:

- `status` (optional): filter by status. Accepted values: `"todo"`, `"in_progress"`, `"review"`, `"done"`, `"archived"`.
- `agent_id` (optional): filter by assigned agent.
- `session_id` (optional): filter by associated session.
- `limit` (optional): max number of tasks to return. Default: `100`.

Response shape:

```json
{
  "tasks": [
    {
      "id": "task-abc123def456",
      "title": "Review deployment checklist",
      "description": "Go through the deployment checklist and verify all items",
      "status": "todo",
      "priority": "high",
      "agent_id": null,
      "session_id": null,
      "created_at": "2026-04-23T19:35:12.000000+00:00",
      "updated_at": "2026-04-23T19:35:12.000000+00:00",
      "completed_at": null
    }
  ],
  "total": 1
}
```

Field definitions:

- `id` (string): unique task identifier (generated from timestamp + random)
- `title` (string): task title (required)
- `description` (string): optional detailed description
- `status` (string): one of `"todo"`, `"in_progress"`, `"review"`, `"done"`, `"archived"`
- `priority` (string): one of `"low"`, `"medium"`, `"high"`
- `agent_id` (string): optional associated agent ID
- `session_id` (string): optional associated session ID
- `created_at` (string): ISO 8601 UTC timestamp when the task was created
- `updated_at` (string): ISO 8601 UTC timestamp of last update
- `completed_at` (string): ISO 8601 UTC timestamp when status became `"done"`, or null

Data source:

- Tasks are stored in SQLite via `TaskDB` class in `hermes_state.py`.
- Tasks are purely administrative — not tied to the agent runtime.

Limitations:

- Tasks are manual/administrative items, not automatically created by the agent runtime.
- No sub-tasks or dependencies in this version.
- No due dates in this version.

Errors:

- `401` when auth is missing/invalid
- `500` for unexpected server errors

## `POST /api/tasks`

Purpose: create a new administrative task.

Auth: required.

Request body:

```json
{
  "title": "Review deployment checklist",
  "description": "Go through the deployment checklist and verify all items",
  "priority": "high",
  "agent_id": null,
  "session_id": null
}
```

Accepted body fields:

- `title` (required): task title
- `description` (optional): detailed description
- `priority` (optional): `"low"`, `"medium"`, or `"high"`. Default: `"medium"`
- `agent_id` (optional): agent ID to assign the task to
- `session_id` (optional): session ID to associate with the task

Response shape (returns the created task):

```json
{
  "id": "task-abc123def456",
  "title": "Review deployment checklist",
  "description": "Go through the deployment checklist and verify all items",
  "status": "todo",
  "priority": "high",
  "agent_id": null,
  "session_id": null,
  "created_at": "2026-04-23T19:35:12.000000+00:00",
  "updated_at": "2026-04-23T19:35:12.000000+00:00",
  "completed_at": null
}
```

Errors:

- `400` when `title` is missing or empty
- `401` when auth is missing/invalid
- `500` for unexpected server errors

WebSocket:

- Emits `task.created` event over `WS /ws` on success.

## `PATCH /api/tasks/{id}`

Purpose: update an existing task.

Auth: required.

Request body (all fields optional):

```json
{
  "title": "Updated title",
  "description": "Updated description",
  "status": "in_progress",
  "priority": "low",
  "agent_id": "agent-123",
  "session_id": null
}
```

Accepted body fields:

- `title` (optional): updated task title
- `description` (optional): updated description
- `status` (optional): one of `"todo"`, `"in_progress"`, `"review"`, `"done"`, `"archived"`
- `priority` (optional): one of `"low"`, `"medium"`, `"high"`
- `agent_id` (optional): updated agent assignment
- `session_id` (optional): updated session association

Behavior:

- When `status` transitions to `"done"`, `completed_at` is automatically set to the current timestamp.
- Setting `session_id` to `null` clears the session association.

Response shape (returns the updated task):

```json
{
  "id": "task-abc123def456",
  "title": "Updated title",
  "description": "Updated description",
  "status": "in_progress",
  "priority": "low",
  "agent_id": "agent-123",
  "session_id": null,
  "created_at": "2026-04-23T19:35:12.000000+00:00",
  "updated_at": "2026-04-23T19:40:00.000000+00:00",
  "completed_at": null
}
```

Errors:

- `400` when no fields are provided, or when `status`/`priority` has an invalid value
- `401` when auth is missing/invalid
- `404` when the task ID does not exist
- `500` for unexpected server errors

WebSocket:

- Emits `task.updated` event over `WS /ws` on success.

## `DELETE /api/tasks/{id}`

Purpose: permanently delete a task.

Auth: required.

Response shape:

```json
{
  "ok": true
}
```

Errors:

- `401` when auth is missing/invalid
- `404` when the task ID does not exist
- `500` for unexpected server errors

WebSocket:

- Emits `task.deleted` event over `WS /ws` on success.

## WebSocket events: `approval.requested`, `approval.resolved`, `agent.updated`, `task.created`, `task.updated`, `task.deleted`

These events are emitted over `WS /ws` when approvals or agent state change.

### `approval.requested`

Emitted when a new approval is added to the gateway queue (i.e. an agent encounters a dangerous command and blocks).

```json
{
  "type": "approval.requested",
  "sent_at": "2026-04-23T19:35:12.000000+00:00",
  "session_id": "20260423_103015_a1b2c3",
  "approval": {
    "id": "approval-abc123",
    "session_id": "20260423_103015_a1b2c3",
    "agent_id": "agent-main",
    "status": "pending",
    "title": "recursive delete",
    "kind": "command",
    "details": "Command matches dangerous pattern: recursive delete",
    "command": "rm -rf /tmp/old-build",
    "created_at": "2026-04-23T19:35:12.000000+00:00"
  }
}
```

### `approval.resolved`

Emitted after an approval is approved or rejected (via the REST API or gateway command).

```json
{
  "type": "approval.resolved",
  "sent_at": "2026-04-23T19:35:45.000000+00:00",
  "session_id": "20260423_103015_a1b2c3",
  "approval_id": "approval-abc123",
  "status": "approved"
}
```

### `agent.updated`

Emitted when agent state changes are detected (currently: when sessions are updated via `POST /api/chat`).

```json
{
  "type": "agent.updated",
  "sent_at": "2026-04-23T19:31:02.882411+00:00",
  "session_id": "20260423_103015_a1b2c3",
  "agent": {
    "id": "agent-main",
    "session_id": "20260423_103015_a1b2c3",
    "name": "Hermes",
    "kind": "primary",
    "status": "active",
    "model": "gpt-5.4",
    "platform": "web",
    "updated_at": "2026-04-23T19:31:02.882411+00:00"
  }
}
```

### `task.created`

Emitted when a new task is created via `POST /api/tasks`.

```json
{
  "type": "task.created",
  "sent_at": "2026-04-23T19:35:12.000000+00:00",
  "task": {
    "id": "task-abc123def456",
    "title": "Review deployment checklist",
    "description": "Go through the deployment checklist",
    "status": "todo",
    "priority": "high",
    "agent_id": null,
    "session_id": null,
    "created_at": "2026-04-23T19:35:12.000000+00:00",
    "updated_at": "2026-04-23T19:35:12.000000+00:00",
    "completed_at": null
  }
}
```

### `task.updated`

Emitted when a task is updated via `PATCH /api/tasks/{id}`.

```json
{
  "type": "task.updated",
  "sent_at": "2026-04-23T19:40:00.000000+00:00",
  "task": {
    "id": "task-abc123def456",
    "title": "Review deployment checklist",
    "description": "Go through the deployment checklist",
    "status": "in_progress",
    "priority": "high",
    "agent_id": null,
    "session_id": null,
    "created_at": "2026-04-23T19:35:12.000000+00:00",
    "updated_at": "2026-04-23T19:40:00.000000+00:00",
    "completed_at": null
  }
}
```

### `task.deleted`

Emitted when a task is deleted via `DELETE /api/tasks/{id}`.

```json
{
  "type": "task.deleted",
  "sent_at": "2026-04-23T19:45:00.000000+00:00",
  "task_id": "task-abc123def456"
}
```

## `GET /api/summary/daily`

Purpose: return an administrative daily summary with aggregated metrics from real backend data.

Auth: required.

Query params:

- `date` (optional): target date in `YYYY-MM-DD` format (UTC). Defaults to today.

Response shape:

```json
{
  "date": "2026-04-23",
  "sessions": {
    "total_today": 5,
    "active": 2
  },
  "tasks": {
    "total": 9,
    "todo": 2,
    "in_progress": 3,
    "review": 1,
    "done": 2,
    "archived": 1,
    "completed_today": 3
  },
  "approvals": {
    "pending": 1,
    "approved_today": 4,
    "rejected_today": 1
  },
  "agents": {
    "active": 2,
    "idle": 3
  }
}
```

Field definitions:

- `date` (string): the date this summary covers, in `YYYY-MM-DD` UTC format.
- `sessions.total_today` (integer): count of sessions with `started_at` on the target date.
- `sessions.active` (integer): count of sessions with no `ended_at` AND last message within 5 minutes of now. Heuristic — see note below.
- `tasks.total` (integer): total count of all tasks.
- `tasks.{status}` (integer): count of tasks with each status (`todo`, `in_progress`, `review`, `done`, `archived`).
- `tasks.completed_today` (integer): count of tasks with `completed_at` on the target date.
- `approvals.pending` (integer): count of approvals with `status = 'pending'`.
- `approvals.approved_today` (integer): count of approvals with `status = 'approved'` AND `resolved_at` on the target date.
- `approvals.rejected_today` (integer): count of approvals with `status = 'rejected'` AND `resolved_at` on the target date.
- `agents.active` (integer): count of agents (derived from sessions) that are currently active (same heuristic as `sessions.active`).
- `agents.idle` (integer): count of agents with no `ended_at` but last activity older than 5 minutes.

Heuristics and limitations:

- `sessions.active` and `agents.active` use the same 5-minute window as `GET /api/agents` — they reflect operational presence, not exact state.
- `sessions.total_today` is an exact count from `started_at` timestamp.
- `tasks.completed_today` is exact when `completed_at` is set.
- `approvals.approved_today` / `rejected_today` are exact when `resolved_at` is set.
- The summary is computed on every request — no caching.
- Server restart does not affect summary accuracy.

Errors:

- `400` when `date` format is invalid
- `401` when auth is missing/invalid
- `500` for unexpected server errors

Notes:

- The frontend should recompute local derived state (e.g. "did I approve anything today?") from this payload.
- No WebSocket events are emitted for summary changes — the frontend should re-fetch on relevant events (`task.updated`, `approval.resolved`, `session.updated`).

## `GET /api/sessions/{session_id}/artifacts`

Purpose: return the list of file-change artifacts (diffs, patches, written files) produced during a session.

Auth: required.

Path params:

- `session_id` (required): exact session ID or unique session ID prefix

Response shape:

```json
{
  "session_id": "20260423_103015_a1b2c3",
  "artifacts": [
    {
      "id": "msg-456",
      "tool_call_id": "call_abc123",
      "tool_name": "patch",
      "path": "src/services/connection.ts",
      "status": "modified",
      "diff": "--- a/src/services/connection.ts\n+++ b/src/services/connection.ts\n@@ -1,3 +1,4 @@\n old line\n+new line\n",
      "additions": 1,
      "deletions": 1,
      "timestamp": 1776943829.208
    }
  ],
  "total": 1
}
```

Field definitions:

- `id` (string): the message ID from the messages table
- `tool_call_id` (string): the tool call ID from the message
- `tool_name` (string): the tool that produced this artifact — `"patch"` (produces diff) or `"write_file"` (no diff)
- `path` (string): file path modified, derived from `files_modified[0]` for `patch` results
- `status` (string): one of `"added"`, `"modified"`, `"deleted"` — derived from diff analysis
- `diff` (string): unified diff content. Empty string `""` when the tool does not produce a diff (e.g. `write_file`)
- `additions` (integer): count of `+` lines in the diff (0 if no diff)
- `deletions` (integer): count of `-` lines in the diff (0 if no diff)
- `timestamp` (number): Unix timestamp from the message

Data source:

- Artifacts are extracted from session messages with `role = 'tool'` and `tool_name IN ('patch', 'write_file')`.
- The `content` field of each message is parsed as JSON. For `patch` results, the `diff` field is extracted. For `write_file` results, no diff is available.
- Multiple file changes within one `patch` result (e.g. V4A multi-file patch) are expanded into one artifact per file.

Limitations:

- `write_file` tool results do not contain diff content — the `diff` field will be `""` and `additions`/`deletions` will be `0`.
- Artifacts are derived from tool call results stored in message history — no live filesystem scan is performed.
- The endpoint does not emit WebSocket events. The frontend should fetch artifacts via HTTP when navigating to code preview.

Errors:

- `401` when auth is missing/invalid
- `404` when the session does not exist or prefix is ambiguous
- `500` for unexpected server errors
