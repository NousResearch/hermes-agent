# HermesWeb Integration Follow-ups

## Delivered in this round (agents + approvals sprint)

- Added `GET /api/agents` — derives agent list from session store, returns `primary` and `subagent` kinds with derived status (`active`/`idle`/`ended`/`error`)
- Added `GET /api/approvals` — reads from the live `_gateway_queues` in-memory approval queue, plus SQLite-backed approval history
- Added `POST /api/approvals/{id}/approve` — calls `resolve_gateway_approval()` to unblock the waiting agent thread, persists resolution to SQLite
- Added `POST /api/approvals/{id}/reject` — same resolution path with `"deny"` choice, persists resolution to SQLite
- Added `approval.requested` WebSocket event — hooked via `register_approval_listener()` in `tools/approval.py`, emits when an approval is queued
- Added `approval.resolved` WebSocket event — emitted when an approval is approved or rejected via REST API
- Added `agent.updated` WebSocket event — emitted after `POST /api/chat` completes with agent state
- Added SQLite persistence for approvals (ApprovalDB class in hermes_state.py) — approval history survives server restarts
- Updated frontend types (`Agent`, `Approval`) to match new backend field names
- Updated frontend components (`AgentCard`, `ApprovalQueue`, `PendingApprovals`, `ActiveAgents`) to use new field names
- Updated frontend WS handlers (`connection.ts`) to use new event names
- Updated `docs/hermesweb-backend-contracts.md` with full contracts for all new endpoints and events
- Added `docs/approvals-persistence.md` with architecture documentation for the approval persistence layer

## Delivered in tasks sprint

- Added `GET /api/tasks` — list tasks with optional filters (status, agent_id, session_id)
- Added `POST /api/tasks` — create a new administrative task
- Added `PATCH /api/tasks/{id}` — update an existing task (title, description, status, priority, agent_id, session_id)
- Added `DELETE /api/tasks/{id}` — permanently delete a task
- Added `task.created` WebSocket event — emitted when a task is created
- Added `task.updated` WebSocket event — emitted when a task is updated
- Added `task.deleted` WebSocket event — emitted when a task is deleted
- Added TaskDB class in hermes_state.py — SQLite-backed task storage
- Updated `docs/hermesweb-backend-contracts.md` with full contracts for all task endpoints and events

## Still missing

- richer backend status events over websocket (gateway state changes, platform events)
- diff/file/code preview artifact endpoints
- replayable event log or durable websocket resume
- `GET /api/summary/daily` — daily summary aggregation

## Next recommended order

1. Implement richer `agent.updated` events from the live runtime (currently only emitted after chat turns)
2. Add `GET /api/summary/daily` by aggregating session/task/approval data
3. Add `diff/file` artifact endpoints for the code preview feature

## Known risks

- `GET /api/agents` derives agents from persisted sessions, not from live runtime presence. Active status is inferred from recency (< 5 min), not a heartbeat.
- `POST /api/chat` is synchronous. Long turns block until completion.
- `/ws` is in-memory only. Server restarts drop all subscriptions and there is no replay buffer.
- Auth token is an ephemeral session token generated per server start (injected via HTML). Works for local development but requires the web server to be the token source.
- Pending approvals after restart: `_gateway_queues` is in-memory only. Server restarts clear pending approvals (blocked agent threads receive implicit denial). ApprovalDB preserves the record but the thread cannot be unblocked. See [approvals-persistence.md](../docs/approvals-persistence.md).
- Tasks are purely administrative/manual items, not automatically created by the agent runtime.
