# Approvals Persistence

This document describes the SQLite persistence layer for the approval system.

## Overview

Approvals are now persisted to SQLite, replacing the previous in-memory only approach. This ensures approval history survives server restarts.

## Data Model

The `approvals` table schema:

| Column        | Type             | Description                                               |
| ------------- | ---------------- | --------------------------------------------------------- |
| `id`          | TEXT PRIMARY KEY | Unique approval identifier                                |
| `session_id`  | TEXT             | Hermes session this approval belongs to                   |
| `agent_id`    | TEXT             | Agent identifier (e.g., `agent-<session_id>`)             |
| `status`      | TEXT             | `pending`, `approved`, or `rejected`                      |
| `title`       | TEXT             | Short human-readable summary                              |
| `kind`        | TEXT             | Always `command` in this version                          |
| `details`     | TEXT             | Longer explanation of why approval was requested          |
| `command`     | TEXT             | The actual command that triggered the approval            |
| `created_at`  | TEXT             | ISO 8601 UTC timestamp when created                       |
| `updated_at`  | TEXT             | ISO 8601 UTC timestamp of last update                     |
| `resolved_at` | TEXT             | ISO 8601 UTC timestamp when resolved (nullable)           |
| `resolved_by` | TEXT             | Session key that resolved it (nullable)                   |
| `choice`      | TEXT             | Resolution choice: `session`, `deny`, `always` (nullable) |

### Indexes

- `idx_approvals_session` on `session_id`
- `idx_approvals_status` on `status`
- `idx_approvals_created` on `created_at DESC`

## Runtime Architecture

Approval state uses a dual-source pattern:

### Pending Approvals

**Source of truth**: `tools.approval._gateway_queues` (in-memory)

The gateway queue is required because it holds `threading.Event` objects that unblock agent threads when approvals are resolved. This cannot be persisted to SQLite.

**Persistence**: ApprovalDB maintains a parallel persistent record of pending approvals via `upsert_from_queue()`. This allows:

- Recovery of pending approvals after a crash (though threads cannot be re-blocked)
- Audit trail of all approvals including those that were pending at crash time

### Resolved Approvals

**Source of truth**: ApprovalDB (SQLite)

Once an approval is resolved (approved or rejected), it is stored exclusively in SQLite. The in-memory queue entry is removed by `resolve_gateway_approval()`.

## Key Operations

### Creating an Approval

When `_on_approval_queued` fires (via the `register_approval_listener()` hook):

1. The approval is added to `_gateway_queues` (in-memory, required for threading)
2. `ApprovalDB.upsert_from_queue()` persists the approval record

### Resolving an Approval

When `POST /api/approvals/{id}/approve` or `/reject` is called:

1. `resolve_gateway_approval()` is called to unblock the agent thread
2. `ApprovalDB.resolve_approval()` updates the status in SQLite

### Listing Approvals

`GET /api/approvals` combines:

- **Pending**: from `_gateway_queues` (in-memory, live)
- **Resolved**: from `ApprovalDB.list_approvals()` (SQLite, persisted)

## Schema Evolution

| Version | Change                  |
| ------- | ----------------------- |
| 7       | Initial approvals table |

Migrations are handled automatically in `SessionDB._init_schema()` and `ApprovalDB._init_schema()`.

## Limitations

### Pending Approvals After Restart

When the server restarts:

1. Pending approvals are lost from `_gateway_queues` (threading.Event cannot be restored)
2. The persistent record in ApprovalDB remains with `status=pending`
3. These approvals will appear in `GET /api/approvals` but:
   - They cannot be resolved via the API (no associated thread to unblock)
   - The blocked agent threads received an implicit denial

This is a fundamental limitation since `threading.Event` state cannot be persisted.

### Crash Recovery

If the server crashes while an approval is pending:

- The approval record remains in SQLite with `status=pending`
- On restart, it will be visible in the API but cannot be resolved

## Files

- `hermes_state.py` - Contains `ApprovalDB` class
- `hermes_cli/web_server.py` - Integrates ApprovalDB with the web API
