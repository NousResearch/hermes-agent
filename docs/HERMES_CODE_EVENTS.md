# Hermes Code Events (P3)

P3 adds a central realtime event fanout layer for Code Mode, backed by
persisted `code_events` records and additive WebSocket subscriptions.

## Event Bus

Central service: `hermes_cli/code/event_bus.py`

Responsibilities:

- normalize event envelope
- recursive payload redaction
- persist into `code_events`
- in-process fanout to subscribers
- replay/catch-up from persisted events
- filter matching by Code Mode dimensions

All published events are redacted before persistence and before fanout.

## Envelope

```json
{
  "id": "uuid",
  "type": "code.approval.approved",
  "version": 1,
  "timestamp": "2026-01-01T00:00:00+00:00",
  "workspace_id": "optional",
  "code_session_id": "optional",
  "orchestrated_run_id": "optional",
  "approval_id": "optional",
  "github_repo_full_name": "optional",
  "payload": {},
  "metadata": {}
}
```

## WebSocket

Additive endpoint:

- `WS /api/code/events/ws`

Supported query params:

- `type`
- `workspace_id`
- `code_session_id`
- `orchestrated_run_id`
- `approval_id`
- `github_repo_full_name`
- `since_id`
- `replay=true|false`
- `limit`
- `token` (dashboard session token)

Behavior:

1. Optional replay/catch-up (ordered by persisted insertion order)
2. Live filtered stream
3. Heartbeat frames when idle
4. Safe disconnect and subscriber cleanup

## REST

- `GET /api/code/events`
- `GET /api/code/events/recent`
- `GET /api/code/events/summary`
- `GET /api/code/events/subscriptions`

Filter support:

- `type`
- `workspace_id`
- `code_session_id`
- `orchestrated_run_id`
- `approval_id`
- `github_repo_full_name`
- `since_id`
- `limit`

## Replay semantics

- `since_id` does not rely on UUID ordering.
- Replay is deterministic using persisted insertion order.
- Returned items are events strictly after `since_id`.

## Security

Redaction applies recursively to keys including:

- `token`
- `access_token`
- `refresh_token`
- `authorization`
- `private_key`
- `webhook_secret`
- `client_secret`
- `secret`
- `password`
- `HERMES_GITHUB_DEV_PAT`

Raw webhook payload bodies and secret-bearing values are not exposed.

## Compatibility

- Existing `/api/ws` and dashboard sidecar websocket behavior are unchanged.
- Existing REST polling remains supported.
- Legacy event names are preserved where needed; canonical P3 names are added.

## Deferred

- No frontend event panel implementation in this phase.
- `hermesWeb` UI remains deferred.
