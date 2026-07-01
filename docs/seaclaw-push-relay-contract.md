# SeaClaw Push Relay Contract

Hermes emits push **notification intents** only. It does not hold APNs keys,
certificates, key IDs, team IDs, or provider tokens. A small relay service owns
APNs credentials and accepts authenticated intent POSTs from Hermes.

## Gateway configuration

```yaml
push_notifications:
  enabled: true
  relay_url: "https://relay.example.com/v1/hermes/notification-intents"
  relay_token_env: "HERMES_PUSH_RELAY_TOKEN"
  registration_store: "push_devices.json"
  redact_body: true
  timeout_seconds: 3
  events:
    - approval.request
    - clarify.request
    - message.complete
    - subagent.complete
    - background.complete
    - preview.restart.complete
```

`HERMES_PUSH_RELAY_TOKEN` is a secret and belongs in `~/.hermes/.env`, never in
source code. `relay_url` is the exact endpoint Hermes POSTs to.

## iOS registration RPC

SeaClaw registers after it has a live Hermes gateway session. The JSON-RPC
connection is authenticated by the existing gateway transport/session model.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": "register-push",
  "method": "push.register",
  "params": {
    "session_id": "<live session_id from session.create/resume>",
    "device_id": "ios-device-stable-id",
    "platform": "apns",
    "device_token": "<APNs device token, optional if endpoint_id is set>",
    "endpoint_id": "<relay-owned endpoint id, optional if device_token is set>",
    "events": ["approval.request", "clarify.request", "message.complete"],
    "redact_body": true
  }
}
```

Response never echoes `device_token`:

```json
{
  "registration": {
    "device_id": "ios-device-stable-id",
    "platform": "apns",
    "endpoint_id": "relay-endpoint-123",
    "session_key": "<durable stored_session_id>",
    "last_live_session_id": "<live session_id>",
    "events": ["approval.request", "clarify.request", "message.complete"],
    "redact_body": true,
    "has_device_token": true,
    "created_at": "2026-07-01T00:00:00Z",
    "updated_at": "2026-07-01T00:00:00Z"
  },
  "relay": {
    "enabled": true,
    "configured": true,
    "url_configured": true,
    "token_env": "HERMES_PUSH_RELAY_TOKEN",
    "token_configured": true
  }
}
```

Use `push.list {session_id}` to inspect registrations for the current session,
and `push.unregister {session_id, device_id}` to remove one. A registration is
bound to the durable `session_key`/`stored_session_id`; action callbacks may use
the included live `session_id` while that turn is active.

## Relay POST shape

Hermes POSTs JSON to `relay_url` with:

```http
Authorization: Bearer $HERMES_PUSH_RELAY_TOKEN
Content-Type: application/json
User-Agent: Hermes-Push-Intent/1
```

Body:

```json
{
  "contract_version": 1,
  "intent_id": "4f83a4c5b8be4f08b6f4c81725d2a632",
  "created_at": "2026-07-01T00:00:00Z",
  "target": {
    "device_id": "ios-device-stable-id",
    "platform": "apns",
    "endpoint_id": "relay-endpoint-123"
  },
  "session": {
    "stored_session_id": "20260701_120000_ab12cd",
    "session_key": "20260701_120000_ab12cd",
    "live_session_id": "a1b2c3d4"
  },
  "event": {
    "category": "approval.request",
    "title": "Hermes approval needed",
    "body": "Review the pending approval in SeaClaw.",
    "redacted": true,
    "row_id": null,
    "message_id": null
  },
  "action_context": {
    "kind": "approval",
    "rpc_method": "approval.respond",
    "params_base": {"session_id": "a1b2c3d4"},
    "choice_param": "choice",
    "choices": ["once", "session", "always", "deny"],
    "fifo_session_keyed": true,
    "request_id": null
  }
}
```

If `endpoint_id` is absent, `target.device_token` is included instead. The relay
must treat both as sensitive routing data and should not log raw tokens.

## Implemented event categories

Hermes emits intents for:

| Category | Trigger | Action context |
| --- | --- | --- |
| `approval.request` | Terminal/tool approval prompt | `approval.respond` |
| `clarify.request` | Blocking clarify prompt | `clarify.respond` |
| `message.complete` | Agent turn completed | none |
| `subagent.complete` | Delegated child/subagent completion event | none |
| `background.complete` | Background completion event if emitted by the gateway | none |
| `preview.restart.complete` | Preview/background restart task completed | none |

Cron deliveries and messaging-platform-only completions are intentionally not
hooked in this contract yet; they do not flow through the TUI JSON-RPC gateway
event spine that SeaClaw consumes.

## Action callbacks from iOS

Approval notifications call the existing JSON-RPC method:

```json
{
  "jsonrpc": "2.0",
  "id": "approval-action",
  "method": "approval.respond",
  "params": {
    "session_id": "<live_session_id from intent>",
    "choice": "once"
  }
}
```

Valid approval choices are `once`, `session`, `always`, and `deny`. Approval
requests currently have no `request_id`; Hermes resolves them FIFO for the
durable session key behind the live `session_id`. If multiple approvals are
pending in one session, SeaClaw cannot target a specific one until Hermes adds
approval request IDs.

Clarify notifications call:

```json
{
  "jsonrpc": "2.0",
  "id": "clarify-action",
  "method": "clarify.respond",
  "params": {
    "request_id": "<request_id from intent.action_context>",
    "answer": "User answer"
  }
}
```

When `redact_body` is true, notification title/body are generic. SeaClaw can
open the app and fetch live gateway state for full content instead of relying on
the APNs alert body.
