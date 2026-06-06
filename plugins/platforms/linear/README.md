# Linear Agent Sessions platform plugin

This plugin adds a native Hermes gateway adapter for Linear AgentSession webhooks.

## Configuration

Environment-only setup is supported:

```bash
export LINEAR_WEBHOOK_SECRET=...
export LINEAR_ACCESS_TOKEN=...   # or LINEAR_API_KEY
export LINEAR_HOST=0.0.0.0       # optional
export LINEAR_PORT=8655          # optional
```

Or in `config.yaml`:

```yaml
platforms:
  linear:
    enabled: true
    extra:
      webhook_secret: "..."
      token: "..."
      host: "0.0.0.0"
      port: 8655
```

## Endpoint

Configure Linear to send AgentSessionEvent webhooks to:

```text
POST /linear/agent-sessions
```

The adapter validates `Linear-Signature` as HMAC-SHA256 over the raw request body using `LINEAR_WEBHOOK_SECRET`, uses `Linear-Delivery` for idempotency, acknowledges accepted and duplicate events with HTTP 200, and processes the agent turn in the background.

## Mapping

Each Linear AgentSession maps to one Hermes DM session:

- `chat_id`: `agentSession:{id}`
- `chat_type`: `dm`
- `message_id`: `Linear-Delivery` or the Linear Agent Activity id
- `raw_message`: full webhook payload

Outbound Hermes responses are sent as Linear Agent Activities with content `{type: "response", body: ...}` by default.
