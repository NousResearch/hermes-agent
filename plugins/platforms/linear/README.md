# Linear Agent Sessions platform plugin

This plugin adds a native Hermes gateway adapter for Linear AgentSession webhooks.

## Configuration

Environment-only setup is supported:

```bash
export LINEAR_WEBHOOK_SECRET=...
export LINEAR_ACCESS_TOKEN=...   # or LINEAR_API_KEY
export LINEAR_HOST=0.0.0.0       # optional
export LINEAR_PORT=8655          # optional
export LINEAR_ALLOW_ALL_USERS=true  # optional, or use LINEAR_ALLOWED_USERS
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

Linear uses the gateway's normal user authorization layer. Set
`LINEAR_ALLOWED_USERS` to a comma-separated allowlist, set
`LINEAR_ALLOW_ALL_USERS=true`, or approve users through the normal pairing
flow.

## Endpoint

Configure Linear to send AgentSessionEvent webhooks to:

```text
POST /linear/agent-sessions
```

The adapter validates `Linear-Signature` as HMAC-SHA256 over the raw request body using `LINEAR_WEBHOOK_SECRET`, uses `Linear-Delivery` for idempotency, acknowledges accepted and duplicate events with HTTP 200, and processes the agent turn in the background.

For a local gateway, expose the endpoint through a durable public HTTPS URL
such as Tailscale Funnel or a hosted reverse proxy. Configure this URL in the
Linear Agent/OAuth app settings, not the generic workspace webhook API:

```text
https://<public-host>/linear/agent-sessions
```

The generic Linear workspace webhook API accepts resources such as issues and
comments, but it does not accept AgentSession events. Do not rely on a fallback
poller or replay script for normal operation; those can hide broken app
delivery and should be treated as short-lived rescue tools only.

## Mapping

Each Linear AgentSession maps to one Hermes DM session:

- `chat_id`: `agentSession:{id}`
- `chat_type`: `dm`
- `message_id`: `Linear-Delivery` or the Linear Agent Activity id
- `raw_message`: full webhook payload

Outbound Hermes responses are sent as Linear Agent Activities with content `{type: "response", body: ...}` by default.

## Agent activities and plans

The adapter uses Linear's Agent Session primitives directly:

- On `created`, it emits a `thought` activity quickly so Linear does not mark the session unresponsive.
- When Hermes starts work, it emits an ephemeral `action` activity and updates the Agent Session `plan` with one `inProgress` step summarizing the request.
- When Hermes completes or is stopped, it updates the plan step to `completed` or `canceled`.
- When Hermes' `clarify` tool asks the user for feedback or a decision, the adapter emits an `elicitation` activity and updates the plan to show that it is waiting for a user response. Multiple-choice questions use Linear's `select` signal with `signalMetadata.options`.
- Dangerous-command approvals also use an `elicitation` activity with `select` options. The option values are Hermes' normal `/approve`, `/approve session`, `/approve always`, and `/deny` commands, so Linear reuses the same approval resolver as every other gateway platform.

Linear Agent Session plans must be replaced as a full array on each update, so the adapter keeps plans intentionally short. It does not mirror every internal tool call into the plan; detailed progress still belongs in Hermes' normal activity stream and final response.

Selected options and free-form replies arrive as normal Linear `prompted` events. Hermes captures the next clarify reply through its existing text-intercept and resumes the blocked agent turn. Slash-command control replies are passed through without appended issue context so approval and stop commands stay parseable.

## Active issue work

Linear's issue list and board surfaces treat active agent work as issue-linked,
session-scoped state. When Hermes begins processing an AgentSession that is
attached to an issue, the adapter follows Linear's agent best-practice guidance:

- if the issue is not already in a `started`, `completed`, or `canceled`
  workflow state, move it to the team's first `started` state by lowest
  position;
- if the issue has no `Issue.delegate`, set the Linear app user as the
  delegate;
- never overwrite an existing delegate.

The visible "active agent" indicator is transient: it is expected only while the
Linear AgentSession is active or awaiting input. Once Hermes sends its final
response and marks the AgentSession complete, the issue can remain delegated to
Hermes and in progress without showing as actively running.
