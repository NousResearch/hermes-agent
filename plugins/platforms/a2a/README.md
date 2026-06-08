# A2A — Agent-to-Agent protocol for Hermes

Talk to other agents, and let other agents talk to you, over the open
[A2A protocol](https://a2a-protocol.org). Works with any A2A-compliant peer
(another Hermes, LangChain, CrewAI, Google ADK, OpenClaw, …). Stdlib only — no
`a2a-sdk` dependency.

## Enable

```bash
hermes gateway setup      # pick A2A, or:
```

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900

# peers you want to call (outbound):
a2a_agents:
  researcher:
    url: "http://localhost:9999"
    auth: { type: bearer, token: "sk-..." }
    timeout: 120
```

## Outbound — call other agents

The agent gets three tools:

- `a2a_discover(url)` — what can this agent do?
- `a2a_call(agent, message, context_id?)` — send it a task, get the reply.
- `a2a_list()` — configured peers + saved conversations.

## Inbound — be callable

When the `a2a` platform is enabled, Hermes serves an Agent Card at
`http://<host>:<port>/.well-known/agent.json` and accepts JSON-RPC
`message/send` tasks. Incoming tasks are injected into your **live** agent
session — the same agent that's talking to you, with full memory — and the
reply is returned over A2A.

## Security

- **No bearer token ⇒ localhost only.** The server binds `127.0.0.1` and
  refuses to widen unless you set both `A2A_BEARER_TOKEN` and `A2A_HOST`.
- Inbound text is run through prompt-injection filters and framed as untrusted
  peer input.
- Outbound text is scrubbed of credential-shaped strings.
- Every exchange is logged to `~/.hermes/a2a_audit.jsonl`.
- Conversations persist to `~/.hermes/a2a_conversations/` — they survive context
  compaction and restarts.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `A2A_BEARER_TOKEN` | _(unset)_ | Required on inbound calls. Unset ⇒ localhost-only. |
| `A2A_HOST` | `127.0.0.1` | Bind host. Only widens with a token set. |
| `A2A_PORT` | `9900` | Inbound port. |
| `A2A_AGENT_NAME` | hostname-derived | Name on the Agent Card. |
| `A2A_ALLOW_ALL_USERS` | `false` | Allow any authed peer (dev only). |

See `DESIGN.md` for architecture and the requirement-tracing table.
