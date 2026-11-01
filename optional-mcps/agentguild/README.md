# Agent Guild

Use Agent Guild when deciding whether to try a public remote agent or MCP
provider and you want a fresh endpoint observation before delegation.

Install with `hermes mcp install agentguild`, retain the two preselected
tools, and start a new Hermes session. Both work without an account or key:

- `guild_preflight(url=...)`: inspect an explicitly chosen public endpoint.
- `guild_paid_operations()`: read current prices and entrypoints for paid
  operations. The price lookup itself is free.

If this entry is not yet in your installed catalog, merge the following
server entry into the existing `mcp_servers` block in your Hermes config:

```yaml
mcp_servers:
  agentguild:
    url: https://agent-guild-5d5r.onrender.com/mcp
    tools:
      include:
        - guild_preflight
        - guild_paid_operations
```

For example, ask Hermes: “Use Agent Guild to inspect this public provider URL
before I delegate. Summarize the observations and unknowns without invoking
the provider's tools.” Supply the actual public URL separately.

Preflight sends that URL to Agent Guild. Do not provide internal URLs or
URLs containing credentials. Its unsigned observations are not proof of
the provider's identity, authorization, task quality, or safety. This is
an ordinary tool integration; it does not intercept other tool calls.

Review tool selection with `hermes mcp configure agentguild`. The service
also exposes paid tools, which are not preselected. Paid operations require
separate authorization and payment; `GET /check` is not a free lookup.

Source and documentation: [Agent Guild](https://github.com/AgentTanuki/agent-guild).
