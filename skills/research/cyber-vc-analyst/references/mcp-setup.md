# Return On Security MCP Setup

This skill expects Return on Security to be configured as an MCP server in the
active runtime.

In Hermes config, prefer the alias `returnonsecurity`, which usually yields the
tool prefix:

```text
mcp_returnonsecurity_
```

In Codex CLI, a server added as `signal-mcp` will usually yield the tool
prefix:

```text
mcp_signal_mcp_
```

The skill accepts either prefix and should inspect the live tool inventory
rather than assuming one exact alias.

## Hermes Config Example

Add this to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  returnonsecurity:
    url: "https://signal.returnonsecurity.com"
    headers:
      Authorization: "Bearer <token-if-required>"
```

If the service uses a different auth scheme, keep the alias the same and update
only the header values.

## Codex CLI Example

To add and authenticate the same server in Codex:

```bash
codex mcp add signal-mcp --url https://mcp.returnonsecurity.com/mcp
codex mcp login signal-mcp
```

If Safari approves the OAuth request but fails on the localhost callback, copy
the full `http://127.0.0.1:<port>/callback/...` URL immediately and replay it
while the Codex listener is still running.

## Notes

- Do not reload MCP mid-analysis unless the user explicitly asks. Reloading MCP
  changes the tool list and can invalidate prompt caching.
- If the config was just changed, restart Hermes or run `/reload-mcp` in a
  separate explicit step before using this skill.
- In Codex, newly authenticated MCP tools may not appear in the current agent
  turn. Restart the session if auth succeeded but the tool prefix is still
  absent from the live tool list.
- The skill should depend only on the `mcp_returnonsecurity_*` or
  `mcp_signal_mcp_*` prefixes, not on a specific tool name, because the
  server's exact tool inventory may evolve.

## Expected Usage

When the tools are present, use them for:

- category and market structure checks
- competitor and adjacent vendor discovery
- trend and thematic validation
- likely buyer and acquirer landscape
- adjacent architecture shifts in cyber markets
