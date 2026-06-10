# GBrain memory provider

The GBrain memory provider lets Hermes use a GBrain MCP HTTP endpoint as an external memory provider. It is designed for the single-writer GBrain governance pattern:

- Default/non-steward Hermes profiles use a read-only GBrain MCP proxy.
- Athena/steward profile is the only profile that may be configured for writes.
- Read-only mode is the default and refuses direct writes.

## Setup

Run:

```bash
hermes memory setup gbrain
```

Or configure manually in the active profile's `config.yaml`:

```yaml
memory:
  provider: gbrain
  gbrain:
    endpoint: "http://127.0.0.1:3132/mcp"
    mode: "read-only"
    source_id: "__all__"
    max_results: 6
    timeout: 5.0
```

The setup wizard also writes non-secret provider settings to profile-local `$HERMES_HOME/gbrain-memory.json`. Do not put bearer tokens, OAuth secrets, database URLs, SSH material, or other credentials in this file.

## Transport

This provider uses stdlib HTTP JSON-RPC against GBrain's MCP endpoint. It does not call the local `gbrain` CLI or connect directly to the GBrain database. MCP keeps Hermes profile policy separate from GBrain internals and matches the existing local topology where non-Athena profiles use a read-only proxy such as:

```yaml
memory:
  provider: gbrain
  gbrain:
    endpoint: "http://127.0.0.1:3132/mcp"
    mode: "read-only"
```

## Modes

### `read-only` (default)

Allowed:

- Search/query GBrain via the configured MCP endpoint.
- Inject compact recalled context into the next turn.
- Use the `gbrain_memory_search` tool.

Blocked:

- Automatic turn ingestion.
- Mirroring built-in memory writes.
- Creating, updating, ingesting, or deleting GBrain pages.

The `gbrain_memory_store_candidate` tool returns a blocked response with the candidate payload so the operator can route it to Athena/steward review.

### `read-write` (explicit opt-in)

Intended only for Athena/steward profile or another profile with explicit user authorization.

```yaml
memory:
  provider: gbrain
  gbrain:
    endpoint: "http://127.0.0.1:3131/mcp"
    mode: "read-write"
    write_tool: "create_page"
```

In read-write mode, the provider still does not auto-ingest every conversation turn. It only mirrors explicit built-in memory writes and explicit `gbrain_memory_store_candidate` calls to the configured `write_tool`.

## Tools

- `gbrain_memory_search(query, limit=6)`: queries GBrain through MCP and returns formatted recalled context plus the raw MCP result.
- `gbrain_memory_store_candidate(content, target="memory")`: in read-only mode, returns `blocked: true` with Athena handoff guidance; in read-write mode, calls the configured write tool.

## Configuration reference

| Key | Default | Notes |
| --- | --- | --- |
| `endpoint` | `http://127.0.0.1:3132/mcp` | HTTP MCP endpoint. Use the read-only proxy for non-Athena profiles. |
| `mode` | `read-only` | `read-only` or explicit `read-write`. |
| `source_id` | `__all__` | GBrain source scope passed to the query tool. |
| `max_results` | `6` | Clamped to 1–20. |
| `timeout` | `5.0` | HTTP timeout seconds, clamped to 0.5–30. |
| `query_tool` | `query` | Underlying GBrain MCP query tool name. |
| `query_tool_fallbacks` | `gbrain_query, search` | Tried if `query` is unavailable. |
| `write_tool` | `create_page` | Used only in explicit `read-write` mode. |

## Safety notes

- Keep GBrain credentials out of docs, tests, logs, and provider config.
- Prefer `http://127.0.0.1:3132/mcp` read-only proxy for default profiles.
- Use Athena/steward profile for durable write curation.
- Treat read-write mode as a privileged local configuration, not a normal default.
