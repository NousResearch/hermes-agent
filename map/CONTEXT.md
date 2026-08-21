# Map Context

Universe scoping and name-collision rules for the ICM system map.

## Universe definition

A universe is a bounded namespace where names are unique. Each card belongs
to exactly one universe. Unqualified names in the map resolve within their
own universe.

### Canonical universes

| Universe | Scope | Example names |
|----------|-------|---------------|
| `repo` | Files, directories, and code symbols in the repo. | `run_agent.py`, `AIAgent` |
| `runtime` | Processes, sessions, and live services. | `gateway`, `session:123` |
| `config` | Settings and feature flags. | `model`, `timeout` |
| `mcp` | MCP servers, tools, and transports. | `graphify`, `stdio` |

## Name collision rules

1. Two cards in the same universe MUST NOT share the same `id`.
2. Two cards in different universes MAY share the same `id`; tools that
   merge across universes MUST qualify with `universe:id`.
3. If a collision is detected, the older card (by filesystem mtime or
   explicit `created_at`) wins unless a `merge` directive is present.

## Qualified references

Use `universe:id` when linking across universes:
- `repo:AGENTS.md`
- `runtime:gateway`
- `mcp:graphify`

Unqualified references resolve within the same universe.
