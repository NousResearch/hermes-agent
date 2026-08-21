# Hermes ICM System Map

Stable catalog for the Inter-Context Map (ICM). This directory stores
lightweight, agent-readable cards describing objects and processes in the
Hermes repo and runtime.

## Where things live

| Path | Purpose |
|------|---------|
| `map/_meta/schema.md` | Canonical card types and fields. Read this before writing any card. |
| `map/_templates/` | Instance templates: `object.md`, `process.md`. Copy, fill, and extend. |
| `map/CONTEXT.md` | Universe definitions and name-collision rules. |

## How to walk

1. Open `map/CLAUDE.md` (this file) to orient.
2. Read `map/_meta/schema.md` for the card grammar.
3. Open the template you need from `map/_templates/`.
4. Check `map/CONTEXT.md` if you are unsure which universe a name belongs to or whether a collision exists.

A cold agent can navigate the entire map from this entry plus at most two more reads.

## Integration points

- Graphify persistence: Graphify nodes can reference ICM cards by `id`.
- MCP surfaces: MCP tools may expose card metadata through `graphify`-style queries.
