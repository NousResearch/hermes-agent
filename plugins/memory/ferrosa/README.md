# Ferrosa Memory Provider Plugin

Hermes MemoryProvider plugin that connects to a running ferrosa-memory MCP server.

## Overview

This plugin provides persistent memory for Hermes Agent backed by ferrosa-memory,
a semantic knowledge graph with:

- **Entity-based memory** — facts, decisions, preferences stored as typed entities
- **Temporal facts** — auto-superseded history tracking entity states over time
- **Skill catalog** — global reusable methodologies (TDD, debugging, refactoring, etc.)
- **Intention system** — deferred actions triggered by context (prospective memory)
- **Consolidation** — automatic dream/consolidation that discovers hidden connections

## Architecture

The plugin:

1. Reads its MCP endpoint URL from `FERROSA_MEMORY_URL` env var or saved config
2. Provides `prefetch()` via `hybrid_search` for semantic recall before each turn
3. Mirrors `sync_turn()` and `on_memory_write()` via `smart_ingest`
4. Runs `run_consolidation` on `on_session_end`

It does NOT re-expose fmem tools — those are already available directly as MCP tools
under the `fmem_` prefixes (e.g. `fmem_smart_ingest`, `fmem_hybrid_search`).

## Configuration

```bash
hermes config set memory.provider ferrosa
hermes config set plugins.ferrosa.url "http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp"
```

Or via the `hermes memory setup` wizard.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FERROSA_MEMORY_URL` | Full MCP HTTP endpoint with credentials | `http://ferrosa_user:ferrosa_user@127.0.0.1:18765/mcp` |
| `FERROSA_MEMORY_TENANT_ID` | Tenant override for multi-tenant deployments | (auto-detected) |

## Files

- `__init__.py` — `MemoryProvider` implementation
- `plugin.yaml` — plugin metadata
- `README.md` — this file
