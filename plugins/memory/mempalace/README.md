# MemPalace Memory Provider

Native [MemPalace](https://github.com/MemPalace/mempalace) integration for Hermes Agent — replaces the MCP bridge with direct Python API calls for lower latency and richer integration.

## Features

- **Semantic search** over the palace graph (L0 closets + L1 memories)
- **Knowledge Graph** queries and updates
- **Automatic diary journaling** at session end
- **Turn archiving** — conversations automatically mined into the palace
- **Background prefetch** — pre-warms recall for the next turn
- **Tool exposure** — 5 tools available to the model

## Tools

| Tool | Description |
|------|-------------|
| `mempalace_search` | Semantic search across all palace drawers |
| `mempalace_kg_query` | Query structured facts in the Knowledge Graph |
| `mempalace_kg_add` | Add or invalidate KG facts |
| `mempalace_diary_read` | Read recent diary entries |
| `mempalace_diary_write` | Write diary entries (AAAK format) |

## Setup

1. **Install MemPalace** (if not already):
   ```bash
   pip install mempalace
   ```

2. **Initialize MemPalace** (first time only):
   ```bash
   mempalace init --palace /root/.mempalace/palace
   ```

3. **Configure Hermes**:
   ```bash
   # Option A: env var
   export MEMPALACE_PATH=/root/.mempalace/palace

   # Option B: set in Hermes
   hermes memory setup
   # Select "mempalace" when prompted
   ```

4. **Activate**:
   ```bash
   hermes config set memory.provider mempalace
   ```
   Then start a new session (`/reset` or restart the gateway).

## Configuration

| Config key | Default | Description |
|-----------|---------|-------------|
| `palace_path` | `/root/.mempalace/palace` | Path to the palace directory |
| `agent_name` | `jarvis` | Agent identifier for diary |

Set via `hermes memory setup` or directly in `config.yaml`:
```yaml
memory:
  provider: mempalace
  mempalace:
    palace_path: /root/.mempalace/palace
    agent_name: jarvis
```

## How It Works

- `prefetch()` is called before each LLM API call, injecting semantic recall context
- `queue_prefetch()` pre-warms the next turn's recall in the background
- `sync_turn()` archives completed turns to the palace in real time
- `on_session_end()` writes a diary entry summarizing the session
- `on_memory_write()` mirrors built-in memory writes to MemPalace

## Architecture

```
plugins/memory/mempalace/
├── __init__.py      # MemPalaceMemoryProvider + register()
├── plugin.yaml      # Metadata
└── README.md        # This file
```

## Requirements

- Python 3.10+
- mempalace >= 3.0
- chromadb (installed by mempalace)

## Upstream

This plugin is part of the MemPalace native integration for Hermes Agent.
Contributions welcome: https://github.com/NousResearch/hermes-agent