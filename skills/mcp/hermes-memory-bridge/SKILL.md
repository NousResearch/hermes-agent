---
name: hermes-memory-bridge
description: Bidirectional memory bridge — exposes Hermes MEMORY.md and USER.md to any MCP client (Claude Code, Cursor, VS Code, etc.) via a lightweight FastMCP stdio server with FTS5 session search.
version: 1.0.0
author: easyvibecoding
license: MIT
metadata:
  hermes:
    tags: [MCP, Memory, Claude Code, Integration, Bridge]
    related_skills: [native-mcp, claude-code]
---

# Hermes Memory Bridge

Expose Hermes Agent's persistent memory system (MEMORY.md, USER.md) and session history (state.db) to any MCP-compatible client via a lightweight FastMCP stdio server.

## Problem

`hermes mcp serve` exposes conversation browsing tools but does **not** expose memory read/write. External agents like Claude Code cannot access Hermes memory without a custom bridge.

## Solution

A self-contained FastMCP server (`mcp-servers/hermes-memory-mcp.py`) that provides four tools:

| Tool | Description |
|------|-------------|
| `read_memory(store)` | Read MEMORY.md, USER.md, or both |
| `add_memory_entry(store, entry, old_text)` | Append or substring-replace in a memory store |
| `memory_status()` | Char usage and section count for each store |
| `session_search(query, limit, source)` | FTS5 full-text search across past sessions in state.db |

## Prerequisites

- **Python 3.10+**
- **uv** — for automatic dependency management (`brew install uv` on macOS)
- **mcp[cli]** — installed automatically by the runner script via `uv run --with`

## Quick Start

### 1. Register in Hermes config

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-memory:
    command: ~/.hermes/mcp-servers/hermes-memory-runner.sh
    args: []
    enabled: true
```

### 2. Register in Claude Code

Add to `~/.claude.json` (or project-level `.claude/settings.local.json`):

```json
{
  "mcpServers": {
    "hermes-memory": {
      "command": "/bin/bash",
      "args": ["~/.hermes/mcp-servers/hermes-memory-runner.sh"]
    }
  }
}
```

### 3. Verify

In Claude Code:

```
> /mcp
> memory_status()
```

Expected output:
```
MEMORY.md: 892/2,200 chars (40.5%) — 5 sections
USER.md: 461/1,375 chars (33.5%) — 5 sections
```

## Architecture

```
~/.hermes/
├── memories/
│   ├── MEMORY.md          # General memory (2,200 char limit)
│   └── USER.md            # User profile (1,375 char limit)
├── state.db               # SQLite with FTS5 session index
└── mcp-servers/
    ├── hermes-memory-mcp.py      # FastMCP server
    └── hermes-memory-runner.sh   # Shell wrapper for uv
```

### Memory Format

Sections in MEMORY.md and USER.md are separated by `§` (section sign). The `add_memory_entry` tool respects this convention — new entries are appended with `§` separators, and the char limit is enforced before every write.

### Session Search

The `session_search` tool queries `state.db` via FTS5 full-text search. It supports:
- Boolean operators: `AND`, `OR`, `NOT`
- Phrase search: `"exact phrase"`
- Source filtering: `source="claude-code"` to search only Claude Code sessions

The database is opened in **read-only** mode to prevent any accidental modifications.

## Important Notes

### Frozen Snapshot Behavior

Hermes injects memory as a frozen snapshot at session start. If Claude Code writes to memory via this bridge, the changes are persisted to disk immediately but will **not** appear in the current Hermes session's context. They will be available when Hermes starts a new session.

### Dual Registration

Hermes and Claude Code read different config files:
- Hermes: `~/.hermes/config.yaml` → `mcp_servers`
- Claude Code: `~/.claude.json` → `mcpServers`

Register the server in both to enable bidirectional access.

### Security

- `state.db` is opened read-only — session search cannot modify history
- Memory writes enforce char limits — no risk of unbounded growth
- The server runs locally via stdio — no network exposure
