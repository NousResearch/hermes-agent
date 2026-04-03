# Cyber Memory Memory Provider

Cyber Memory adds first-class Hermes memory support backed by the
[`cyber-memory`](https://github.com/RamboRogers/cyber-memory) single-binary
MCP server.

## Requirements

- `cyber-memory` binary installed
- Python MCP client available (the Hermes memory setup flow installs `mcp`)

## Setup

```bash
hermes memory setup    # select "cyber_memory"
```

Or manually:

```bash
hermes config set memory.provider cyber_memory
```

The provider stores its config at:

```text
$HERMES_HOME/cyber-memory.json
```

Default profile-scoped database path:

```text
$HERMES_HOME/cyber-memory/db.sqlite3
```

The provider launches `cyber-memory` over stdio MCP internally and sets
`CYBER_MEMORY_DB` so each Hermes profile gets isolated storage.

## Config

| Key | Default | Description |
|-----|---------|-------------|
| `command` | `cyber-memory` | Binary name or absolute path |
| `db_path` | `$HERMES_HOME/cyber-memory/db.sqlite3` | Profile-scoped SQLite database |

## Tools

| Tool | Description |
|------|-------------|
| `cyber_memory_store` | Store a new memory |
| `cyber_memory_recall` | Semantic recall with ranking |
| `cyber_memory_search` | Full-text search |
| `cyber_memory_relate` | Create a graph relation |
| `cyber_memory_graph` | Traverse connected memories |
| `cyber_memory_update` | Update content/tags/importance |
| `cyber_memory_forget` | Delete a memory |
| `cyber_memory_stats` | Show database stats |

## Troubleshooting

- **Provider unavailable** — ensure `cyber-memory -version` works and Hermes has
  the Python `mcp` package available.
- **Wrong database location** — set `db_path` explicitly via `hermes memory setup`
  or edit `$HERMES_HOME/cyber-memory.json`.
- **First run is slow** — Cyber Memory may download its embedded model assets on
  first use.
