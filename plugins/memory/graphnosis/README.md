# Graphnosis Memory Provider for Hermes Agent

Local encrypted memory via the Graphnosis desktop app. No API key required.

## Prerequisites

1. Install [Graphnosis](https://graphnosis.com/download)
2. Unlock your cortex (Graphnosis must be running)
3. MCP socket at `~/.graphnosis/mcp.sock`

## Quick setup

```bash
hermes memory setup          # select "graphnosis"
hermes mcp install graphnosis  # recommended — full MCP tool surface
hermes graphnosis status
```

Or set manually in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: graphnosis

mcp_servers:
  graphnosis:
    command: npx
    args: ["-y", "@graphnosis/mcp-relay", "${HOME}/.graphnosis/mcp.sock"]
    enabled: true
```

Config file: `$HERMES_HOME/graphnosis.json`

| Key | Default | Description |
|-----|---------|-------------|
| `socket_path` | `~/.graphnosis/mcp.sock` | MCP socket path |
| `default_engram` | `""` | Default engram for remember |
| `prefetch_max_tokens` | `1500` | Prefetch token budget |

## Tools (memory provider)

| Tool | Purpose |
|------|---------|
| `graphnosis_recall` | Semantic memory search |
| `graphnosis_remember` | Save durable notes |
| `graphnosis_stats` | Engram / capacity overview |

## MCP catalog (recommended)

Install for full tools: `edit`, `forget`, `cross_search`, skills, consent.

```bash
hermes mcp install graphnosis
```

Tools appear as `mcp_graphnosis_*` in Hermes sessions.

## CLI

```bash
hermes graphnosis status
hermes graphnosis test-recall "project priorities"
```

Only available when `memory.provider: graphnosis`.
