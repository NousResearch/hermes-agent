# jMunch MCP Suite — Optional Skill for Hermes Agent

Token-efficient retrieval for **code**, **documentation**, and **tabular data** via three MCP servers.

## What's Included

| Server | PyPI Package | Tools | Domain |
|--------|-------------|-------|--------|
| jCodeMunch | `jcodemunch-mcp` | 52 | Code intelligence (70+ languages) |
| jDocMunch | `jdocmunch-mcp` | 13 | Documentation retrieval |
| jDataMunch | `jdatamunch-mcp` | 18 | Tabular data (CSV/Excel/Parquet) |

## Quick Setup

1. **Install** (pick what you need):

```bash
pip install jcodemunch-mcp jdocmunch-mcp jdatamunch-mcp
```

2. **Configure** `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  jcodemunch:
    command: "uvx"
    args: ["jcodemunch-mcp"]
  jdocmunch:
    command: "uvx"
    args: ["jdocmunch-mcp"]
  jdatamunch:
    command: "uvx"
    args: ["jdatamunch-mcp"]
```

3. **Restart Hermes** and start using the tools.

## Benchmarks

- **37x** fewer tokens vs raw file reading
- **19.6x** fewer tool calls
- **12.4 billion** tokens saved across all users

## Links

- [jCodeMunch](https://github.com/jgravelle/jcodemunch-mcp) | [jDocMunch](https://github.com/jgravelle/jdocmunch-mcp) | [jDataMunch](https://github.com/jgravelle/jdatamunch-mcp)
- [jMRI Specification](https://github.com/jgravelle/mcp-retrieval-spec) (Apache 2.0)
