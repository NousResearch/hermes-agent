# Hermes Docs MCP Server

**Search, navigate, and read the complete Hermes Agent documentation — from within any MCP client, including Hermes itself.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

No more switching tabs. No more stale knowledge. Add one line to your `~/.hermes/config.yaml` and your agent can search its own docs in two tool calls.

## Quick Start

```bash
# Install
pip install -e .

# Build the doc index
python -m server.hermes_docs_server rebuild

# Add to Hermes config (~/.hermes/config.yaml)
mcp_servers:
  hermes-docs:
    command: "python"
    args: ["-m", "server.hermes_docs_server"]
    tools: true
    prompts: false
    resources: false
```

Restart Hermes and ask: *"Search the docs for how MCP auth works."*

## Why?

Hermes Agent's documentation lives at [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs) — a well-structured Docusaurus site. But the *agent* can't natively search it. You either:

1. Open a browser tab and search manually
2. Ask the agent (which uses its training data — may be stale or imprecise)

**Hermes Docs MCP Server** fixes this. It indexes the docs into a local SQLite FTS5 database and exposes structured tools. The agent fetches *current* documentation, with snippets and full page content, every time.

## Tools

| Tool | Description |
|------|-------------|
| `search_docs(query, limit=10)` | FTS5 full-text search across all documentation |
| `read_page(slug)` | Read a full doc page by URL slug |
| `list_sections()` | Browse documentation structure |
| `list_pages(section)` | List pages within a section |
| `doc_stats()` | Index statistics |

## Requirements

- Python 3.10+
- [MCP SDK](https://github.com/modelcontextprotocol/python-sdk) (`pip install mcp`)
- `httpx` (for index building — `pip install httpx`)

## Architecture

```
hermes-docs-mcp-server/
├── server/
│   └── hermes_docs_server.py    # MCP server (FastMCP-compatible) + SQLite FTS5 index
├── scripts/
│   └── fetch_hermes_docs.py     # Doc indexer: scrapes live site OR reads local repo
├── SKILL.md                      # Hermes optional-skill definition
├── README.md
└── pyproject.toml
```

## Index Management

- **Live index**: `python -m server.hermes_docs_server rebuild` — scrapes the Hermes docs site
- **From repo**: `python -m scripts.fetch_hermes_docs --repo /path/to/hermes-agent` — builds from a local checkout
- **Custom path**: set `HERMES_DOCS_INDEX_PATH` env var

The index uses **sqlite3 FTS5** (stdlib) — zero external search infrastructure.

## License

MIT — same as Hermes Agent. The indexed documentation is from the Hermes Agent repo (MIT).

## Author

[Joerg Peetz](https://github.com/JPeetz) — part of the Hermes Agent growth and contribution initiative.
