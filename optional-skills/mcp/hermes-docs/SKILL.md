---
name: hermes-docs
description: Search, navigate, and read Hermes Agent docs via MCP.
version: 1.0.0
author: Joerg Peetz (JPeetz)
license: MIT
platforms:
  - linux
  - macos
  - windows
metadata:
  hermes: true
tags:
  - MCP
  - Documentation
  - Search
  - Hermes
  - Self-Documenting
  - Developer-Experience
related_skills:
  - fastmcp
  - mcporter
  - mcp-oauth-remote-gateway
---

# Hermes Docs MCP Server

Let Hermes search, navigate, and read its own documentation directly — no browser, no website tab, no guessing whether a flag still works.

The `hermes-docs` MCP server indexes the complete Hermes Agent documentation and exposes it as structured MCP tools. Add it to your `~/.hermes/config.yaml` and your agent can find any doc page in two tool calls: `search_docs` to discover, `read_page` to get the full content.

## When to Use

- You want to ask Hermes _"how does X work?"_ and get an answer from the *current* docs — not stale training data
- You need to find a specific config flag, CLI command, or feature guide quickly
- You're writing a skill or PR and need the exact doc reference
- You want the agent to self-document: _"Show me the docs for MCP configuration"_ → tool call → returns the official guide

Keywords: "docs", "documentation", "how do I", "what does X do", "find in docs", "search docs"

## Prerequisites

```bash
# Install the MCP server (from the hermes-docs-mcp-server directory)
pip install -e .

# Or directly from the repo
pip install git+https://github.com/JPeetz/hermes-docs-mcp-server.git
```

## Setup

1. Install the server and build the doc index:

```bash
# First-time index build
python -m server.hermes_docs_server rebuild
```

2. Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  hermes-docs:
    command: "python"
    args:
      - "-m"
      - "server.hermes_docs_server"
    tools: true
    prompts: false
    resources: false
```

3. Restart Hermes and try:

```
Search the Hermes docs for how to configure MCP servers
```

## Tools

### `search_docs(query: str, limit: int = 10)`

Full-text search across all documentation pages. Uses SQLite FTS5 with Porter stemming — `configure`, `configuring`, `configuration` all match.

Returns matching pages with title, section, and a relevant snippet.

**Examples:**
- `search_docs("MCP OAuth configuration")` — find all pages about MCP OAuth
- `search_docs("profile isolation")` — find profile-related docs
- `search_docs("cron schedule")` — find cron/scheduling docs

### `read_page(slug: str)`

Read the complete content of a documentation page by its URL slug. Use search first to find the slug, then read the page.

**Slug examples:**
- `"user-guide/features/mcp"` → the MCP feature guide
- `"user-guide/configuration"` → configuration reference
- `"getting-started/installation"` → installation guide

### `list_sections()`

List all documentation sections with page counts — useful for browsing structure:

```json
{
  "sections": [
    {"section": "Features", "page_count": 8},
    {"section": "Getting Started", "page_count": 3},
    {"section": "Guides", "page_count": 5},
    {"section": "Reference", "page_count": 4},
    {"section": "User Guide", "page_count": 14}
  ]
}
```

### `list_pages(section: str)`

List all pages within a documentation section. Returns slug and title for each page.

### `doc_stats()`

Get index statistics: total pages, characters indexed, and metadata.

## Index Management

The index is SQLite-backed (FTS5) and stored at `./hermes_docs_index.sqlite` by default:

```bash
# Rebuild the index from the live website
python -m server.hermes_docs_server rebuild

# Use a custom index path
HERMES_DOCS_INDEX_PATH=/path/to/index.sqlite python -m server.hermes_docs_server
```

The index auto-updates on rebuild. No external services, no API keys, no network required after indexing.

## Architecture

```
hermes-docs-mcp-server/
├── server/
│   └── hermes_docs_server.py    # MCP server + SQLite FTS5 index
├── scripts/
│   └── fetch_hermes_docs.py     # Doc scraper (live site or local repo)
├── SKILL.md                      # ← This file (Hermes skill definition)
├── README.md                     # Project README
└── pyproject.toml                # Python package
```

The server uses **sqlite3 FTS5** (built into Python's stdlib) for search — zero additional dependencies. The indexer optionally uses `httpx` for web scraping.

## License

MIT — same as Hermes Agent.
