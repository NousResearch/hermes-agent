---
name: error-knowledge-mcp
description: "Cross-project error pattern knowledge base MCP server. Records, searches, and archives bug patterns so agents learn from past mistakes."
version: 1.0.0
author: Hermes Agent Community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [mcp, knowledge, errors, debugging, learning]
    related_skills: [native-mcp, qmd]
---

# error-knowledge-mcp

A local MCP server that maintains a cross-project error pattern knowledge
base. Every time a bug is diagnosed and fixed, the agent can record the root
cause, reproduction steps, and fix summary. Subsequent sessions query this
knowledge base to avoid repeating the same mistakes.

## How It Works

Records are stored as flat markdown files with YAML frontmatter:

```
~/.hermes/knowledge/errors/
├── _index.json                          ← cached index (auto-rebuilt)
├── generic/
│   ├── csharp/
│   │   └── nullref-ef-core-lazy-loading.md
│   └── python/
│       └── nonetype-or-greater.md
└── business-specific/
    ├── my-project/
    │   └── pagination-off-by-one.md
    └── another-project/
        └── timeout-too-low.md
```

Two scopes keep knowledge organised:

| Scope | Purpose | Directory |
|-------|---------|-----------|
| `generic` | Language-level patterns reusable across projects | `generic/<lang>/` |
| `business-specific` | Project-local logic errors | `business-specific/<project>/` |

## Tools

| Tool | Description |
|------|-------------|
| `search_error_patterns` | BM25 search over stored records. Filter by keywords, category, language, project, or scope. |
| `record_error_pattern` | Save a new error record. Auto-deduplicates (same scope + same lang + similar title → skipped). |
| `knowledge_stats` | View total count, breakdown by scope, language, and project. |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ERROR_KNOWLEDGE_ROOT` | `~/.hermes/knowledge/errors/` | Storage directory for error records |
| `ERROR_KNOWLEDGE_AUTO_ARCHIVE` | `5000` | Auto-archive threshold (move flat files into subdirectories) |

## Registration

Add to your Hermes config's `mcp_servers` section:

```yaml
mcp_servers:
  error-knowledge:
    command: python
    args:
      - path/to/error-knowledge-mcp/scripts/server.py
    timeout: 30
```

## Deduplication

When recording, the server checks for an existing record with the same
`scope` + `lang` + a title that contains or is contained by the new title.
If found, the new record is skipped with a `"reason": "duplicate"` response.

## Auto-Archive

When the total file count reaches `AUTO_THRESHOLD` (default 5000), flat
files at the root level are automatically migrated into their scope-specific
subdirectories.
