---
id: optional-mcps
kind: object
universe: repo
name: optional-mcps
summary: >-
  Optional MCP catalog integrations, including Graphify and third-party service adapters.
aliases: []
tags: []
shape: object
path: optional-mcps
interface:
  - MCPCatalog
  - MCPServer
  - mcp_serve
depends_on:
  - repo:mcp_serve.py
  - repo:optional-mcps/graphify/manifest.yaml
  - repo:hermes_cli/subcommands/mcp.py
---

# optional-mcps

Optional MCP catalog integrations, including Graphify and third-party service adapters.

## Purpose

Optional MCP catalog integrations, including Graphify and third-party service adapters.

## Inputs

- Repository file tree under `optional-mcps`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:mcp_serve.py`
- `repo:optional-mcps/graphify/manifest.yaml`
- `repo:hermes_cli/subcommands/mcp.py`

## Live Graphify Stats

```json
{
  "by_kind": {},
  "edges": 0,
  "indexed_files": 0,
  "symbols": 0
}
```

