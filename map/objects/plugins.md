---
id: plugins
kind: object
universe: repo
name: plugins
summary: >-
  Plugin system for memory providers, web search backends, observability, and extended capabilities.
aliases: []
tags: []
shape: object
path: plugins
interface:
  - PluginRegistry
  - MemoryProvider
  - WebSearchProvider
  - ContextEngine
depends_on:
  - repo:plugins/__init__.py
  - repo:tools/registry.py
  - repo:hermes_cli/commands.py
---

# plugins

Plugin system for memory providers, web search backends, observability, and extended capabilities.

## Purpose

Plugin system for memory providers, web search backends, observability, and extended capabilities.

## Inputs

- Repository file tree under `plugins`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:plugins/__init__.py`
- `repo:tools/registry.py`
- `repo:hermes_cli/commands.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 250,
    "function": 1730,
    "method": 2643,
    "module": 204
  },
  "edges": 68526,
  "indexed_files": 204,
  "symbols": 4827
}
```

