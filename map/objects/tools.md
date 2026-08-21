---
id: tools
kind: object
universe: repo
name: tools
summary: >-
  Built-in tool implementations and central discovery registry for model-callable tools.
aliases: []
tags: []
shape: object
path: tools
interface:
  - ToolRegistry
  - discover_builtin_tools
  - get_definitions
  - dispatch
depends_on:
  - repo:tools/registry.py
  - repo:toolsets.py
  - repo:model_tools.py
---

# tools

Built-in tool implementations and central discovery registry for model-callable tools.

## Purpose

Built-in tool implementations and central discovery registry for model-callable tools.

## Inputs

- Repository file tree under `tools`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:tools/registry.py`
- `repo:toolsets.py`
- `repo:model_tools.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 169,
    "function": 2749,
    "method": 1008,
    "module": 148
  },
  "edges": 44037,
  "indexed_files": 148,
  "symbols": 4074
}
```

