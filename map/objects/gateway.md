---
id: gateway
kind: object
universe: runtime
name: gateway
summary: >-
  Messaging gateway runner with platform adapters and async turn handling.
aliases: []
tags: []
shape: object
path: gateway
interface:
  - GatewayRunner
  - start_gateway
  - GatewaySession
depends_on:
  - repo:gateway/run.py
  - repo:gateway/session.py
  - repo:agent/async_utils.py
---

# gateway

Messaging gateway runner with platform adapters and async turn handling.

## Purpose

Messaging gateway runner with platform adapters and async turn handling.

## Inputs

- Repository file tree under `gateway`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:gateway/run.py`
- `repo:gateway/session.py`
- `repo:agent/async_utils.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 159,
    "function": 936,
    "method": 1895,
    "module": 91
  },
  "edges": 50832,
  "indexed_files": 91,
  "symbols": 3081
}
```

