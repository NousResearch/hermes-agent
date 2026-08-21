---
id: src-graphify
kind: object
universe: repo
name: src-graphify
summary: >-
  Live Graphify index source: graph model, query engine, and JSON persistence.
aliases: []
tags: []
shape: object
path: src/graphify
interface:
  - GraphModel
  - GraphJsonRepository
  - GraphifyQueryEngine
  - build_graph
depends_on:
  - repo:src/graphify/model.py
  - repo:src/graphify/query.py
  - repo:agent/graphify.py
---

# src-graphify

Live Graphify index source: graph model, query engine, and JSON persistence.

## Purpose

Live Graphify index source: graph model, query engine, and JSON persistence.

## Inputs

- Repository file tree under `src/graphify`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:src/graphify/model.py`
- `repo:src/graphify/query.py`
- `repo:agent/graphify.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 6,
    "method": 25,
    "module": 3
  },
  "edges": 274,
  "indexed_files": 3,
  "symbols": 34
}
```

