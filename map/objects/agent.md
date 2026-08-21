---
id: agent
kind: object
universe: repo
name: agent
summary: >-
  Core agent runtime with conversation orchestration, memory, caching, and provider adapters.
aliases: []
tags: []
shape: object
path: agent
interface:
  - AIAgent
  - run_conversation
  - chat
  - AgentLoop
depends_on:
  - repo:run_agent.py
  - repo:model_tools.py
  - repo:agent/graphify.py
---

# agent

Core agent runtime with conversation orchestration, memory, caching, and provider adapters.

## Purpose

Core agent runtime with conversation orchestration, memory, caching, and provider adapters.

## Inputs

- Repository file tree under `agent`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:run_agent.py`
- `repo:model_tools.py`
- `repo:agent/graphify.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 249,
    "function": 2722,
    "method": 944,
    "module": 196
  },
  "edges": 44991,
  "indexed_files": 196,
  "symbols": 4111
}
```

