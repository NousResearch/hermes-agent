---
id: hermes-cli
kind: object
universe: repo
name: hermes_cli
summary: >-
  CLI subcommands, setup wizard, skin engine, and slash-command registry.
aliases: []
tags: []
shape: object
path: hermes_cli
interface:
  - HermesCLI
  - COMMAND_REGISTRY
  - setup_wizard
  - skin_engine
depends_on:
  - repo:cli.py
  - repo:hermes_cli/commands.py
  - repo:hermes_cli/skin_engine.py
---

# hermes_cli

CLI subcommands, setup wizard, skin engine, and slash-command registry.

## Purpose

CLI subcommands, setup wizard, skin engine, and slash-command registry.

## Inputs

- Repository file tree under `hermes_cli`
- Python source files for Graphify symbol and edge extraction
- Module docstrings and AST definitions for live indexing

## Outputs

- Structured symbol table: classes, functions, methods, modules
- Edges: IMPORTS, CONTAINS, CALLS, REFERENCES
- JSON graph payload written to `graphify-out/graph.json`

## Dependencies

- `repo:cli.py`
- `repo:hermes_cli/commands.py`
- `repo:hermes_cli/skin_engine.py`

## Live Graphify Stats

```json
{
  "by_kind": {
    "class": 329,
    "function": 4985,
    "method": 688,
    "module": 285
  },
  "edges": 78758,
  "indexed_files": 285,
  "symbols": 6287
}
```

