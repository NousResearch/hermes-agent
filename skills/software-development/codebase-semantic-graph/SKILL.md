---
name: codebase-semantic-graph
description: Lightweight static code graph for impact analysis
version: 1.0.0
author: Anthony Lopez
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [software-development, static-analysis, impact-analysis, code-graph]
    category: software-development
---

# Codebase Semantic Graph

Use `ezra-graph` when you need quick impact analysis across local codebases without blind grep.

## When to Use

- Before refactoring a module and you want to see likely callers.
- When scoping a change and need an initial blast-radius map.
- To find orphan functions that may be dead code.

## Quick Reference

| Command | Purpose |
|---|---|
| `ezra-graph refresh` | Scan configured roots and rebuild the graph. |
| `ezra-graph refresh --root /path/to/repo` | Use a non-default scan root. |
| `ezra-graph callers <symbol>` | Show ranked callers of a symbol. |
| `ezra-graph blast-radius <file>` | Files importing or calling symbols in `<file>`. |
| `ezra-graph orphans` | Likely uncalled functions. |

## Artifact

Default database: `~/.hermes/ezra/graph/ezra-graph.sqlite` (override with `EZRA_GRAPH_DB`).

Default scan roots: the current working directory, or a `:`-separated list via `EZRA_GRAPH_ROOTS`.

## Procedure

1. Run `ezra-graph refresh` to populate the graph.
2. Query with `ezra-graph callers json.dumps`, `ezra-graph blast-radius /path/to/module.py`, or `ezra-graph orphans`.
3. Interpret results as triage hints, not proof — dynamic dispatch, string imports, and arbitrary expressions are intentionally skipped.

## Pitfalls

- `callers`, `blast-radius`, and `orphans` require an existing database. Run `refresh` first.
- JS/TS extraction is regex-based and intentionally best-effort.
- Use `EZRA_GRAPH_ROOTS` or repeated `--root` for large, non-local repositories.

## Verification

Confirm `ezra-graph refresh` produced non-zero file and symbol counts, and verify the database path exists, before relying on query output.
