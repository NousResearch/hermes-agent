---
name: codebase-semantic-graph
description: Build and query the Ezra/Hermes static code graph with ezra-graph.
---

# Codebase Semantic Graph

Use `ezra-graph` when you need quick impact analysis across Hermes, OpenClaw, and Mission Control without blind grep.

## Artifact

Default database:

```bash
/Users/Prime/.ezra/graph/ezra-graph.sqlite
```

Default scan roots:

- `/Users/Prime/.hermes/hermes-agent`
- `/Users/Prime/.openclaw/openclaw`
- `/Users/Prime/.openclaw/mission-control`

## Refresh

```bash
ezra-graph refresh
```

Optional scan roots:

```bash
ezra-graph refresh --root /path/to/repo --root /path/to/other
```

Alias debug mode emits a bounded sample of top-level Python import aliases that were used for call resolution:

```bash
ezra-graph refresh --alias-debug
```

## Resolution model

Python extraction uses `ast` and records full dotted call paths for `ast.Attribute` chains, including:

- `json.dumps(...)`
- `subprocess.run(...)`
- `ctx.register_tool(...)`
- `self.client.messages.create(...)`

Before persistence, top-level imports are used as an alias table:

- `import json as j; j.dumps(...)` records as `json.dumps` with `raw_callee=j.dumps`
- `from pathlib import Path as P; P.home(...)` records as `pathlib.Path.home` with `raw_callee=P.home`

Calls on arbitrary expressions remain conservative; use ripgrep/manual review when dynamic dispatch matters.

## Queries

Who calls a symbol:

```bash
ezra-graph callers json.dumps --limit 2000
```

Caller rows are ranked by default: frequency descending, then file diversity, then module locality. For raw path/line order:

```bash
ezra-graph callers json.dumps --no-rank
```

Blast radius for a file:

```bash
ezra-graph blast-radius /Users/Prime/.hermes/hermes-agent/tools/registry.py
```

Likely orphan functions:

```bash
ezra-graph orphans --limit 25
```

## Notes

- Python uses `ast`; JS/TS uses lightweight regex extraction.
- Results are meant for PR descriptions and refactor triage, not as a sole correctness proof.
- Refresh target for Prime should stay under 60s.
