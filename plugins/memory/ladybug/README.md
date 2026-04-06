# Ladybug Memory Plugin

Local, file-based memory for Hermes Agent backed by
[LadybugMemory](https://github.com/your-org/ladybug) — a columnar embedded graph
database (`.lbdb`), similar to DuckDB.
No API keys, no cloud — everything stays on disk in your `HERMES_HOME`.

## Features

| Capability | Details |
|---|---|
| Storage | Typed entries with importance scores (1–10) |
| Search | BM25 keyword search (`ladybug_search`) |
| Recall | Importance-weighted retrieval (`ladybug_recall`) |
| Graph | Named edges between entries (`ladybug_link`, `ladybug_related`) |
| Entity KG | GLiNER2 entity extraction (optional, `ladybug_entity`) |
| Prefetch | Background recall before every turn |
| Mirror | Syncs built-in MEMORY.md / USER.md writes automatically |

## Installation

```bash
pip install ladybug-memory          # required
pip install ladybug-memory[extract] # for entity extraction
```

## Activation

```bash
hermes memory select ladybug
```

Or manually in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: ladybug
```

## Configuration (optional)

All keys go under `memory.ladybug` in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: ladybug
  ladybug:
    db_path: ~/.hermes/ladybug.lbdb   # default
    prefetch_limit: 6                 # memories surfaced before each turn
    min_importance: 3                 # importance threshold for prefetch
    auto_link: false                  # chain-link mirrored built-in writes
```

## Tools exposed to the agent

| Tool | Purpose |
|---|---|
| `ladybug_store` | Persist a new memory entry |
| `ladybug_search` | Keyword / BM25 search |
| `ladybug_recall` | Retrieve recent / high-importance memories |
| `ladybug_update` | Correct or update a memory by ID |
| `ladybug_delete` | Delete a memory by ID |
| `ladybug_link` | Create a named relationship between two memories |
| `ladybug_related` | Traverse the relationship graph |
| `ladybug_entity` | Entity-level KG queries (requires GLiNER2) |

## Memory types

`general` · `preference` · `fact` · `project` · `person` · `event` · `task`

## Importance scores

1–10 scale. Higher scores surface more often in prefetch recall.
Built-in MEMORY.md / USER.md mirrors use importance **6** (explicit user signal).
Use `ladybug_update` to tune scores over time.
