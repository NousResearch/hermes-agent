# Mnemoria Memory Plugin

Cognitive memory system for hermes-agent combining ACT-R activation scoring, typed facts with metabolic decay, Hebbian link formation, and RL Q-value reranking.

## Requirements

```bash
pip install mnemoria
# or with sentence-transformers for better semantic recall:
pip install 'mnemoria[embeddings]'
```

## Setup

```bash
hermes memory setup
# select "mnemoria" when prompted
```

Or set manually in config:

```
memory.provider = mnemoria
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HERMES_MNEMORIA_DB` | `~/.hermes/mnemoria.db` | SQLite database path |

When using hermes profiles, each profile gets its own database automatically (`~/.hermes/mnemoria-{profile}.db`).

## Lifecycle Integration

Mnemoria participates in the full agent lifecycle:

| Hook | What it does |
|------|-------------|
| **initialize** | Context-aware setup — read-only mode for cron/flush, per-profile DB scoping |
| **system_prompt_block** | Injects identity facts (Constraints/Decisions) + tool usage hints |
| **prefetch / queue_prefetch** | Background pre-warming for faster recall on next turn |
| **on_memory_write** | Mirrors MEMORY.md/USER.md writes as typed Mnemoria facts |
| **on_pre_compress** | Extracts facts from messages before context compression discards them |
| **on_delegation** | Stores delegation outcomes + extracts facts from subagent results |
| **on_session_end** | Final fact extraction + consolidation (promote/demote/prune) |

## Tools

| Tool | Description |
|------|-------------|
| `mnemoria_write` | Store a fact using plain text or MEMORY_SPEC notation |
| `mnemoria_recall` | Semantic recall with 4-signal fusion |
| `mnemoria_search` | Fast FTS5 keyword search |
| `mnemoria_reflect` | Synthesize all facts about a topic, grouped by type |
| `mnemoria_reward` | RL feedback signal for Q-value training |
| `mnemoria_explore` | Multi-hop discovery via Personalized PageRank |
| `mnemoria_stats` | Store health check (fact count, gauge %) |
| `mnemoria_consolidate` | Run maintenance (promote/demote/prune) |

## MEMORY_SPEC Notation

| Notation | Type | Decay Rate | Example |
|----------|------|-----------|---------|
| `C[t]:` | Constraint | Slow (0.3x) | `C[db.id]: UUID mandatory` |
| `D[t]:` | Decision | Medium (0.7x) | `D[auth]: JWT 7d refresh 6d` |
| `V[t]:` | Value | Normal (1.0x) | `V[api.prod]: api.example.com` |
| `?[t]:` | Unknown | Fast (2.0x) | `?[cache]: should we cache?` |

## Links

- Repo: https://github.com/Tranquil-Flow/mnemoria
- PyPI: `pip install mnemoria`
- License: AGPL-3.0
