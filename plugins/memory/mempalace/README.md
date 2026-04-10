# MemPalace — Hermes Memory Provider Plugin

[MemPalace](https://github.com/milla-jovovich/mempalace) integration for
[Hermes Agent](https://github.com/NousResearch/hermes-agent). Stores verbatim
conversations in a ChromaDB-backed memory palace with semantic search and a
temporal knowledge graph.

**96.6% LongMemEval R@5** — the highest recall score ever published for an AI
memory system. Local-only, zero API calls, free.

## Install

```bash
pip install mempalace    # pulls in chromadb
hermes memory setup      # select "mempalace"
```

Or manually in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mempalace
```

## How It Works

MemPalace uses a 4-layer memory stack:

| Layer | What | Tokens | When |
|-------|------|--------|------|
| L0 | Identity (`~/.mempalace/identity.txt`) | ~100 | Always (system prompt) |
| L1 | Essential Story (top 15 drawers) | ~500-800 | Always (system prompt) |
| L2 | On-Demand (wing/room filtered) | ~200-500 | Tool call |
| L3 | Deep Search (full semantic) | unlimited | Tool call / prefetch |

Conversations are stored verbatim — no summarization, no extraction. The
philosophy: store everything, make it findable.

## Tools Exposed

| Tool | Description |
|------|-------------|
| `palace_search` | Semantic search across all memories |
| `palace_store` | Store a new memory with wing/room/importance |
| `palace_browse` | Navigate the palace structure (wings, rooms, counts) |
| `palace_kg` | Knowledge graph: query, add, timeline, invalidate |
| `palace_status` | Palace stats and layer info |

## Lifecycle Hooks

- **`sync_turn`** — Every conversation turn is stored as a drawer (background thread)
- **`on_session_end`** — Full session is archived with high importance
- **`on_pre_compress`** — Saves context before Hermes discards it during compression
- **`on_memory_write`** — Mirrors built-in memory writes (add/replace) to the palace

## Configuration

In `~/.hermes/config.yaml`:

```yaml
plugins:
  mempalace:
    palace_path: ~/.mempalace/palace     # ChromaDB storage location
    wing: hermes                         # default wing for conversations
    kg_enabled: true                     # temporal knowledge graph
    identity_path: ~/.mempalace/identity.txt
```

## Palace Architecture

```
Palace (ChromaDB)
├── Wings (projects/topics)
│   ├── hermes/
│   │   ├── conversation    — live turn-by-turn storage
│   │   ├── session_archive — end-of-session consolidated
│   │   ├── precompress     — saved before context compression
│   │   ├── memory_user     — mirrored from built-in USER.md
│   │   ├── memory_memory   — mirrored from built-in MEMORY.md
│   │   └── general         — explicit palace_store calls
│   └── (other wings...)
└── Knowledge Graph (SQLite)
    ├── Entities (people, projects, tools)
    └── Triples (subject → predicate → object, with temporal validity)
```

## Credits

- [MemPalace](https://github.com/milla-jovovich/mempalace) by Ben Sigman ([@bensig](https://github.com/milla-jovovich))
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research
- Plugin by Jean Clawd van Amsterdam
