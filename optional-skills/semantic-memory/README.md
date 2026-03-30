# semantic-memory — Optional Skill for Hermes

**Semantic, self-evolving memory backed by real vector embeddings.**

This optional skill upgrades Hermes's memory from flat markdown (MEMORY.md,
2,200-char limit) to a persistent SQLite + embedding store with hybrid
BM25 + cosine retrieval and temporal decay.

## The problem this solves

Hermes's built-in memory system has two components today:

1. **MEMORY.md** — injected into every system prompt. Hard cap of ~2,200
   chars. Great for hot facts, but fills up fast on long-running projects.
2. **session_search** — FTS5 keyword search over past sessions. Fast, but
   finds only exact keyword matches. Searching for "payment issues" won't
   find a session that talked about "Stripe checkout broke."

**The gap**: there's no way to store an unbounded set of facts and retrieve
them by *meaning*. This skill fills that gap.

## What this adds

| Capability | Built-in Hermes | With semantic-memory |
|------------|-----------------|----------------------|
| Memory capacity | ~2,200 chars | Unlimited (SQLite) |
| Search type | Keyword (FTS5) | Hybrid: BM25 + semantic (384-dim vectors) |
| Temporal weighting | None | Decay (λ=0.05) + reference boost |
| Fact relationships | None | Graph (co-access, entity, domain) |
| Auto-tagging | None | Domain + entity detection on ingest |
| Session capture | Manual | `session_hook.py` auto-extracts facts |

## Architecture

```
┌─────────────────────────────────────────┐
│  SURFACE BUFFER  (hot facts, activated) │  ← Agent reads here
├─────────────────────────────────────────┤
│  RETRIEVAL LAYER (hybrid search)        │  ← BM25 + Semantic + Temporal
├─────────────────────────────────────────┤
│  DEEP LAYER      (activation engine)   │  ← Pushes facts up, not pulled
├─────────────────────────────────────────┤
│  STORAGE LAYER   (SQLite + embeddings) │  ← memory.db, schema.sql
└─────────────────────────────────────────┘
```

## Why optional-skills (not bundled)

Per CONTRIBUTING.md: bundled skills should be "broadly useful to most users"
without heavyweight dependencies. `fastembed` downloads a ~130MB model
(BAAI/bge-small-en-v1.5) on first use — appropriate for optional-skills,
not bundled.

Users who want semantic memory opt in with:
```bash
hermes skills install semantic-memory
pip install fastembed
```

## Files

```
optional-skills/semantic-memory/
├── SKILL.md                    ← Agent-facing instructions
├── README.md                   ← This file
├── scripts/
│   ├── mem                     ← CLI entry point (store/search/stats/recent/ingest)
│   ├── memory_engine.py        ← Core DB + embedding operations
│   ├── hybrid_retriever.py     ← BM25 + cosine + temporal ranking
│   ├── embedder.py             ← fastembed wrapper (BAAI/bge-small-en-v1.5)
│   ├── deep_layer.py           ← Reverse-flow activation engine
│   ├── decay_scheduler.py      ← Temporal decay with λ=0.05
│   ├── session_hook.py         ← Auto-extract facts from session text
│   ├── context_selector.py     ← Token-budgeted context injection
│   ├── semantic_index.py       ← Embedding-based search
│   └── integrated_retriever.py ← Unified search interface
├── db/
│   └── schema.sql              ← SQLite schema (auto-applied on first use)
└── config/
    └── memory-engine.yaml      ← Default configuration
```

## Dependencies

| Package | Version | Why |
|---------|---------|-----|
| `fastembed` | ≥0.3 | Local embeddings, no API key, free |

No new dependencies on the Hermes core. `fastembed` is user-installed.
The skill degrades gracefully: if fastembed isn't installed, `mem` prints
a clear install message and exits.

## Zero changes to Hermes core

This skill requires **no modifications** to:
- `run_agent.py`
- `tools/memory_tool.py`
- `hermes_state.py`
- Any other core file

It is purely additive. The agent calls it via the existing `terminal` tool.

## Testing

```bash
# Install dep
pip install fastembed

# First run — auto-creates ~/.hermes/memory-engine/db/memory.db
python optional-skills/semantic-memory/scripts/mem stats

# Store and retrieve
python optional-skills/semantic-memory/scripts/mem store "Test fact: Stripe checkout fixed by removing consent_collection"
python optional-skills/semantic-memory/scripts/mem search "what happened with payments"
# Expected: returns the fact above with high cosine similarity

# Session hook
python optional-skills/semantic-memory/scripts/session_hook.py "Fixed auth bug. Deployed to staging."
python optional-skills/semantic-memory/scripts/mem recent 5
```

## Origin

Core memory engine extracted and cleaned from
[S+Memory](https://github.com/FalconOrtiz/SPlus-Memory) — a layered
semantic memory system built specifically for Hermes integration.
AGI modules, multi-agent orchestration, and Convex backend were excluded
to keep this skill focused and dependency-free.

MIT License.
