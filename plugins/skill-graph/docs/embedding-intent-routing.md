# Skill Graph — Embedding + Intent Routing (Design)

> Status: Design (approved direction, not yet implemented)
> Scope: skill-graph plugin + hermes-agent core hook + adapter normalization
> Author: Hauzer S. Lee
> Related: hermes-mem `docs/gpu-embedding.md` (proven embedding backend),
>          `docs/skill-graph.md` (current FTS5-only discovery)

## Problem

At 900+ skills, skill discovery relies on the LLM issuing
`skill_graph_search(query)` — a tool call whose quality depends on the LLM
guessing the right query words against a purely lexical (FTS5) index.
Costs per user message:

- 2 LLM round-trips (search + load) before any real work
- Retrieval quality is lottery — FTS5 is AND-semantics, BM25 gets diluted on
  long multi-topic messages, Chinese is char-split (no stemming)
- Every task pays the discovery tax (pre_tool_call gating forces a search)

## Goals

1. **Deterministic discovery** — retrieval driven by code (embeddings),
   not LLM query guessing. LLM only does multiple-choice, not open search.
2. **Multi-intent support** — a long message with N topics gets per-intent
   candidates, each with guaranteed minimum candidates (not diluted).
3. **Zero manual maintenance** — embeddings refresh automatically when a
   skill changes (existing post_tool_call hook already sees skill_manage
   create/edit/patch/delete).
4. **Cross-model safe injection** — candidates reach the LLM without
   polluting user history and without tool-call dialect incompatibilities
   when switching providers (deepseek ↔ kimi).

## Architecture

```
┌─ Offline Index Layer (low frequency) ────────────────────────────┐
│ skill description + name + tags → embedding (bge-m3, 1024-dim)    │
│   primary:   TEI service  (http://localhost:8081, GPU)            │
│   fallback:  CPU sentence-transformers BAAI/bge-m3 (lazy load)    │
│ incremental: post_tool_call hook — on skill_manage               │
│              create/edit/patch/delete → recompute that skill only │
└──────────────────────────────────────────────────────────────────┘
┌─ Online Injection Layer (every user message) ────────────────────┐
│ ① one lightweight LLM call (deepseek-v4-flash):                  │
│    split intents (multi-topic) + scene determination             │
│ ② per-intent embedding retrieval → topK=5 each                   │
│    + scene weighting + usage stats weighting                     │
│ ③ merge + dedupe → 20-30 candidates (≥2-3 per intent)            │
│ ④ tool-call injection (assistant tool_calls + tool result pair)  │
│    after user message, before LLM reasoning                      │
│ ⑤ adapter layer normalizes tool_calls dialect — root-fixes       │
│    cross-provider switching (deepseek/kimi) incompatibility      │
└──────────────────────────────────────────────────────────────────┘
┌─ Execution Layer ────────────────────────────────────────────────┐
│ LLM uses injected candidates directly; skill_load for full text;  │
│ ad-hoc skill_graph_search as fallback; candidates carry 1-hop    │
│ graph neighbors                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Component 1 — EmbeddingClient (new file: `plugins/skill-graph/embedding_client.py`)

Refactored from `hermes-mem/hermes_mem/tei_client.py` (MIT, same author
ecosystem). Keeps the three proven mechanisms, decouples config source.

```
EmbeddingClient
  __init__(config_reader: Callable[[], dict])   # injected, not hardcoded
  embed(texts: list[str]) -> list[list[float]]  # normalized (dot=cos)
  health_check() -> bool                        # 2s timeout, cached 60s

Backend selection (config keys, same defaults as hermes-mem):
  embedding_backend: auto    # tei | ollama | cpu | auto
  embedding_api_url: http://localhost:8081
  embedding_model: bge-m3

Config read from skills.config.skill-graph.* (plugin-own section).
Protocol auto-detect on endpoint:
  /health      → TEI
  /api/tags    → Ollama
  /v1/models   → OpenAI-compatible
Any failure → CPU fallback (sentence-transformers BAAI/bge-m3, lazy-load).
```

Why not copy `tei_client.py` verbatim:

- Config path is hardcoded to `plugins.config.hermes-mem` in 3+ functions —
  violates the decision to use `skills.config.skill-graph`.
- Includes hermes-mem-specific API (`get_extraction_endpoint`, extraction
  queue helpers) irrelevant to skill-graph.
- A generic client with an injected config reader lets a third plugin reuse
  it without another fork.

Reuse note: the TEI service `hermes-mem-embedding-tei-server` (port 8081,
bge-m3, 1024-dim, XLM-RoBERTa) is already running — verified live with a
Chinese embed test. No new GPU/model setup needed.

## Component 2 — Vector index (SQLite)

New table in the existing skill-graph DB (`skill_nodes` gains embedding
columns, or a separate `skill_embeddings` table):

```
CREATE TABLE skill_embeddings (
    skill_name  TEXT PRIMARY KEY REFERENCES skill_nodes(name),
    vector      BLOB NOT NULL,          -- 1024 × float32 = 4096 bytes
    model       TEXT NOT NULL,          -- 'bge-m3' (record model for invalidation)
    dim         INTEGER NOT NULL,       -- 1024
    updated_at  REAL NOT NULL
);
CREATE INDEX idx_embeddings_model ON skill_embeddings(model);
```

- **Full rebuild**: `/skill-graph rebuild` recomputes embeddings for all
  skills (local TEI batch, fast; ~900 × short descriptions).
- **Incremental**: post_tool_call hook (already present, intercepts
  skill_manage create/edit/patch/delete) → recompute that one skill.
- **Query-time**: cosine (dot on normalized vectors) over all rows, topK
  per intent.
- Model-change invalidation: if `embedding_model` config changes, embeddings
  with a different `model` value are recomputed on next rebuild.

## Component 3 — Intent split (one lightweight LLM call)

Before retrieval, a single plain chat completion (no SOUL.md, no memory
injection — just the user message):

```
Input:  user message text
Output: JSON { "intents": ["<topic A as a sentence>", "<topic B as a sentence>", ...],
               "scene": "<coding|writing|research|design|devops|hermes|media|common>" }
```

- Multi-intent: the splitter separates topics; each intent is a *sentence*,
  not keywords — sentence-level embedding queries match better than word
  lists.
- Scene is determined in the same call (zero extra cost) and used as a soft
  weight in retrieval — revives the currently-dead `scenes` data without
  relying on the LLM to pass a `scenes` parameter to the search tool.
- Model: `deepseek-v4-flash` (enrichment model, fast/cheap).
- Cost: ~200 tokens out per message. Cacheable if message is unchanged.

## Component 4 — Injection (tool-call pair) + adapter normalization

**Decision (user): most aggressive option — tool-call injection + adapter
layer unified conversion.**

Mechanism:

1. New core hook in `agent/conversation_loop.py`: after user message is
   appended, before LLM first reasoning — plugins may inject a tool-call
   pair (assistant message with `tool_calls` + matching `tool` result).
   This is a real hook with a concrete consumer (skill-graph), satisfying
   the "no speculative infrastructure" bar.
2. skill-graph plugin builds the pair:
   - assistant message: `tool_calls=[{id, type:"function",
     function:{name:"skill_graph_search", arguments:"{...intent results...}"}}]`
   - tool result: `{role:"tool", tool_call_id:<same id>, content:"<candidates>"}`
   - Role alternation stays valid (user → assistant(tool_calls) → tool → assistant).
3. **Adapter normalization** (root fix for cross-provider switching):
   hermes adapters (`anthropic_adapter`, openai path, etc.) already convert
   message formats; extend them to canonicalize `tool_calls` dialect so a
   history produced under deepseek is parseable when the session switches
   to kimi and back. This is the definitive fix for the reported
   "deepseek does not recognize kimi's tool call messages" issue.

Injection shape must keep the LLM from treating candidates as a user
message (it won't — they arrive as a tool result), and keep prompt caching
stable (injection happens on the same position each turn, so the cached
prefix up to the user message is preserved).

## Component 5 — Retrieval & ranking

Per intent i:

```
query_embedding = client.embed([intent_i])
candidates = cosine topK (K=5) over skill_embeddings
score = cosine × (1 + 0.3 × scene_match) × (1 + 0.1 × usage_boost)
         usage_boost from skill_term_stats (search_count/load_count)
```

Merge across intents, dedupe, cap total ~20-30. Each candidate carries:
name + one-line description + 1-hop graph neighbors (from skill_edges) so
the LLM sees related skills even if topK is narrow.

## Configuration

```yaml
skills:
  config:
    skill-graph:
      embedding_backend: auto      # tei | ollama | cpu | auto (default auto)
      embedding_api_url: http://localhost:8081   # same default as hermes-mem
      embedding_model: bge-m3
      intent_split_model: deepseek-v4-flash      # reuse enrichment model default
      inject_topk_per_intent: 5
      inject_max_candidates: 30
      inject_min_per_intent: 3
```

## Implementation order

1. `embedding_client.py` (plugin) — port from tei_client.py, injectable config
2. `skill_embeddings` table + rebuild integration (plugin)
3. Incremental recompute in existing post_tool_call hook (plugin)
4. Intent split call (plugin)
5. Core hook: user-message injection point (hermes-agent, feat branch)
6. Adapter tool_calls normalization (hermes-agent, same feat branch)
7. E2E test: multi-intent message → correct per-intent candidates → LLM
   completes task without ad-hoc search

Steps 5-6 are core changes; per hermes-agent-dev pipeline they land on the
same feat branch as the plugin changes (consumer + hook must not split).

## Open questions

- Tool-call injection vs plain-text inject for the *first* iteration —
  decision is tool-call pair (aggressive), but a text `{"action":"inject"}`
  fallback remains as a safety valve if adapter normalization is delayed.
- Whether `skill_graph_mode` divergence (integration reads config at
  registration; repo reads agent attr in hook) gets unified in the same
  branch — separate concern, needs its own decision.
- Embedding storage in main skill-graph DB vs sidecar DB — main DB keeps
  FTS5 graph queries fast; embedding BLOB (4KB/skill) is fine inline.

## References

- `docs/skill-graph.md` — current FTS5-only architecture
- `hermes-mem/docs/gpu-embedding.md` — proven TEI/Ollama/CPU backend design
- `hermes-mem/hermes_mem/tei_client.py` — source for embedding_client.py
- `skill-discovery-routing` skill — analysis of scenes dead-code, costs
