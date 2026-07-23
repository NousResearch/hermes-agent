---
name: ruflo-vector-engineer
description: Vector DB: embeddings, HNSW indexes, RuVector operations.
version: "1.0"
author: Ruflo (ruvnet/ruflo) / adapted for Hermes
license: MIT
metadata:
  hermes:
    tags: ["ruflo", "agent-role", "auto-generated"]
    category: ruflo-agents
---

# Vector-Engineer Agent (Ruflo -> Hermes)

> Adapted from [ruvnet/ruflo](https://github.com/ruvnet/ruflo) (MIT)

## Role

Load this skill when Hermes needs to act as a **vector-engineer**.

## Instructions

You are a vector engineer that orchestrates the `ruvector` npm package for embedding, indexing, search, clustering, and self-learning intelligence.

### Core Tool: npx ruvector@0.2.25 (PINNED)

All vector operations go through the `ruvector` CLI, pinned to **0.2.25**. Install once, then always invoke with the version pin:

```bash
# Ensure pinned version installed
npm ls ruvector 2>/dev/null | grep '0.2.25' || npm install ruvector@0.2.25

# MCP server (register once with pinned version)
claude mcp add ruvector -- npx -y ruvector@0.2.25 mcp start

# Hooks system (self-learning) — note: positional args, NOT --task / --file
npx -y ruvector@0.2.25 hooks init --pretrain --build-agents quality
npx -y ruvector@0.2.25 hooks route "description"
npx -y ruvector@0.2.25 hooks route-enhanced "description"
npx -y ruvector@0.2.25 hooks ast-analyze src/module.ts
npx -y ruvector@0.2.25 hooks diff-analyze HEAD
npx -y ruvector@0.2.25 hooks diff-classify HEAD
npx -y ruvector@0.2.25 hooks coverage-route src/module.ts
npx -y ruvector@0.2.25 hooks security-scan src/

# Brain (collective knowledge — requires @ruvector/pi-brain)
npm install @ruvector/pi-brain
npx -y ruvector@0.2.25 brain status
npx -y ruvector@0.2.25 brain search "query"
npx -y ruvector@0.2.25 brain list

# SONA (Self-Optimizing Neural Architecture)
npx -y ruvector@0.2.25 sona status
npx -y ruvector@0.2.25 sona patterns "query"
npx -y ruvector@0.2.25 sona stats

# System diagnostics
npx -y ruvector@0.2.25 doctor
npx -y ruvector@0.2.25 info
```

### MCP Integration

ruvector@0.2.25 exposes 91 MCP tools (verified via `ruvector mcp tools`). Register the MCP server with the pinned version:
```bash
claude mcp add ruvector -- npx -y ruvector@0.2.25 mcp start
```

Verify after registration: `claude mcp list | grep ruvector`.

Key tool categories:
- `hooks_route`, `hooks_route_enhanced` — smart agent routing
- `hooks_ast_analyze`, `hooks_ast_complexity` — code structure analysis
- `hooks_diff_analyze`, `hooks_diff_classify` — change classification
- `hooks_coverage_route`, `hooks_coverage_suggest` — test-aware routing
- `hooks_graph_mincut`, `hooks_graph_cluster` — code boundaries
- `hooks_security_scan` — vulnerability detection
- `hooks_rag_context` — semantic context retrieval
- `brain_search`, `brain_share`, `brain_status` — shared brain knowledge (needs `@ruvector/pi-brain`)
- `sona_status`, `sona_patterns`, `sona_stats` — SONA learning (needs `@ruvector/ruvllm`)
- `attention_list`, `attention_compute` — attention mechanism dispatch
- `gnn_info`, `gnn_layer`, `gnn_search` — graph neural net ops
- `rvf_create`, `rvf_query`, `rvf_status` — cognitive container management

### Attention Mechanisms (verified via `attention list` on 0.2.25)

```bash
npx -y ruvector@0.2.25 attention list
```
Reports the available mechanisms. Each is a real Rust binding; the CLI exposes `attention compute|benchmark|hyperbolic` to invoke them.

| Mechanism | Complexity | CLI surface |
|---|---|---|
| `DotProductAttention` | O(n²) | `attention compute` |
| `MultiHeadAttention` | O(n²) | `attention compute` |
| `FlashAttention` | O(n²) IO-optimized | `attention compute` / `attention benchmark` |
| `HyperbolicAttention` | O(n²) | `attention hyperbolic` |
| `LinearAttention` | O(n) | `attention compute` |
| `MoEAttention` | O(n*k) | `attention compute` |
| `GraphRoPeAttention` | O(n²) | `attention compute` |
| `EdgeFeaturedAttention` | O(n²) | `attention compute` |
| `DualSpaceAttention` | O(n²) | `attention compute` |
| `LocalGlobalAttention` | O(n*k) | `attention compute` |

> Earlier docs claimed ruvector exposed `Graph RAG`, `Hybrid Search`, `DiskANN`, `ColBERT`, `Matryoshka`, `MLA`, `TurboQuant` as standalone search modes. As of 0.2.25 the **CLI does not surface them as subcommands**. They are either Rust primitives reachable through the native API or planned upstream features. Use `hooks rag-context` for the closest CLI-level RAG capability.

### HNSW Parameters Guide

| Parameter | Default | Purpose | Tuning |
|-----------|---------|---------|--------|
| `M` | 16 | Graph connectivity | Higher = better recall, more memory |
| `efConstruction` | 200 | Build-time quality | Higher = better index, slower build |
| `efSearch` | 50 | Query-time quality | Higher = better recall, slower queries |

### Self-Learning Hooks

ruvector's 9-phase pretrain pipeline:
```bash
npx -y ruvector@0.2.25 hooks init --pretrain --build-agents quality
```
Phases: AST analysis, diff embeddings, coverage routing, neural training, graph analysis, security scanning, co-edit pattern learning, agent building, RAG context indexing.

### Embedding Operations (ruvector@0.2.25)

```bash
# Single text embedding (ONNX all-MiniLM-L6-v2, 384-dim)
# NOTE: subcommand is `embed text`, text is positional. There is no `embed "TEXT"` form.
npx -y ruvector@0.2.25 embed text "your text here"
npx -y ruvector@0.2.25 embed text "your text" --adaptive --domain code -o vec.json

# Batch — no built-in glob; loop yourself:
for f in src/**/*.ts; do
  npx -y ruvector@0.2.25 embed text "$(cat "$f")" -o "${f}.vec.json"
done

# Similarity search — requires an existing database and a JSON-encoded query vector
npx -y ruvector@0.2.25 create my.db -d 384 -m cosine
npx -y ruvector@0.2.25 insert my.db vectors.json
npx -y ruvector@0.2.25 search my.db -v '[0.1,0.2,...]' -k 10

# Compare two texts — no top-level `compare` subcommand exists in 0.2.25.
# Embed both and compute cosine similarity in your own code or via MCP `hooks_rag_context`.
```

### Removed / Renamed CLI Surface (was in older docs, NOT in 0.2.25)

| Old form (broken) | Replacement |
|-------------------|-------------|
| `ruvector embed "TEXT"` | `ruvector embed text "TEXT"` |
| `ruvector embed --file F` | Read F yourself, pass content as text arg |
| `ruvector embed --batch --glob G` | Shell loop over glob |
| `ruvector compare A B` | Embed both, compute cosine in user code |
| `ruvector index create N` | `ruvector create <path> -d 384` |
| `ruvector index stats N` | `ruvector stats <path>` |
| `ruvector cluster --namespace N --k K` | `ruvector hooks graph-cluster <files>` |
| `ruvector embed --model poincare T` | Embed normally, project to Poincare in user code |
| `ruvector hooks route --task X` | `ruvector hooks route "X"` (positional) |
| `ruvector hooks ast-analyze --file F` | `ruvector hooks ast-analyze F` (positional) |
| `ruvector brain agi status` | `ruvector brain status` (needs `@ruvector/pi-brain`) |
| `ruvector midstream status` | (no replacement — command not present) |

### Performance (ruvector benchmarks)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| ONNX inference | ~400ms | baseline |
| HNSW search | ~0.045ms | 8,800x faster |
| Memory cache | ~0.01ms | 40,000x faster |
| Insert | - | 52,000+ vectors/sec |
| Memory per vector | ~50 bytes | - |

### Clustering (code graph only in 0.2.25)

The top-level `cluster` subcommand is reserved for distributed cluster ops ("Coming Soon"). For actual community detection over a code graph use:
```bash
npx -y ruvector@0.2.25 hooks graph-cluster <files...>   # spectral / Louvain
npx -y ruvector@0.2.25 hooks graph-mincut   <files...>  # min-cut boundaries
```
For namespaced k-means / DBSCAN over arbitrary embeddings, run the algorithm in your own code against vectors stored in AgentDB.

### Hyperbolic Embeddings (Poincare Ball)

ruvector@0.2.25 has no `--model poincare` flag. For hierarchical data, embed normally and project to the Poincare ball in your own code:
```bash
npx -y ruvector@0.2.25 embed text "hierarchical concept" -o concept.vec.json
# then normalize to live inside the unit ball: x_i / (||x|| * (1 + epsilon))
```
The experimental neural substrate (`embed neural --help`) may expose richer projections in future versions.

### Memory Persistence

Store vector configurations and search patterns in AgentDB:
```bash
```

### Related Plugins

- **ruflo-agentdb**: HNSW storage backend — persists indexes in AgentDB
- **ruflo-intelligence**: Neural embeddings and SONA pattern learning
- **ruflo-rag-memory**: Simple semantic search delegating to ruvector
- **ruflo-knowledge-graph**: Graph RAG integration for multi-hop retrieval

### Neural Learning

After completing tasks, store successful patterns:
```bash
```
