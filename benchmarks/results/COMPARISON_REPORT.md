# Memory Backend Benchmark Comparison Report

**Date:** 2026-04-05/06
**Environment:** macOS, Apple Silicon, 16GB RAM, seed=42 (deterministic fixtures)
**Framework version:** 0.1.0 with capability-based category skipping
**Suites:** 15 (A-O), all with runners

## Full Results Matrix

All values are accuracy percentages (seed=42). `sk` = capability-skipped (fair),
`—` = not yet run.

| Suite | Category | #Scen | base | holo | mnem | mem0 | honch | hind |
|-------|----------|-------|------|------|------|------|-------|------|
| **A** | semantic_recall+3 | 200 | 82.5 | 70.0 | **86.0** | 75.5 | 74.8~ | 61.9 |
| **B** | compress+consolidation | 30 | 90.0 | 93.3 | **100** | sk | sk | sk |
| **C** | scopes | 20 | **65.0** | sk | sk | sk | sk | sk |
| **D** | adversarial | 15 | 86.7 | **100** | **100** | **100** | 93.3~ | 93.3 |
| **E** | scale | 8 | **100** | **100** | **100** | **100** | **100**~ | **100** |
| **F** | integration | 11 | sk | sk | sk | sk | sk | sk |
| **G** | qlearning | var | sk | sk | sk | sk | sk | sk |
| **H** | dedup (only) | 8 | **100** | 87.5 | 75.0 | **87.5** | 87.5~ | — |
| **I** | conv+stress | 15 | **86.7** | 80.0 | **86.7** | 73.3 | —~ | — |
| **J** | topic_shift | 8 | **100** | **100** | **100** | **100** | **100**~ | **100** |
| **K** | compress_survival | 8 | **100** | **100** | 75.0 | 75.0 | **100**~ | **100** |
| **L** | delegation | 8 | **75.0** | 62.5 | 50.0 | 50.0 | **75.0**~ | **75.0** |
| **M** | format_sensitivity | 10 | **90.0** | **90.0** | **90.0** | **90.0** | 80.0~ | 80.0 |
| **N** | retrieval_ablation | 9 | **88.9** | 66.7 | **88.9** | 66.7 | 77.8~ | 66.7 |
| **O** | timestamp | 8 | **100** | 62.5 | 62.5 | sk | sk | sk |

**Bold** = best or tied-for-best. `~` = degraded mode (see notes). `—` = not yet run.

### Aggregate Scores (comparable suites: A, D, E, J, K, L, M, N)

These suites have no required capabilities and all 6 backends can run them.

| Rank | Backend | Mean | Type | Notes |
|------|---------|------|------|-------|
| 1 | **baseline-flat** | **90.4%** | reference | Word overlap — surprisingly strong |
| 2 | honcho | 87.6%~ | local Docker | Degraded: no semantic search~ |
| 3 | mnemoria | 86.2% | local | New plugin (this PR) |
| 3 | holographic | 86.2% | local | Existing plugin |
| 5 | hindsight | 84.6% | local | Embeddings only, no LLM extraction |
| 6 | mem0 | 82.2% | cloud | After reset fix (+8.6% vs broken reset) |

### Additional Backends (not fully tested)

| Backend | Status | Blocker |
|---------|--------|---------|
| byterover | D=6.7%* M=30%* N=0%* | Rate-limited on free tier. Not a fair test. |
| openviking | 0% | Async VLM extraction too slow with local 14B model. |
| retaindb | not tested | Free signup at retaindb.com, needs account creation. |

## Fairness Notes

### Properly Tested (fair comparison)
- **baseline-flat**: Full capabilities exercised. Reference implementation.
- **holographic**: Full capabilities exercised. HRR + SQLite + FTS5.
- **mnemoria**: Full capabilities exercised. ONNX embeddings + BM25 + PPR.
- **mem0**: Fixed `reset()` to call `delete_all()` between scenarios. Cloud API.
- **hindsight**: Local embeddings (bge-small-en-v1.5) + reranker (ms-marco-MiniLM).
  Ran with `provider=none` — no LLM fact extraction. `retain()` stores raw text,
  `recall()` uses semantic similarity. This is fair for retrieval quality but
  does not exercise hindsight's full knowledge graph capabilities.

### Degraded Mode (results carry tilde)
- **honcho~**: Local Docker server has an expired OpenAI key (renews Apr 9).
  Embeddings fail, so our adapter falls back to client-side word overlap on
  raw session messages. Effectively tests message storage + basic recall,
  NOT honcho's semantic reasoning. Retest after Apr 9 for fair comparison.
- **byterover***: Hit daily rate limit on free ByteRover plan. Most queries
  returned empty. Not a valid benchmark — rerun after limit resets or with
  upgraded plan / BYO provider key.
- **openviking**: VLM extraction pipeline is async. With a local 14B model,
  each fact takes ~minutes to extract. Benchmark requires synchronous
  store-then-recall. Would need a fast cloud LLM or architectural changes.

## Bugs Found During Benchmarking

| Bug | Impact | Fix |
|-----|--------|-----|
| mem0 adapter `reset()` was no-op | Scores 8-19% lower than reality | Fixed: now calls `delete_all(user_id=...)` |
| mem0 adapter `search()` missing filters | 400 error on v2 API | Fixed: added `filters={"user_id": ...}` |
| Capability skipping not applied | Backends scored on unsupported features | Fixed: runner now checks `backend_supports_category()` |
| honcho adapter used v1 API | Honcho server runs v3 | Rewrote adapter for v3 REST API |
| hindsight adapter async/sync conflict | aiohttp session errors | Rewrote with dedicated worker thread |

## Suite Documentation

Each suite tests specific memory capabilities grounded in research:

| Suite | What It Tests | Research Basis |
|-------|---------------|----------------|
| A | Semantic recall, contradictions, temporal decay, cross-references, importance | ACT-R (Anderson & Lebiere 1998), Spreading Activation (Collins & Loftus 1975), BM25 (Robertson et al. 1995) |
| B | Compression survival, consolidation cycles | Complementary Learning Systems (McClelland et al. 1995) |
| C | Scope isolation (project/session boundaries) | Context-dependent memory retrieval |
| D | Adversarial prompt injection resistance | Prompt Injection (Perez & Ribeiro 2022) |
| E | Scale (10-200 facts), needle-in-haystack | Signal-to-noise discrimination, BM25/RRF |
| F | Full lifecycle (store, time, access, consolidate, recall) | End-to-end system verification |
| G | Q-value learning from reward signals | Q-Learning (Watkins & Dayan 1992) |
| H | Dedup, supersession, typed decay, notation parsing, scope lifecycle | Graph integrity (Tarjan 1972), structured knowledge |
| I | Multi-turn conversation, capacity stress (50-1000 facts) | Conversational memory patterns |
| J | Topic shift recall (recover context after pivot) | Working memory interference |
| K | Compression survival (critical facts survive summarization + noise) | Context window compression robustness |
| L | Delegation memory (child agent results recallable) | Multi-agent coordination |
| M | Format sensitivity (structured output handling) | Omni-SimpleMem (arXiv:2604.01007) |
| N | Retrieval ablation (keyword-only vs semantic-only signals) | Omni-SimpleMem (arXiv:2604.01007) |
| O | Timestamp integrity (temporal ordering) | Omni-SimpleMem (arXiv:2604.01007) |

## Performance

| Backend | Suite A | Suite D | Suite I | Type |
|---------|---------|---------|---------|------|
| baseline-flat | 0.1s | 0.0s | 0.0s | In-process |
| holographic | 2.5s | 0.3s | 43.5s | In-process (SQLite) |
| hindsight | 130s | 12s | — | Local API (embedded PG) |
| mnemoria | 716s | 57s | 2620s | Subprocess (ONNX) |
| mem0 | 1422s | 165s | — | Cloud API + reset |
| honcho | 2016s | 196s | — | Docker API |
