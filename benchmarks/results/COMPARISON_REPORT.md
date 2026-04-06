# Memory Backend Benchmark Comparison Report

**Date:** 2026-04-05/06
**Environment:** macOS, Apple Silicon, 16GB RAM, seed=42 (deterministic fixtures)
**Framework version:** 0.1.0 with capability-based category skipping
**Suites:** 15 (A-O), all with runners

---

## Testing Transparency & Reproducibility

**These benchmarks were run by the developers of this framework.** We encourage
independent verification before drawing conclusions from these results.

### How to reproduce

All results are deterministic. To reproduce any score:

```bash
python -m benchmarks --backend <name> --suite all --seeds 42
```

Fixture data is committed in `benchmarks/suite_{a..o}/fixtures/`. The heuristic
judge is built-in and requires no API keys.

### Known limitations

- **Single seed**: Results use seed=42 only. Multi-seed runs (`--runs 3+`)
  would provide confidence intervals. We did not run multi-seed for all backends.
- **Heuristic judge**: Scoring uses keyword/substring matching, not LLM
  evaluation. This is deterministic and reproducible but may miss semantic
  equivalence. Use `--judge-model <model>` for LLM-based scoring.
- **honcho (degraded)**: Tested with an expired OpenAI key -- embeddings
  failed, adapter fell back to client-side word overlap. All honcho scores
  are marked with `~`. Must be retested with working embeddings.
- **hindsight (partial)**: Tested with `provider=none` (no LLM extraction).
  Stores raw text, retrieves by embedding similarity only. Does NOT exercise
  hindsight's full knowledge graph capabilities.
- **byterover**: Hit daily rate limit on free tier. Most queries returned
  empty. Partial results included but NOT valid for comparison.
- **openviking**: Async VLM extraction too slow with a local 14B model.
  Effectively untested. Adapter is provided but needs a fast cloud LLM.
- **retaindb**: Adapter is provided but no account was created to run
  benchmarks. Untested.
- **No reset for some cloud backends**: mem0 calls `delete_all()` between
  scenarios. hindsight has no delete API -- facts may accumulate across
  scenarios within a single run, which could inflate or deflate scores.

### Pre-merge requirements

The following must be completed before this PR should be merged:

1. **byterover**: Rerun with a working API key or upgraded plan
2. **openviking**: Rerun with a fast cloud LLM backend, or document as
   "local-only, not benchmarkable at synchronous speeds"
3. **retaindb**: Create account, run full suite, add results
4. **honcho**: Retest after OpenAI key renewal for fair embedding-based scores

### What a third party should re-verify

1. Run the full suite on each backend independently
2. Compare scores against this report
3. Test with multiple seeds (`--seeds 42 43 44 45 46`)
4. Use LLM judge (`--judge-model claude-3-haiku-20240307`) for semantic
   scoring on ambiguous scenarios
5. Verify honcho with working OpenAI embeddings
6. Verify hindsight with LLM extraction enabled

---

## Full Results Matrix

All values are accuracy percentages (seed=42). `sk` = capability-skipped (fair),
`--` = not yet run or not valid.

| Suite | Category | #Scen | base | holo | mem0 | honch~ | hind |
|-------|----------|-------|------|------|------|--------|------|
| **A** | semantic_recall+3 | 200 | 82.5 | 70.0 | 75.5 | 74.8~ | 61.9 |
| **B** | compress+consolidation | 30 | 90.0 | 93.3 | sk | sk | sk |
| **C** | scopes | 20 | **65.0** | sk | sk | sk | sk |
| **D** | adversarial | 15 | 86.7 | **100** | **100** | 93.3~ | 93.3 |
| **E** | scale | 8 | **100** | **100** | **100** | **100**~ | **100** |
| **F** | integration | 11 | sk | sk | sk | sk | sk |
| **G** | qlearning | var | sk | sk | sk | sk | sk |
| **H** | dedup (only) | 8 | **100** | 87.5 | **87.5** | 87.5~ | -- |
| **I** | conv+stress | 15 | **86.7** | 80.0 | 73.3 | --~ | -- |
| **J** | topic_shift | 8 | **100** | **100** | **100** | **100**~ | **100** |
| **K** | compress_survival | 8 | **100** | **100** | 75.0 | **100**~ | **100** |
| **L** | delegation | 8 | **75.0** | 62.5 | 50.0 | **75.0**~ | **75.0** |
| **M** | format_sensitivity | 10 | **90.0** | **90.0** | **90.0** | 80.0~ | 80.0 |
| **N** | retrieval_ablation | 9 | **88.9** | 66.7 | 66.7 | 77.8~ | 66.7 |
| **O** | timestamp | 8 | **100** | 62.5 | sk | sk | sk |

**Bold** = best or tied-for-best. `~` = degraded mode (see notes). `--` = not yet run.

### Backends with incomplete results

| Backend | Status | What was tested | Blocker |
|---------|--------|-----------------|---------|
| byterover | partial | D=6.7%, M=30%, N=0% | Rate-limited on free tier. Not valid for comparison. |
| openviking | failed | 0% on all attempted | VLM extraction too slow with local 14B model. |
| retaindb | untested | -- | No account created. Free tier at retaindb.com. |

### Aggregate Scores (comparable suites: A, D, E, J, K, L, M, N)

These suites have no required capabilities and all tested backends can run them.

| Rank | Backend | Mean | Type | Notes |
|------|---------|------|------|-------|
| 1 | **baseline-flat** | **90.4%** | reference | Word overlap -- surprisingly strong |
| 2 | honcho | 87.6%~ | local Docker | Degraded: no semantic search~ |
| 3 | holographic | 86.2% | local | Existing plugin |
| 4 | hindsight | 84.6%* | local | *J/K/L only; H/I not yet run |
| 5 | mem0 | 82.2% | cloud | After reset fix (+8.6% vs broken reset) |

*Hindsight aggregate uses all 8 comparable suites where data is available.

### Observations

- **baseline-flat leads**: The reference TF-IDF + word-overlap backend scores
  highest. This is likely because the heuristic judge also uses word overlap,
  creating a favorable alignment. LLM-judge runs may shift rankings.
- **All backends handle scale well**: Suite E (10-200 facts) shows 100% across
  all tested backends.
- **Delegation memory is hard**: Suite L (child agent results) is the weakest
  category for most backends, topping out at 75%.
- **Adversarial resilience is strong**: Most backends score 86%+ on prompt
  injection resistance (Suite D).

## Fairness Notes

### Properly tested (fair comparison)
- **baseline-flat**: Full capabilities exercised. Reference implementation.
- **holographic**: Full capabilities exercised. HRR + SQLite + FTS5.
- **mem0**: Fixed `reset()` to call `delete_all()` between scenarios. Cloud API.

### Degraded mode (results carry tilde ~)
- **honcho~**: Local Docker server has an expired OpenAI key. Embeddings fail,
  so adapter falls back to client-side word overlap on raw session messages.
  Effectively tests message storage + basic recall, NOT honcho's semantic
  reasoning. Must retest after key renewal.

### Partial / limited mode
- **hindsight**: Local embeddings (bge-small-en-v1.5) + reranker (ms-marco-MiniLM).
  Ran with `provider=none` -- no LLM fact extraction. `retain()` stores raw text,
  `recall()` uses semantic similarity. Fair for retrieval quality but does not
  exercise hindsight's full knowledge graph capabilities.

### Not yet valid
- **byterover**: Rate-limited. Results not usable for comparison.
- **openviking**: Too slow for synchronous benchmarking with local model.
- **retaindb**: Not tested.

## Bugs Found During Benchmarking

| Bug | Impact | Fix |
|-----|--------|-----|
| mem0 adapter `reset()` was no-op | Scores 8-19% lower than reality | Fixed: now calls `delete_all(user_id=...)` |
| mem0 adapter `search()` missing filters | 400 error on v2 API | Fixed: added `filters={"user_id": ...}` |
| Capability skipping not applied | Backends scored on unsupported features | Fixed: runner checks `backend_supports_category()` |
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
| hindsight | 130s | 12s | -- | Local API (embedded PG) |
| mem0 | 1422s | 165s | -- | Cloud API + reset |
| honcho | 2016s | 196s | -- | Docker API |
