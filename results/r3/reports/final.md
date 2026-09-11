# R3 Final Report — Holographic Scalability + Retrieval Intelligence

## Baseline
- Commit `f97a4102dd`, Hermes 0.21.1, Python 3.14.3, numpy 2.5.1,
  pytest 9.1.1, SQLite 3.50.4. Frozen: `results/r3/baseline/env.json`.
- Pre-change gate: 94/94 (CI-parity env). R2 retrieval P@1 0.9528/MRR 0.9623.

## Storage
- Root cause (profiled @n≈25, dim 1024): insert+fsync ~3.7ms, HRR encode
  ~5.9ms (64 SHA-256/fact-token), bank rebuild ~4.5ms growing O(n).
- Optimizations (additive, exact): bounded atom LRU cache; `add_facts_batch`
  (one txn, one bank update/category, atomic rollback); exact incremental
  bank via stored complex sums (`memory_banks.vector_sum`, nullable,
  fallback rebuild); `_write` commits only outside explicit txns.
- Latency: singles@1k 13.5ms → batch@1k **0.91ms (15x)**; @100 11.8→0.8ms.
- WAL identical-workload (scratch, 1k batch): 502 vs 580ms (+13%; the 3x
  singles advantage amortizes away) — production default UNCHANGED.
- Crash safety: batch atomicity, uncommitted-batch recovery, derived-bank
  rebuild, legacy-bank fallback — all tested. No silent corruption.

## Retrieval
- R2 dataset: P@1 0.9528→**0.9623**, MRR→**0.9717** (thai section 0.667→1.0).
- R1 gold post-R3: byte-identical (0.875/0.8906/0.9375).
- Tie-break: (score, updated_at, fact_id); trust still dominates (tested).
- Thai: stdlib bigram fallback, P@1 on Thai queries fixed, FP measured.
- Decision: NO DEPENDENCY (stdlib only) for Thai.
- Remaining misses (documented): sanitizer hyphen-merge (contract),
  tie without temporal model, stemming-level paraphrase.

## Thai: bigram fallback, NO/OPTIONAL/REQUIRED → NO DEPENDENCY.

## Semantic
- Backend discovery: no ollama binary/dir, no 127.0.0.1:11434 → unavailable.
- Dataset: 80 queries, classes LEXICAL/TOKEN/RANK/ENTITY/SEMANTIC
  (measured artifact `results/r3/semantic/gap_separation.json`).
- Result is ENVIRONMENT-DEPENDENT at top-3 near-tie boundaries (HRR weight
  on/off flips ±1 hit — root-caused, not flaky):
  numpy env: LEXICAL 1.0 / TOKEN 0.8 / RANK 1.0 / ENTITY 1.0 / SEMANTIC 0.2;
  no-numpy env: TOKEN 1.0 / SEMANTIC 0.217 (13/60).
  Conclusion invariant either way: deterministic ceiling proven, OPT-IN.
  (Methodology note: artifacts are harness outputs written by the tests
  themselves on each run — reproducible per environment, not independent.)

## LLM: normal 0, retrieval 0, prefetch 0, session 0, dream 0, benchmark 0.

## Scale
- 100: 10.4ms/add, 1.1ms/search. 1k: 15.0/2.6ms. 10k: 176ms, p95 22.5ms.
- 50k: 810ms/add, p50 92.5/p95 123.9/p99 143.5ms, update 2.6s, DB 238.6MB.
- 100k: impractical (~40h+ extrapolated) — documented, not faked.

## Security
- R2 suite repeated green; semantic attacks (paraphrase/instruction in
  source_refs/secret/authority-escalation/fake-truth/cross-project) refused
  by contract validation + trust ceiling; prefetch leakage 0; isolation
  (separate DBs) verified; no new egress.

## Regression: bare CI-parity env (numpy present) 119/119 green;
##   official runner (numpy-less venv) 93 passed, 0 failed, 6 skipped —
##   all 6 are numpy-absence skips by repo convention (4 R3 marks +
##   2 pre-existing whole-module importorskips: retrieval 13t, vector 11t).
## A/B: A=R2 frozen; B=R3 (quality +, storage 15x, security same-or-better,
##   R1 identical); C=disabled-semantic parity (identical outputs, 0 calls).
## Promotion: storage+retrieval hardening → PROMOTE; semantic backend → REJECTED
##   (no backend, deterministic wins); Thai dep → NO DEPENDENCY.
## Remaining: stemming-level paraphrase, temporal lifecycle model, Thai
##   segmentation dep re-evaluation, tie-break recency beyond insertion order.
## Reproduce: `bash scripts/run_tests.sh tests/plugins/memory/test_r3_storage.py
##   tests/plugins/memory/test_r3_retrieval.py tests/plugins/memory/test_r3_semantic.py
##   tests/plugins/memory/test_r3_quality.py` (plus R1/R2 files for full gate).
