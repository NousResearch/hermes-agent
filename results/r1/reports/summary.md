# R1 Memory Quality Benchmark — Report (HEAD f97a4102dd)

## Baseline (frozen, reproducible)
- Commit `f97a4102dd3864eed0c85132850ce7e06f13e09a`, tree clean at freeze.
- Python 3.14.3, numpy 2.5.1, pytest 9.1.1, SQLite 3.50.4.
- Stock holographic suite: **53/53 passed**.
- No local inference backend in runtime (auxiliary routing needs API keys;
  only an ollama-cloud model cache exists). No credentials present.
- **Scope note:** the prior phase's cognitive-layer files are absent from
  disk (verified by filesystem search); R1 benchmarks PRESENT TRUTH =
  stock Holographic. No silent rebuild was performed.

## Method
- One command: `python3 -m pytest tests/plugins/memory/test_r1_benchmark.py -q -p no:cacheprovider`
- Gold datasets A–H live in the test file (outside production code).
- Arms: `R1_ARM=baseline|candidate`, artifacts in `results/r1/<arm>/`.
- Gates: suite completes, `llm_calls == 0`, malformed input never raises,
  exact-dedupe stable, no row loss. Quality gaps are FINDINGS, not gates.

## Baseline quality (deterministic stock)
- Retrieval (16 queries): P@1 **0.875**, MRR **0.891**, R@5 **0.938**, nDCG@5 **0.902**.
- Formation: exact dedupe stable, no row growth, auto_extract decision→project OK.
- Safety: 11 malformed inputs, zero crashes. Secrets/injections stored
  verbatim (no refusal/quarantine in stock scope).
- Context: prefetch 85 chars / ~21 tokens / 0.71 ms, correct format.
- Scale: 100 facts → 10.4 ms/add, 1.1 ms/search, 108 KB;
  1000 facts → 15.0 ms/add, 2.6 ms/search, 460 KB. 10k/100k not practical
  here (per-commit fsync, journal_mode=DELETE on SQLite 3.50.4).

## Failure analysis
| ID | Class | Detail |
|----|-------|--------|
| qa2 "Which database?" | TYPE B/D | shared "project/use" tokens outrank database↔SQLite synonym gap |
| qb1 low-overlap paraphrase | TYPE D | zero token overlap → zero hits; token methods cannot bridge it |
| contradict() misses entity-less `k=v` pair | TYPE B/G | contradict() needs shared entities; entity-less facts skipped |
| secrets/injections accepted, no quarantine | TYPE G | stock has no classifier/firewall (out of R1 scope to fix) |
| classifier/temporal/lifecycle | TYPE G | capability absent, recorded not scored |
| dataset flaws found+fixed | TYPE E | qb1 verbatim-dupe, h_n4 near-dupe of answer (fixed, re-measured) |

Only qb1 (TYPE D) justifies a semantic model — and only as OPT-IN: no
backend exists offline.

## Semantic brain (OPT-IN interface only)
- `plugins/memory/holographic/semantic_brain.py`: SEMANTIC_BRAIN_AVAILABLE=False,
  ConfidenceGate, validated structured output, trust ceiling 0.6 (suggested 0.45,
  tier candidate), bounded SemanticCache (default 200, LRU), budget caps,
  secret/instruction screening pre- AND post-backend, dream pass default no-op.
- NOT wired into retrieval/prefetch/add paths. Zero normal-path calls by construction.
- 19 unit tests with deterministic stub backend (incl. cache-bypass-budget fix,
  bearer/JWT screening, A/B parity).

## A/B comparison
- A (deterministic) vs B (deterministic + disabled semantic): quality delta **0**,
  latency delta **0**, token delta **0**, LLM delta **0**, size delta **0**.
- Conclusion per promotion rule: deterministic wins by default; semantic stays
  **OPT-IN (interface only)**. A "do not ship a backend" outcome is the honest result.

## Promotion: OPT-IN (interface) / backend REJECTED (no evidence-possible offline)
## Security: passed (11 malformed safe; stub-adversarial refused; no network/secret egress)
## Regression: 53/53 stock + 7 benchmark + 19 semantic = 79/79
## Remaining: deterministic ranking synonyms (qa2), entity-less contradict, stock
##   screening/firewall, 10k+ scale profiling on WAL-capable SQLite — all follow-ups.
