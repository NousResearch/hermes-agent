# Truth Ledger evaluation report

Generated at: 2026-07-20T22:16:56.560412Z
Corpus: `docs/truth-ledger/evaluation/corpus.jsonl`

## Model/provider configuration

- Configured provider: `openai-codex`
- Configured model: `gpt-5.6-sol`
- Evaluation status: `measured`
- Extractor status counts: {'ok': 100, 'none': 45}
- Observed extractor routes: {'openai-codex/gpt-5.6-sol': 145}
- Provenance mismatches: 0
- Reported token/cost totals: tokens=0, estimated_cost_usd=0.0

## Corpus summary

- Total fixtures: 145
- Extracted turns: 145
- Expected admissible: 90
- Predicted admitted: 90
- No-fact fixtures: 55
- Deterministic gate skips: 45
- Duplicate suppressions: 10

## Quality metrics

- Precision: 1.0000
- Recall: 1.0000
- No-fact abstention accuracy: 1.0000
- Overall accuracy: 1.0000
- Leakage rate: 0.0000

## Acceptance gates

- Precision >= 0.95: PASS
- Recall >= 0.95: PASS
- No-fact abstention >= 0.95: PASS
- Leakage rate == 0: PASS
- Overall verdict: **PASS**

## Performance

- Hook enqueue latency ms: {'n': 250, 'p50': 0.873, 'p95': 1.159, 'p99': 1.973, 'max': 10.965}
- Hook pending spool size: 128030 bytes across 250 envelopes
- Hook path model/network calls: none (hook enqueues only)

### Projection rebuild benchmark

| facts | rebuild_ms | active | current_view_bytes |
|---:|---:|---:|---:|
| 1000 | 3.697 | 1000 | 224670 |
| 10000 | 55.076 | 10000 | 2276670 |
| 100000 | 461.974 | 100000 | 23066670 |

## Notes

- This report is generated from sanitized synthetic fixtures only.
- No private transcripts or raw conversation histories are persisted.
- Recall is reported and does not override precision gate policy.
