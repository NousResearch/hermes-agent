# Retrieval Ranking: First-Source Bias Debugging Pattern

Session class: RAG/local-memory retrieval returning chunks from the first documents instead of the most actionable answer evidence.

## Symptom

A user asks an action-oriented question such as "What should I improve?" or "What should I do next?" and the system repeatedly retrieves early chunks from the first imported/source-ordered documents, even when a later document contains a stronger actionable plan.

## Root-cause checklist

1. Inspect the database/query layer for global `LIMIT` before source balancing.
   - Anti-pattern: one FTS query over all selected source IDs with `ORDER BY rank, source_id, chunk_index LIMIT k`.
   - Why it fails: repeated weak matches in the first source can fill the pool before later sources are even represented.
2. Check every retrieval path, not just the main hybrid path.
   - Local-memory fallback/backends may have their own FTS sanitizer, scoring, or truncation.
   - If these differ, fixing hybrid retrieval leaves another first-source path alive.
3. Check whether question intent is lexical-only.
   - Queries like "How should I improve this?" may match generic `improve` noise but miss actionable answer terms (`actionable`, `plan`, `fix`, `implement`, `verify`).
4. Preserve scope boundaries.
   - Candidate balancing must never widen `explicit` or `none` scopes.
   - Use parameterized SQL for source filtering; do not string-format source IDs into queries.

## Regression fixture pattern

Create at least two sources:

- `aaa-first`: many early chunks with repeated but weak query terms, e.g. `General improve background note N. This is descriptive historical context only.`
- `zzz-actionable`: one later chunk with strong answer/action content, e.g. `Actionable improvement plan: fix retrieval ranking, implement fair source candidate pooling, and verify with a regression test.`

Assert the later actionable chunk is ranked first for an action-oriented query:

```rust
assert_eq!(outcome.results[0].source_id, "zzz-actionable");
assert!(outcome.results[0].content.contains("Actionable improvement plan"));
```

Also add the same regression for any parallel local-memory/search backend, not only the hybrid retrieval module.

## Durable fix patterns

- Replace global FTS top-k over all sources with per-source candidate pooling, then globally sort/rerank.
- Apply a small local intent rerank boost for action/improvement queries when content includes action markers (`actionable`, `fix`, `implement`, `plan`, `verify`, etc.).
- Expand action/improvement queries to include plausible answer terms, so lexical retrieval can find the actionable chunk before rerank.
- Sort with deterministic tie-breakers after score comparison (`source_id`, `chunk_index`, `chunk_id`) to avoid hash-map-order variance.
- Truncate only after scoring/reranking, not before every source has a chance to contribute.

## Verification pattern

1. Run the new focused regression first and confirm it fails before the fix.
2. Implement the smallest retrieval/scoring change.
3. Run focused retrieval/local-memory suites.
4. Run full Rust test/check gates and existing validation scripts.
5. Run an independent review or self-review specifically checking:
   - source scope is preserved;
   - SQL remains parameterized;
   - candidate balancing is not a hidden scope-widening fallback;
   - first-source bias no longer wins over stronger actionable evidence.
