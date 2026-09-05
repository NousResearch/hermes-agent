# Natural-language session-search language packs

Language packs are data-only registry entries in `hermes_state_nl_expansion.py`.
They must not change the session-search pipeline.

## Required evidence for a new pack

A contribution must include all of the following:

1. **Schema data**: `stopwords`, `affinity_stopwords`, morphology settings and
   a conservative `min_stem` precision floor.
2. **Routing coverage**: at least one query with a language-specific affinity
   marker, plus an ambiguity case for every script/lexical neighbour that could
   otherwise win detection.
3. **Search coverage**: at least three privacy-safe synthetic cases spanning
   stopword removal, inflection and an ordinary query. Add the pack name to
   each case's `packs` field in `nl_search_eval_v1.json`.
4. **Adversarial coverage**: add a lexical-near-miss or cross-language cognate
   case when the new language shares vocabulary with an existing pack.
5. **Human validation**: a fluent speaker or a documented authoritative grammar
   source must review stopwords, affinity markers and morphology assumptions.
   Record the source/reviewer in the PR body; do not add personal identifiers
   to the repository.

## Evaluation protocol

Run the pack-aware harness against only the packs present on the branch:

```bash
PYTHONPATH=. python scripts/nl_search_eval.py --packs default,new-pack
```

The runner creates one temporary SQLite database per case. It reports
`hit@1`, `recall@5`, `precision@5`, MRR, absent-query accuracy, and p50/p95
latency. It is a controlled mechanism regression suite, not a claim about
population-level relevance.

## Review boundaries

- `SessionDB.search_messages()` remains strict by default.
- Only conversational callers opt into `natural_language=True`.
- Explicit FTS5 syntax is never expanded.
- New packs may not add runtime dependencies or database schema changes.
- Keep broad OR recovery behind the explicit conversational path and include
  a relevance counterexample whenever its vocabulary is widened.

## Controlled rollout

`HERMES_SEARCH_NL_ROUTE_LOG=1` emits only successful NL fallback events:
route, selected language, elapsed time, result count and query length. Query
text stays out of logs. Use this in a bounded rollout to determine whether a
pack is selected and useful before proposing it as a default behavior change.

For temporary text-level diagnostics only:

```bash
HERMES_SEARCH_LOG_QUERY=1
```

Do not enable that setting as a normal production default.
