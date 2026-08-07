# Holographic memory recall diagnostics

Task: t_0e2fbd42
Profile: responsible_memory_rollout

## Located implementation paths

- Holographic memory provider and `fact_store` / `fact_feedback` tool schemas: `plugins/memory/holographic/__init__.py`
- SQLite fact store, entity extraction, FTS5 schema, trust scoring, HRR vector persistence: `plugins/memory/holographic/store.py`
- Hybrid FTS/Jaccard/HRR retrieval plus `probe`, `related`, `reason`, and `contradict`: `plugins/memory/holographic/retrieval.py`
- HRR phase-vector algebra: `plugins/memory/holographic/holographic.py`
- Transcript ground truth tool: `tools/session_search_tool.py`
- Transcript SQLite/FTS5 implementation: `hermes_state.py`

## Root cause confirmed

The false negatives are repo-owned behavior, not just an operator issue.

1. Profile path ambiguity: configs that used `~/.hermes/memory_store.db` resolved through `Path.expanduser()` and process `HOME`, not the active profile `HERMES_HOME`. That can make a worker read/write a default/root store while the profile prompt claims holographic memory is active.
2. Retrieval candidate gating: `FactRetriever.search()` previously returned empty as soon as strict FTS5 `MATCH` found no rows. FTS5 multi-token queries are effectively AND-gated, so a query with one extra operational token (`session`) could miss a fact that had all other relevant terms in content/tags.
3. HRR-only false positives: `probe()` and `reason()` returned top-N rows even when every score was just random-vector noise. Tiny stores with unrelated rollout markers therefore looked like successful recall for entities such as `Lineage status`.
4. Ingestion remains a policy/product gap: the provider mirrors explicit built-in memory writes and optional narrow user-message auto extraction. It does not automatically convert session_search-confirmed corrections, assistant summaries, Kanban comments, or tool handoffs into facts.

## Changes made

- `plugins/memory/holographic/__init__.py`
  - Added `_resolve_profile_db_path()` so `~/.hermes/...` in holographic config resolves to active `HERMES_HOME` for profile-safe behavior.
  - Added `min_hrr_score` config wiring for probe/reason relevance filtering.
- `plugins/memory/holographic/retrieval.py`
  - Added deterministic overlap fallback candidates when strict FTS5 matching finds nothing.
  - Added a default HRR relevance floor (`min_hrr_score=0.35`) for `probe()` and `reason()` so unrelated rows are suppressed instead of returned as false positives.
- `scripts/evaluate_memory_recall.py`
  - Added an offline-safe JSON corpus harness that reads session_search ground truth from configured state DBs, seeds a temporary memory store, evaluates `search`/`probe`/`reason`, and emits JSON evidence. It never writes to live memory DBs.
- `tests/plugins/memory/test_holographic_provider.py`
  - Added regression coverage for profile-safe path resolution, fallback search recall, HRR false-positive suppression, and fact_store tool output.

## Ingestion recommendation

Do not automatically ingest arbitrary `session_search` results into live durable memory. Raw transcripts can contain stale task-local decisions, secrets in tool output, and context that should not survive a week.

Recommended policy:

1. Use `session_search` as ground truth for recall/evaluation and for human/operator review.
2. After a correction or project decision is verified through `session_search`, the coordinator or responsible lane owner should create an explicit compact `memory`/`fact_store` entry with source-aware wording.
3. A downstream watchdog/evaluator can periodically run `scripts/evaluate_memory_recall.py` against a reviewed corpus and produce missing-fact recommendations, but writes should remain explicit or gated until a redaction/source-classifier exists.

## Fix path recommendation

Primary path: upstream Hermes Agent patch. The path-resolution, FTS fallback, and HRR relevance-floor fixes are general product behavior and have tests.

Secondary path: downstream Hermes Maintenance evaluator. Use the new harness with a reviewed corpus of maintenance decisions (status semantics, Matrix E2EE recovery, Kanban blocked/comment dispatch rules) to catch future ingestion and recall drift.

Operational policy: keep explicit coordinator memory/fact_store writes for verified corrections; do not rely on automatic session-end extraction for assistant/Kanban/tool-derived operational decisions yet.
