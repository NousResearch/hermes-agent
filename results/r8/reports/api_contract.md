# Holographic Current API Contract (R8 canonical)

Derived from production source + R1–R8 passing suites. Historical API ≠
current API (see `test_migration_matrix.md`).

## PUBLIC (supported)

Provider (`HolographicMemoryProvider`, via `MemoryProvider` ABC):
`initialize`, `shutdown`, `prefetch`, `on_session_end` (auto_extract-gated),
`on_memory_write`, `get_tool_schemas`, `handle_tool_call` (single
`fact_store` tool with actions add/search/probe/related/reason/contradict/
update/…), plus ABC-inherited defaults (`sync_turn`, …).

Store (`MemoryStore`): `add_fact`, `add_facts_batch`, `update_fact`,
`remove_fact`, `list_facts`, `record_feedback`, `add_entity_alias`,
`verify_fact`, `revalidate_fact`, `mark_stale`, `revoke_fact`,
`supersede_fact`.

Retrieval (`FactRetriever`): `search`, `contradict`, `related`.

Operations (`operations.py`): `health_check`, `startup_check`, `repair`
(CHECK/REPAIR_DERIVED/REPAIR_SAFE/REPORT), `create_backup`, `verify_backup`,
`restore_to_scratch`, `rotate_backups`, `run_maintenance`
(NORMAL/LIGHT/DEEP), `migration_status`, `ensure_migration`,
`circuit_state`, `recent_events`, `get_stats`, `get_metrics`,
`hrr_available`, `network_dependencies`.

Config keys (`plugins.hermes-memory-store`): `db_path`, `auto_extract`,
`default_trust`, `min_trust_threshold`, `temporal_decay_half_life`,
`hrr_dim`, `hrr_weight`. No other keys exist — do not invent any.

## INTERNAL (do not depend on)

`store._conn/_lock/_write/_one`, `_rebuild_bank`, `_extract_entities`,
`retrieval._fts_candidates/_vector_rows`, `operations._record/_scrub/_over`,
`semantic_brain` (OPT-IN unwired interface, `AVAILABLE=False`).

## HISTORICAL (superseded — retained UNWIRED for provenance, no shim)

The following still exist as importable files but are on NO runtime path
(nothing in the wired provider/store/retrieval/operations imports them —
verified by grep): `MemoryStore.get_fact`/`lineage` (replaced by lifecycle
columns + `fact_lineage`), `FactRetriever.cognitive_search` (replaced by
hardened `search` + R6 budget), `HolographicMemoryProvider.run_maintenance`/
`llm_calls`/`last_diagnostics` (replaced by `operations.*`), the early
`taxonomy`/`cognition`/`cognitive_retrieval`/`maintenance`/`migration`
modules (kept byte-identical from stash@{1}; 34 of their 50 historical tests
still pass). NOTE: `maintenance.py:234 self_heal` can rewrite `trust_score`
if called directly — it is never called by any wired path (the wired repair
path `operations.repair` never rewrites `trust_score` (it only reads it for
detection; trust corruption is human-boundary by design). Mapping + rationale: `test_migration_matrix.md`.

## DEPRECATED / EXPERIMENTAL

None currently. `semantic_brain` is EXPERIMENTAL-OPT-IN (interface only,
no backend, never on the default path).

## Migration contract

`stock legacy DB → current schema → ops metadata`: additive (`IF NOT EXISTS`
/ `ADD COLUMN`), idempotent (rerun safe), transactional (single-writer
`_atomic` savepoints / one `backup()` call), recoverable (verify-first
restore-to-scratch only). Covered states: fresh, legacy, partially migrated,
already migrated (R4 + R6 + R7 suites).

## Test policy

- CURRENT suite (`tests/plugins/memory/test_*.py`): must be 0-failed. These
  assert the contract above — behavior relations, never snapshots or source text.
- HISTORICAL suite (`results/r8/legacy/`, `.HISTORICAL.py` suffix, not
  collected): evidence of past behavior. Never edited to pass, never moved back.
- Gold datasets (R1/R2/R3/R4): frozen; label edits forbidden.

## Release defaults (frozen)

LLM = OFF (0 calls on every path) · network = OFF · paid service = OFF ·
destructive autonomy = OFF (no delete/reset API; trust repair is human-boundary).
