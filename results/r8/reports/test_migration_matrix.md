# R8 Test Migration Matrix — 16 historical failures classified

Source: `tests/plugins/memory/test_holographic_cognitive.py` (50 tests;
34 pass, 16 fail with `AttributeError` only — verified by runner
`--tb=line` breakdown: `get_fact` ×8, `cognitive_search` ×4, `lineage` ×2,
`provider.run_maintenance` ×1, `provider.llm_calls` ×1).

## Classification

| # | old_test | old_api | current_api | reason | status | replacement_test |
|---|---|---|---|---|---|---|
| 1 | test_store_conflict_links_and_supersession | `store.lineage` + `get_fact` | `supersede_fact` + `fact_lineage` + `list_facts` stale flag | pre-R1 lineage API replaced by R4 lifecycle | INTENTIONALLY_OBSOLETE / REPLACED | test_r4_temporal.py; test_r7_integration.py::test_r7_backup_restore_integration |
| 2 | test_feedback_compat | `store.get_fact` | `list_facts` + trust columns | read accessor replaced; helpful delta (+0.05) already in test_r3_quality.py:84 | INTENTIONALLY_OBSOLETE / REPLACED | test_r3_quality.py (+0.05); test_r8_release.py::test_r8_feedback_deltas (adds −0.10) |
| 3 | test_firewall_quarantines_injection | `get_fact` + `cret.firewall_class` | provider `prefetch` firewall (filter-before-limit) | module-level firewall replaced by R2 prefetch firewall | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_hardening.py::test_r2_prefetch_firewall |
| 4 | test_firewall_and_budget | `get_fact` + `apply_firewall`/`apply_budget` | prefetch firewall + R6 bounded retrieval | replaced by R2 + R6 boundaries | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_prefetch_firewall; test_r6_ops.py (bounded) |
| 5 | test_cognitive_search_bounded | `cognitive_search` bundle | `search` + prefetch bound (≤5 lines @1k) | bundle API replaced by R2/R3 retrieval + R6 budget | INTENTIONALLY_OBSOLETE / REPLACED | test_r6_ops.py; test_r8_release.py (bounded retrieval) |
| 6 | test_fts_malformed_fallback | `cognitive_search` (1 call; rest uses current `search`) | `search` | single obsolete call; intent covered | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_hardening.py::test_r2_robustness |
| 7 | test_prefetch_zero_llm | `provider.llm_calls` / `last_diagnostics` | per-report `llm_calls: 0` + `get_metrics` | provider never carried counters in R-line | INTENTIONALLY_OBSOLETE / REPLACED | test_r6_ops.py (all paths); test_r7_integration.py |
| 8 | test_scenario_A_constraint_safe_injection | `cognitive_search` + `firewall_class` | `search` + prefetch firewall | replaced | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_prefetch_firewall; test_r8_release.py::test_r8_constraint_prefetch |
| 9 | test_scenario_B_supersession | `store.get_fact` lifecycle read | `list_facts` stale flag + SQL lifecycle | read accessor replaced | INTENTIONALLY_OBSOLETE / REPLACED | test_r4_temporal.py; test_r7_integration.py |
| 10 | test_scenario_E_dedupe | `store.get_fact` (keys via `cog.dedupe_keys`, still passing) | `list_facts` content read | read accessor replaced; pure-function part still green | INTENTIONALLY_OBSOLETE / REPLACED | test_dedupe_keys_stable (passing); test_r8_release.py::test_r8_dedupe_current_api |
| 11 | test_scenario_F_conflict_set | `store.lineage` | `contradict()` + supersession chain | replaced by R2/R4 conflict model | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_contradiction_entity_less; test_r4_revalidate_conflict |
| 12 | test_scenario_H_injection_quarantined | `cret.firewall_class` | prefetch firewall | same family as #3 | INTENTIONALLY_OBSOLETE / REPLACED | test_r2_prefetch_firewall; test_r5_security.py |
| 13 | test_scenario_J_large_db_bounded | `cognitive_search` @200 facts | bounded prefetch @1k | replaced, stronger bound in R6 | INTENTIONALLY_OBSOLETE / REPLACED | test_r6_ops.py |
| 14 | test_self_heal_repairs | `self_heal` trust auto-rewrite | health REPORTS (no auto-rewrite of authoritative data) | deliberate design change: trust corruption is human-boundary, never silently rewritten | INTENTIONALLY_OBSOLETE (documented, no replacement by design) | test_r6_ops.py (detection); OPERATIONS.md policy |
| 15 | test_migration_idempotent_existing_db | `ensure_cognitive_schema` + `get_fact` | `_init_db` additive migration + `ensure_migration` | cognitive-schema layer replaced | INTENTIONALLY_OBSOLETE / REPLACED | test_r4 migration tests; test_r6_ops.py; test_r7_integration.py (I15) |
| 16 | test_run_maintenance_no_llm | `provider.run_maintenance` + dream | `operations.run_maintenance` | provider-level maintenance replaced by R6 ops | INTENTIONALLY_OBSOLETE / REPLACED | test_r6_ops.py (every report `llm_calls == 0`) |

## Summary

- INTENTIONALLY_OBSOLETE: 16 total = 15 REPLACED (current equivalent cited)
  + 1 with no replacement by design (#14 trust auto-rewrite refused as
  human-boundary).
- REGRESSION: 0
- UNKNOWN: 0
- New current tests added for genuine gaps: unhelpful-feedback delta (−0.10),
  dedupe via current read API, constraint-prefetch scenario (all in
  `test_r8_release.py`, which also serves as the R8-15 canonical smoke).
