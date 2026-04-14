# Obsidian Knowledge Freshness + Retrieval Gate Contract v1 (`XK-203`)

Date: 2026-03-28  
Status: active (canonical for `XK-203`)  
Owner: Architect  
Scope: freshness SLO and abstain-quality retrieval gate for Obsidian knowledge surfaces

---

## 1) Purpose

`XK-203` closes the Obsidian lane quality boundary by requiring two fail-closed controls:

1. **Freshness SLO gate** for Obsidian knowledge surfaces used by retrieval workflows.
2. **Retrieval abstain-quality gate** with deterministic PASS/BLOCK semantics and threshold provenance.

This slice is runtime/evidence focused and builds on:
- `XK-201` lane boundary contract,
- `XK-202` deterministic materialization contract,
- formal-layers fold-in requirements for reasoning/eval/learning compatibility.

---

## 2) Canonical inputs

- `docs/ops/obsidian_knowledge_lane_contract_v1.md`
- `docs/ops/obsidian_to_shared_memory_materialization_contract_v1.md`
- `ops/obsidian/retrieval_eval.py`
- `ops/obsidian/retrieval_search.py`
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XK-203` entry)
- `reports/true_expanded_roadmap_formal_layers_foldin_2026-03-28.md`

---

## 3) Freshness SLO gate (fail-closed)

`XK-203` freshness gate evaluates these canonical surfaces:

1. `state/cron_watchdog/obsidian_hourly_canary_input_hash.json`
2. `memory/obsidian_export_chunks/INDEX.json`
3. `state/continuity/latest/xk_203_vault_validate_strict_runtime_2026-03-28.json`

Thresholds:
- `canary_hash_max_age_seconds = 7200`
- `obsidian_index_max_age_seconds = 86400`
- `vault_validate_max_age_seconds = 86400`

Decision rule:
- PASS only when all required surfaces exist and all ages are within SLO.
- Otherwise BLOCK with explicit stale/missing surface reason.

---

## 4) Retrieval abstain-quality gate (fail-closed)

Runtime command class:
- `python3 ops/obsidian/retrieval_search.py ...`

Gate thresholds (`xk203-gate-thresholds-v1`):
- `doc_intent = policy`
- `min_top_score = 0.70`
- `min_margin = 0.02`
- `max_results = 20`

Decision rule:
- PASS (`ok=true`) only when threshold conditions are met.
- Otherwise deterministic abstain (`ok=false`, `reason="ABSTAIN"`).

Required evidence classes:
1. strict retrieval scorecard (`hit@k`, `MRR`, negative-query guard),
2. abstain threshold decision log (query-level PASS/BLOCK results),
3. threshold regression tests (positive/negative threshold sensitivity).

---

## 5) Formal-layers fold-in obligations

`XK-203` must preserve three fold-in obligations from queue patch guidance:

1. **Reasoning / verification**  
   Retrieval gate decisions are deterministic and threshold-backed (no narrative-only claims).

2. **Evaluation / benchmarking**  
   Strict scorecard must include metric thresholds and explicit gate decision.

3. **Learning / adaptation compatibility**  
   Threshold pack and subsequent adjustments are recorded in a calibration ledger with rationale and refs.

---

## 6) Promotion / eval compatibility constraints

`XK-203` retrieval quality controls are required to remain compatible with existing promotion/eval contracts:

- `docs/ops/promotion_protocol_contract_v1.md`
- `docs/ops/knowledge_review_approval_promotion_queue_v1.md`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`

Compatibility expectation:
- retrieval abstain/non-abstain discipline must not undercut fail-closed promotion/eval semantics,
- calibration ledger must preserve threshold provenance and decision explainability.

---

## 7) Required artifacts for strict DONE

- `state/continuity/latest/xk_203_freshness_snapshots_2026-03-28.json`
- `state/continuity/latest/xk_203_retrieval_gate_decisions_2026-03-28.json`
- `state/continuity/latest/xk_203_threshold_regression_tests_2026-03-28.json`
- `state/continuity/latest/xk_203_retrieval_strict_eval_scorecard_2026-03-28.json`
- `state/continuity/latest/xk_203_abstain_threshold_decision_log_2026-03-28.jsonl`
- `state/continuity/latest/xk_203_freshness_and_retrieval_calibration_ledger_2026-03-28.json`
- `state/continuity/latest/xk_203_freshness_retrieval_gate_validation_2026-03-28.json`
- `reports/xk_203_knowledge_freshness_slo_and_retrieval_gate_closeout_2026-03-28.md`

---

## 8) Validation entrypoints

- `python3 ops/obsidian/vault_validate.py --strict`
- `python3 ops/obsidian/retrieval_eval.py --golden state/continuity/latest/xk_203_retrieval_strict_eval_fixture_2026-03-28.json --strict --top-k 20`
- `python3 ops/obsidian/retrieval_search.py --query <query> --doc-intent policy --max-results 20 --min-top-score 0.70 --min-margin 0.02`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 9) Closeout condition for `XK-203`

`XK-203` is complete only when:

1. freshness snapshot PASS is published,
2. retrieval gate decision artifacts show deterministic thresholded PASS/BLOCK behavior,
3. strict eval scorecard and threshold regression packet both PASS,
4. calibration ledger is updated with threshold provenance,
5. source-of-truth map and queue slice state are updated with evidence refs,
6. no out-of-scope runtime claims are made beyond freshness/retrieval gate depth.
