# XB-403 Downstream Eval and Release Gate Pack Contract v1

Date: 2026-03-29  
Status: active (canonical for `XB-403`)  
Owner: Architect  
Scope: deterministic downstream eval/replay/benchmark/rollback gate pack coupled to C2 release governance and XG domain governance

---

## 1) Purpose

`XB-403` closes the downstream release-depth gap by requiring one deterministic gate pack before any downstream domain release can be promoted.

The pack is fail-closed and requires all of the following to pass in one bounded decision:
1. baseline C2 release ladder gates,
2. XG-802 domain extension gates,
3. Wave2/Batch1-inspired deterministic replay fixture families,
4. domain benchmark scorecard thresholds,
5. rollback simulation and rollback-coupled gate decision evidence.

This slice does **not** implement capability registry runtime (`XB-402`).

---

## 2) Canonical dependencies

- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XB-403` queue truth)
- `docs/ops/downstream_capability_backend_contract_v1.md` (`XB-401`)
- `docs/ops/release_evidence_ladder_contract_v1.md` (C2 baseline)
- `docs/ops/xg_802_domain_release_evidence_ladder_extension_contract_v1.md` (domain extension)
- `reports/wave2_regression_replay_harness_spec_2026-03-20.md`
- `reports/wave2_regression_replay_harness_tasklist_2026-03-20.md`
- `reports/batch1_forward_slice_gtc_runtime_regression_harness_2026-03-10.md`

---

## 3) Canonical artifacts (normative)

### 3.1 Gate-pack contract/schema/template
- `docs/ops/xb_403_downstream_eval_and_release_gate_pack_contract_v1.md`
- `docs/ops/schemas/xb_403_downstream_eval_gate_pack.schema.json`
- `docs/ops/templates/xb_403_downstream_eval_gate_pack.template.json`

### 3.2 Runtime gate implementation
- `scripts/xb_403_downstream_eval_release_gate.py`

### 3.3 Required evidence objects
- pack descriptor (`clawd.xb_403_downstream_eval_gate_pack.v1`)
- deterministic replay fixture results (`clawd.xb_403.deterministic_replay_fixture_results.v1`)
- domain benchmark scorecard (`clawd.xb_403.domain_benchmark_scorecard.v1`)
- rollback simulation report (`clawd.xb_403.rollback_simulation_report.v1`)
- gate decision output (`clawd.xb_403_downstream_eval_release_gate.decision.v1`)

---

## 4) Gate rules (fail-closed)

A decision is `PASS` only if every gate passes in order:

1. **Pack schema gate**
   - Gate pack descriptor validates against `xb_403_downstream_eval_gate_pack.schema.json`.

2. **C2 baseline release ladder gate**
   - `release_evidence_ladder_gate.evaluate_bundle(...)` returns `PASS` for the referenced domain release bundle.

3. **XG-802 domain extension gate**
   - `domain_release_evidence_extension_gate.evaluate(...)` returns `PASS` for the same bundle.

4. **Deterministic replay fixtures gate**
   - replay artifact decision is `PASS`;
   - fixture families `F8`, `F9`, `F10`, `F11`, `F12`, and `FX` are present and each `status=pass`;
   - negative-regression checks preserve expected fail-close outcomes.

5. **Domain benchmark scorecard gate**
   - scorecard decision is `PASS`;
   - metrics list is non-empty;
   - every metric row has `status=pass`.

6. **Rollback simulation gate**
   - rollback report decision is `PASS`;
   - rollback scenarios are non-empty and all `status=pass`;
   - rollback report references both release and domain gate decision outputs.

7. **Cross-artifact release-id parity gate**
   - `release_id` must match across release bundle, replay results, benchmark scorecard, rollback simulation report, and pack descriptor `expected_release_id` (if provided).

If any gate fails, remaining gates are marked `skipped` and the decision is `BLOCK`.

Canonical fail reasons:
- `pack_schema_invalid`
- `pack_schema_gate_unavailable`
- `pack_ref_unresolved`
- `release_ladder_blocked`
- `domain_extension_blocked`
- `replay_fixture_blocked`
- `benchmark_scorecard_blocked`
- `rollback_simulation_blocked`
- `release_id_mismatch`

---

## 5) Validation entrypoints

- `pytest -q tests/test_xb_403_downstream_eval_release_gate_pack.py`
- `python -m json.tool docs/ops/schemas/xb_403_downstream_eval_gate_pack.schema.json`
- `python -m json.tool docs/ops/templates/xb_403_downstream_eval_gate_pack.template.json`
- `python scripts/xb_403_downstream_eval_release_gate.py --pack state/continuity/latest/xb_403_downstream_eval_gate_pack_2026-03-29.json --json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root /home/yeqiuqiu/clawd-architect --map-path /home/yeqiuqiu/clawd-architect/reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 6) Closeout criteria for `XB-403`

`XB-403` is complete only when:
1. contract + schema/template + gate runtime are landed,
2. deterministic replay fixture results, benchmark scorecard, rollback simulation report, and gate decision artifacts exist,
3. gate decision is `PASS` and fail-closed semantics are tested,
4. source-of-truth map includes XB-403 references,
5. queue-layer `XB-403` state is transitioned to `DONE` with explicit evidence refs and status reason.