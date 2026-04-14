# Model Qualification Scorecard v1

Date: 2026-03-30  
Status: active (Program C / C2 baseline)  
Parent artifacts:
- `docs/ops/schemas/model_qualification_packet.schema.json`
- `docs/ops/templates/model_qualification_packet.template.json`
- `docs/ops/model_routing_doctrine_v1.md`
- `docs/ops/model_qualification_rollout_gate_contract_v1.md`

## 1) Purpose

Define a reusable, evidence-linked scorecard used to qualify or hold model promotion requests.

This scorecard makes promotion decisions evidence-based across five required dimensions:
1. latency
2. cost
3. trust
4. determinism
5. failure patterns

The scorecard is required in `model_qualification_packet` (`scorecard` object) and is used as an explicit promotion gate input.

---

## 2) Source posture (archive-mined + current)

This baseline folds in:
- `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md` (C2 deliverable and done-condition)
- `reports/openclaw_missing_layers_system_roadmap_foldin_2026-03-30.md` (S3 model-routing/qualification layer)
- `reports/multimodel_roster_utilization_audit_2026-03-29.md` (misroutes, cost telemetry blind spots, deterministic-boundary issues)
- `reports/multimodel_roster_audit_fold_in_2026-03-29.md` (policy fold-in guardrails)
- `reports/openclaw_current_model_routing_matrix_2026-03-26.md` (operational lane defaults)
- `reports/model_lineup_benchmark_2026-03-21.txt` (qualitative lane fit)
- `docs/ops/model_pool_policy_v1.json` (budgets, allowlists, authority matrix)
- `docs/ops/model_routing_no_llm_matrix_v1.md` (deterministic first + hard gates)
- `docs/ops/model_qualification_rollout_gate_contract_v1.md` (existing qualification checklist + benchmark minima)

---

## 3) Scoring model

## 3.1 Weighted total

Weighted score is computed on a 0-100 scale:

`weighted_score = Σ(dimension_score * weight)`

Default weights (must sum to 1.0):
- latency: `0.20`
- cost: `0.20`
- trust: `0.25`
- determinism: `0.25`
- failure patterns: `0.10`

## 3.2 Rating bands

- `>= 90`: strong
- `80-89.99`: acceptable with targeted fixes
- `70-79.99`: fragile / hold by default
- `< 70`: reject

Weighted score alone is insufficient: guardrails below are fail-closed.

---

## 4) Required dimensions and core metrics

## 4.1 Latency

Required metrics:
- `p50_ms`
- `p95_ms`
- `p99_ms`
- `timeout_rate`
- `score_0_100`
- `evidence_ref`

Interpretation:
- prioritize `p95/p99` and timeout stability over median-only gains.
- capture route-class context (`SPARK` vs `HEAVY`) to avoid false equivalence.

## 4.2 Cost

Required metrics:
- `effective_cost_per_1k_input_usd`
- `effective_cost_per_1k_output_usd`
- `daily_projected_cost_usd`
- `provider_cost_coverage_rate`
- `score_0_100`
- `evidence_ref`

Interpretation:
- cost decisions are invalid when provider coverage is too low (historical blind spot in roster audit).
- use provider-normalized telemetry, not raw zero-cost session rows.

## 4.3 Trust

Required metrics:
- `evidence_pointer_resolution_rate`
- `unsupported_claim_rate`
- `abstain_f1`
- `operator_override_rate`
- `score_0_100`
- `evidence_ref`

Interpretation:
- trust is an output-quality and operator-behavior signal, not just model reputation.
- high override rate means operational trust debt even if benchmark metrics look good.

## 4.4 Determinism

Required metrics:
- `replay_match_rate`
- `structured_output_stability_rate`
- `idempotent_retry_match_rate`
- `score_0_100`
- `evidence_ref`

Interpretation:
- deterministic replay and retry stability are mandatory for safe rollout.
- noisy nondeterminism in structured outputs blocks promotion in control-adjacent lanes.

## 4.5 Failure patterns

Required structure:
- array of observed failure objects (`pattern_id`, `severity`, `observed_rate`, `detectability`, `mitigation`, `status`)

Interpretation:
- a model is not “qualified” unless known failure modes are enumerated and bounded.
- failure inventory must include both quality failures and operational failures (timeouts, cost-coverage gaps, misroute behavior).

---

## 5) Fail-closed guardrails (hard gates)

Promotion recommendation must be `hold` or `reject` if any guardrail fails:
- `evidence_pointer_resolution_rate < 1.0`
- `unsupported_claim_rate > 0.02`
- `abstain_f1 < 0.95`
- `replay_match_rate < 0.98`
- `idempotent_retry_match_rate < 0.98`
- `provider_cost_coverage_rate < 0.90`
- any `failure_patterns` entry with:
  - severity `critical` and status `open`, or
  - severity `high` with no mitigation plan

Additionally, for `promotion_recommendation=promote`, `guardrail_violations` must be empty.

---

## 6) Decision mapping

- **PROMOTE**
  - weighted score `>= 85`
  - all guardrails pass
  - no unresolved high/critical failure pattern risk
- **HOLD**
  - weighted score `70-84.99`, or
  - guardrails violated but recoverable with bounded fixes
- **REJECT**
  - weighted score `< 70`, or
  - unresolved severe failures that cannot be safely mitigated in current lane

`summary.promotion_recommendation` is advisory; rollout authority still follows lane/policy contracts.

---

## 7) Packet binding contract

Scorecard is embedded under:
- `model_qualification_packet.scorecard`

Required nested sections:
- `window`
- `latency`
- `cost`
- `trust`
- `determinism`
- `failure_patterns`
- `summary`

`summary.weights` defines weighting used for `weighted_score_0_100` and must be explicit in every packet.

---

## 8) Operational cadence

For each model/lane qualification event:
1. run checklist + benchmark validation (`qualification` block)
2. compute scorecard dimensions with evidence refs
3. document open failure patterns and mitigation owner
4. set recommendation (`promote|hold|reject`)
5. submit packet to rollout gate review

Re-score at least on:
- route-class change,
- major provider/model version change,
- repeated incident/failure spikes,
- cost telemetry reliability regressions.

---

## 9) C2 completion statement

This v1 scorecard baseline satisfies Program C / C2 requirement for explicit scorecard fields across latency, cost, trust, determinism, and failure patterns, and binds those fields to the canonical model qualification packet contract.
