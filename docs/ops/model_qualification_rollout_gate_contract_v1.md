# Model Qualification + Rollout Gate Contract v1

Date: 2026-03-20  
Status: active (bounded Wave 5 contract)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## Purpose
Define a fail-closed, deterministic first version of model qualification + staged rollout governance.

Scope in v1:
- qualification checklist contract
- lane-based authority matrix for rollout control
- staged rollout states (`shadow`, `canary`, `ring`)
- rollback + kill-switch semantics
- deterministic gate/check decision artifact

## Non-goals (v1)
- no provider-specific runtime refactor
- no automatic model swapping in production call paths
- no scheduler/watchdog rewrite

## Canonical source alignment
This contract is aligned to current Column A and model-routing doctrine plus prior strategy/adoption research artifacts:
- `docs/ops/unified_operating_doctrine_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/model_routing_no_llm_matrix_v1.md`
- `memory/inbound_pdfs/hl_terminal_research_pack_2026-03-03/LLM Strategy for HL Terminal and OpenClaw.txt`
- `multi_model_and_communication_architecture_prompts.txt` (Prompt 1 + Prompt 2 requirement set)

## Column-A preservation (explicit)
This contract preserves Column A invariants:
1. Deterministic gate runner is authority for pass/block, not narrative.
2. Unknown/missing/unavailable checks block (fail-closed).
3. Promotion to wider rollout is stepwise and reversible.
4. Kill-switch path is explicit and higher-authority gated.

## Machine-readable artifact contract
Schema:
- `docs/ops/schemas/model_qualification_packet.schema.json`

Template:
- `docs/ops/templates/model_qualification_packet.template.json`

Deterministic gate runner:
- `scripts/model_rollout_gate_runner.py`

## Qualification checklist (required checks)
All checks below are mandatory in v1 and must be `pass`:
1. `schema_contract_valid`
2. `tool_compatibility_validated`
3. `evidence_pointer_resolution`
4. `abstention_behavior_validated`
5. `fallback_route_defined`
6. `cost_budget_guard_defined`
7. `failure_mode_reviewed`
8. `rollback_playbook_linked`

## Benchmark gate thresholds (v1)
Required minimums:
- `json_valid_rate >= 0.995`
- `evidence_pointer_resolution_rate = 1.0`
- `abstain_f1 >= 0.95`
- `non_abstain_recall >= 0.95`

These map to existing no-LLM/model-routing hard gate doctrine and safe-adoption guidance.

## Lane-based authority matrix (v1)
Rollout authority is lane-scoped and fail-closed.

### Lane capabilities
- `lane.column_c.upgrade_substrate`
  - promote up to: `SHADOW`
  - rollback: allowed
  - kill-switch engage: not allowed
- `lane.column_b.swarm_orchestration`
  - promote up to: `RING_1`
  - rollback: allowed
  - kill-switch engage: not allowed
- `lane.column_a.no_nudge_autonomy`
  - promote up to: `FULL`
  - rollback: allowed
  - kill-switch engage: allowed

Unknown lane IDs are denied.

## Staged rollout states (v1)
Ordered rollout states:
- `DRAFT`
- `QUALIFIED`
- `SHADOW`
- `CANARY`
- `RING_1`
- `RING_2`
- `FULL`

Action model:
- `promote`: must move exactly one forward step.
- `rollback`: must move to a lower state.
- `kill`: must set requested state to `KILLED`.

## Rollback semantics (v1)
Rollback requires:
- explicit `rollback.reason`
- `rollback.fallback_model_ref`
- `rollback.max_rollback_minutes` (bounded, >0)
- owner role recorded

Decision semantics:
- successful rollback gate => final decision state `ROLLED_BACK`
- rollback never depends on provider-specific features

## Kill-switch semantics (v1)
Kill-switch object is mandatory and fail-closed:
- `kill_switch.armed` must be `true` for `CANARY` and wider states.
- `kill_switch.engage_requested=true` requires action `kill` and requested state `KILLED`.
- kill-switch engage is only authorized by `lane.column_a.no_nudge_autonomy` in v1.

Decision semantics:
- successful kill gate => final decision state `KILLED`

## Deterministic gate order (v1)
The runner evaluates gates in this exact order:
1. schema
2. source_refs
3. qualification_checklist
4. benchmark_thresholds
5. lane_authority
6. rollout_transition
7. rollback_killswitch

After first failure, remaining gates are marked `skipped`.

## Fail-closed behavior summary
Block conditions include:
- invalid JSON/schema
- missing source refs / unresolved hash mismatch
- missing/failed checklist checks
- benchmark threshold miss
- lane authority mismatch
- illegal transition
- missing rollback/kill-switch safety fields
- unavailable validator dependencies

## Decision artifact
Runner output schema:
- `clawd.model_rollout_gate.decision.v1`

Decision log:
- append-only JSONL, default path:
  - `state/continuity/model_rollout_gate_runner/decisions.jsonl`

Output includes:
- gate-by-gate pass/fail/skipped rows
- block gate + block reason
- candidate/packet path + sha256
- policy snapshot (thresholds/authority matrix)
- append status of decision record

## Operational usage
- Evaluate packet:
  - `python3 scripts/model_rollout_gate_runner.py --packet <path> --json`
  - optional explicit policy override: `--pool-policy <policy.json> --pool-policy-schema <schema.json>`
- Compatible alias:
  - `python3 scripts/model_rollout_gate_runner.py --candidate <path> --json`
- Disable logging when needed:
  - `python3 scripts/model_rollout_gate_runner.py --packet <path> --no-decision-log --json`

## Bounded adoption checklist
- [ ] Packet JSON validates schema.
- [ ] Mandatory checklist checks are all `pass`.
- [ ] Benchmarks satisfy thresholds.
- [ ] Requested lane has authority for action/state.
- [ ] Transition shape is legal.
- [ ] Rollback + kill-switch fields are complete.
- [ ] Gate runner decision is `PASS` before any rollout-state promotion.

## Deferred (intentional)
- no direct wiring into live model provider execution paths
- no automatic ring advancement scheduler
- no dual-approval quorum logic (can be v2)

## Queue-integrated rollout ledger/controller (Wave 5 Slice 2)
- runtime: `scripts/model_rollout_ledger_controller.py`
- input queue: `state/continuity/model_rollout_gate_runner/decisions.jsonl`
- rollout ledger: `state/continuity/model_rollout_ledger/ledger.jsonl`
- rollout events: `state/continuity/model_rollout_ledger/events.jsonl`
- health snapshot (required for ring dwell checks): `state/continuity/model_rollout_health/latest.json`
  - `schema_version=clawd.model_rollout_health.v1`
  - `generated_at` (UTC ISO8601)
  - `overall_status=healthy`
  - `rings.CANARY|RING_1|RING_2.slo_ok=true`
- dwell defaults: `CANARY=3600s`, `RING_1=7200s`, `RING_2=14400s`
- run (wrapper-dispatched mutating entrypoint): `bash ops/openclaw/continuity.sh model-rollout-controller --json`

## Deterministic health snapshot producer (Wave 5 Slice 3)
- runtime: `scripts/model_rollout_health_snapshot.py`
- source inputs (default):
  - `state/continuity/latest/continuity_now_latest.json`
  - `state/continuity/latest/verify_last.json`
  - `state/continuity/latest/gate_os_latest.json`
- output path (default): `state/continuity/model_rollout_health/latest.json`
- status rule (v1):
  - `overall_status=healthy` only when continuity ready, mutation gate allowed, verify status READY + fresh, and GateOS fail count is zero with mutation allowed.
  - otherwise `overall_status=unhealthy` with deterministic reason list mirrored into ring checks.
- run:
  - `python3 scripts/model_rollout_health_snapshot.py --json`
  - `bash ops/openclaw/continuity.sh model-rollout-health --json`

## Wave 5 Slice 5/6 maturity checkpoint
- unified pool policy is now canonicalized at `docs/ops/model_pool_policy_v1.json` (schema: `docs/ops/schemas/model_pool_policy.schema.json`) and enforced in gate/runtime paths.
- cost-governance telemetry snapshot producer is now canonicalized at `scripts/model_rollout_cost_governance_snapshot.py` (output: `state/continuity/model_rollout_cost/latest.json`).

## Wave 5 Slice 7+ closeout checkpoint (2026-03-21)
- shared pool-policy contract helpers are centralized in `scripts/model_pool_policy_contract.py` and reused in gate/router/cost runtimes.
- long-window route-policy soak/lint snapshot is canonicalized at `scripts/model_route_policy_soak_lint.py` (output: `state/continuity/model_route_policy_soak/latest.json`).
- ring-soak automation snapshot is canonicalized at `scripts/model_rollout_ring_soak_snapshot.py` (output: `state/continuity/model_rollout_soak/latest.json`).
- consolidated rollout operator dashboard snapshot is canonicalized at `scripts/model_rollout_dashboard_snapshot.py` (output: `state/continuity/model_rollout_dashboard/latest.json`).

## Next-wave focus
Wave 5 lane is closeout-ready; remaining maturity work belongs to Wave 6+ (multi-provider qualification depth, bakeoff governance, cockpit action UX integration).
