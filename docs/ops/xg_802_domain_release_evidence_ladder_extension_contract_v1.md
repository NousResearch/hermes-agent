# XG-802 Domain Release Evidence Ladder Extension Contract v1

Date: 2026-03-28  
Status: active (canonical XG-802 release-governance extension)  
Owner: Architect  
Scope: Internal domain-lane release evidence and approval-gating extension only (no public packaging)

---

## 1) Purpose

`XG-802` extends the baseline C2 release evidence ladder so high-risk domain lanes (`XP/XT/XH/XB`, and any future XG-governed lane) cannot activate sensitive behavior without:

1. tiered authorization metadata,
2. explicit human-in-the-loop approval ladder evidence,
3. attribution-complete release artifacts,
4. internal benchmark/quality threshold evidence,
5. rollback readiness proof.

This slice is governance + evidence extension only.
It does **not** enable public/open-source packaging workflows.

---

## 2) Canonical dependencies

- `docs/ops/c3_activation_governance_contract_v1.md` (`XG-801` foundation)
- `docs/ops/release_evidence_ladder_contract_v1.md` (baseline ladder)
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/true_expanded_out_of_core_scope_filter_rules_v1.md`

Queue authority:
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XG-802` entry)
- Formal-layers fold-in authority:
  - `reports/true_expanded_roadmap_formal_layers_foldin_2026-03-28.md`
- External-research fold-in authority:
  - `reports/openclaw_systems_external_research_implication_pack_2026-03-28.md`

---

## 3) Contracted extension artifacts

### 3.1 Schema/template
- Schema: `docs/ops/schemas/domain_release_evidence_bundle.schema.json`
- Template: `docs/ops/templates/domain_release_evidence_bundle.template.json`

### 3.2 Validation/runtime entrypoint
- `python scripts/release_evidence_ladder_gate.py --bundle <bundle.json> --json`
- `python scripts/domain_release_evidence_extension_gate.py --bundle <bundle.json> --json`

The baseline release ladder gate remains mandatory.  
The domain extension gate is additional and fail-closed.

---

## 4) Domain extension requirements (normative)

A domain release evidence bundle is valid only if all rules pass:

1. **Baseline ladder pass**
   - Stage order, stage coverage, evidence refs, rollback recency, compatibility lifecycle, and A6 observability gates all pass.

2. **Internal-only boundary**
   - `internal_release_profile.public_packaging_in_scope` must be `false`.
   - Any `true` value is hard block (`public_packaging_scope_violation`).

3. **Auth-tier metadata completeness**
   - Bundle must declare exactly one `auth_tier` from:
     - `ADMIN`, `OBSERVABILITY`, `INTERNAL`, `PUBLIC`.
   - Bundle must declare one `risk_class` from `RG0_LOW|RG1_MODERATE|RG2_HIGH|RG3_CRITICAL`.

4. **Sensitive-route approval ladder**
   - If `sensitive_action=true`, bundle must include at least one `approval_ladder.steps[*]` row with `status=pass` and explicit `approver_id`.
   - `approval_ladder.required=true` for sensitive routes.

5. **Attribution-complete release artifact**
   - `attribution` must include non-empty:
     - `requested_by`, `approved_by`, `executed_by`, `decision_log_ref`.

6. **Internal quality/benchmark threshold evidence**
   - `internal_release_profile.benchmark_thresholds` must be non-empty.
   - Every benchmark row must be `status=pass` for canary/progressive/broad modes.

7. **Rollback readiness evidence**
   - `internal_release_profile.rollback_readiness.status` must be `pass`.
   - At least one rollback readiness proof ref must be present.

---

## 5) Fail-closed block reasons (canonical for XG-802)

- `domain_schema_invalid`
- `domain_schema_gate_unavailable`
- `public_packaging_scope_violation`
- `auth_tier_missing_or_invalid`
- `risk_class_missing_or_invalid`
- `approval_ladder_missing_for_sensitive_action`
- `approval_ladder_no_pass_step`
- `approval_step_approver_missing`
- `attribution_incomplete`
- `benchmark_threshold_failed`
- `rollback_readiness_not_pass`
- `proof_ref_unresolved`

---

## 6) Closeout criteria for XG-802

`XG-802` is complete only when:

1. this contract is canonicalized,
2. domain bundle schema/template are published,
3. pass evidence exists for baseline release ladder + domain extension gate,
4. sample attribution-complete, auth-tiered, sensitive-action bundle is produced,
5. benchmark threshold and rollback readiness evidence artifacts are produced,
6. source-of-truth map references XG-802 artifacts,
7. queue-layer `XG-802` state is `DONE` with evidence refs,
8. no claims are made about `XG-803` completion.
