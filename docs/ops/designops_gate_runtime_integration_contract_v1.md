# DesignOps Gate Runtime Integration Contract v1 (`XD-103`)

Date: 2026-03-29  
Status: active (canonical `XD-103` runtime integration contract)  
Owner: Architect

---

## 1) Purpose

`XD-103` wires the DesignOps gate stack (`G1..G6`) into live release-governance and continuity surfaces.

This contract ensures that design-lane release bundles fail-close unless all required design gates are explicitly reported with resolved evidence.

---

## 2) Canonical dependencies

- `docs/ops/designops_lane_contract_v1.md` (`XD-101`)
- `docs/ops/designops_schema_pack_migration_notes_v1.md` (`XD-102`)
- `docs/ops/release_evidence_ladder_contract_v1.md` (C2 baseline gate)
- `docs/ops/xg_802_domain_release_evidence_ladder_extension_contract_v1.md` (dependency status source)
- `ops/openclaw/architecture/ui_design_edd.v1.yaml` (gate semantics source)

Queue truth authority:
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json#XD-103`

---

## 3) Runtime integration surface

`XD-103` extends `scripts/release_evidence_ladder_gate.py` with gate `design_gate_stack`.

### 3.1 Bundle contract extension

Release evidence bundles may include:
- `design_gate_stack.schema_version = clawd.design_gate_stack_result.v1`
- `design_gate_stack.ordered_results[]` with exactly six rows in canonical order:
  1. `G1_SCHEMA`
  2. `G2_STRUCTURE`
  3. `G3_A11Y`
  4. `G4_VISUAL`
  5. `G5_RUNTIME`
  6. `G6_ALIGNMENT`

Each row must include:
- `gate_id`
- `status` (`pass|block`)
- `evidence_refs[]`
- `evaluated_at`

### 3.2 Applicability

- If `lane_context.lane_id` resolves to DesignOps (`XD` / `lane.designops*`), `design_gate_stack` is mandatory.
- For non-design lanes, the gate is non-blocking when absent.
- If present for any lane, it is validated fail-close.

### 3.3 Fail-close reasons

- `design_gate_stack_missing`
- `design_gate_stack_invalid`
- `design_gate_stack_order_invalid`
- `design_gate_stack_evidence_missing`
- `design_gate_stack_not_pass`

---

## 4) Continuity surfaces

Runtime publishes DesignOps gate summaries to:

- Latest snapshot: `state/continuity/latest/xd_design_gate_runtime_latest.json`
- Append-only history: `state/continuity/design_governance/xd_design_gate_runtime_history.jsonl`

These surfaces must include release decision linkage (`release_id`, release decision, block gate/reason) and gate-stack details.

---

## 5) Validation entrypoints

- `python scripts/release_evidence_ladder_gate.py --bundle <bundle.json> --json`
- `pytest -q tests/test_release_evidence_ladder_gate.py`
- `bash ops/openclaw/architecture/validate_component_spec.sh --json`
- `bash ops/openclaw/architecture/validate_design_schema_pack.sh --json`

---

## 6) XD-103 closeout criteria

`XD-103` is complete only when:

1. DesignOps gate stack is integrated as a fail-closed runtime gate in release ladder evaluation.
2. Design-lane release bundles block when `design_gate_stack` is missing or any gate is non-pass.
3. Continuity snapshots/history for design gate runtime are emitted.
4. Regression tests cover pass + fail scenarios for design gate integration.
5. Queue truth `XD-103` is updated to `DONE` with evidence refs.
