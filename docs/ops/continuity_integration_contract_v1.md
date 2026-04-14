# Continuity Integration Contract (v1)

Date: 2026-03-30  
Status: active (canonical continuity source-of-truth)  
Owner: Architect  
Scope: OpenClaw system upgrade continuity/ground-truth substrate (Program A / Slice A1)

---

## 1) Purpose

This contract unifies continuity primitives that were previously fragmented across reports, implementation notes, and lane-specific hardening slices.

This v1 contract defines one authoritative integration boundary for:
- handover + checkpoint continuity artifacts,
- context-overflow trigger semantics,
- deterministic ground-truth capture and freshness,
- verify-before-mutate / verify-before-resume gates,
- fail-closed recovery posture.

---

## 2) Authority and scope boundary

## 2.1 Canonical authority
This file is the canonical continuity integration authority for Program A continuity substrate integration.

Canonical machine-readable companions:
- `docs/ops/schemas/continuity_integration_contract.schema.json`
- `docs/ops/templates/continuity_integration_contract.template.json`

## 2.2 Scope-in
This contract governs integration semantics across continuity domains. It does **not** replace domain module contracts; it composes them.

## 2.3 Scope-out
This contract does not redefine:
- orchestrator API/event contracts (Program B),
- model routing qualification contracts (Program C),
- CSI artifacts (Program D).

---

## 3) Normative terms

- **MUST**: hard requirement.
- **SHOULD**: expected default unless explicitly justified.
- **MAY**: optional behavior.
- **Source input**: archived/historical/support material used to derive canonical clauses, but not active authority by itself.
- **Canonical artifact**: active contract/schema/template/runtime surface used for current decision and gate behavior.

---

## 4) Canonical continuity bundle (active)

The A1 bundle is exactly:
1. `docs/ops/continuity_integration_contract_v1.md` (this contract)
2. `docs/ops/schemas/continuity_integration_contract.schema.json`
3. `docs/ops/templates/continuity_integration_contract.template.json`

Runtime continuity surfaces this contract integrates (not replaced):
- `state/continuity/latest/handover_latest.json`
- `state/continuity/latest/current.json`
- `state/continuity/latest/ground_truth_latest.json`
- `state/continuity/latest/verify_last.json`

---

## 5) Integration clauses (authoritative)

### C1. Checkpoint + handover dual-surface contract
- Continuity state MUST be representable as machine-readable checkpoint/handover artifacts with deterministic required fields.
- Human-readable handover text MAY exist, but machine-readable artifacts are authoritative for resume/succession decisions.

### C2. Trigger policy (volumetric + semantic)
- Overflow handling MUST support volumetric thresholds and semantic trigger boundaries.
- Trigger classes MUST include at minimum:
  - pre-execution boundary,
  - post-mutation boundary,
  - post-scheduling-change boundary,
  - failure boundary,
  - explicit handover boundary.

### C3. Deterministic ground-truth capture
- Continuity decisions MUST anchor to host-captured truth (runtime/gateway/cron/process/git/session signals), not narrative memory.
- Ground-truth freshness MUST be checked before mutation resume.

### C4. Verify-before-mutate and verify-before-resume gates
- Mutation-capable paths MUST fail closed when required verification inputs are stale, missing, or contradictory.
- A resumed/successor flow MUST evaluate verification results before mutating actions.

### C5. Fail-closed continuity posture
- Missing/invalid/stale continuity inputs MUST block mutation by default.
- Degraded continuity states MUST emit explicit machine-readable reasons.

### C6. Lineage posture
- Continuity artifacts SHOULD preserve append-oriented lineage and latest-pointer semantics so successor resume is deterministic.

---

## 6) Canonical artifact map to archived continuity/ground-truth materials

The following inputs are explicitly reclassified as **source inputs** (historical/support/archive-mined) and are not active authority on their own.

| Source ID | Source artifact | Classification | Folded into clause(s) | Status in this contract |
|---|---|---|---|---|
| src_pdf_4d1ba541 | `Agent Context Overflow Prevention System` (`sha256:4d1ba5414fe314e580191a51d137932b7efacfffde38a4edc773c8872ed53b9f`) | archive-mined PDF | C1, C2, C4 | source_input_only |
| src_pdf_c34f592e | `AI Continuity Architecture Design` (`sha256:c34f592e30a08559f043e07a90de3cc81c8d8f52809865bef203805ecc00da45`) | archive-mined PDF | C1, C5, C6 | source_input_only |
| src_pdf_f1fc25fc | `Local Assistant Ground Truth Integration Plan` (`sha256:f1fc25fce940b647ddb182551a947fa0c11f5ebb13dd8475028b919022db0dc0`) | archive-mined PDF | C3, C4, C5 | source_input_only |
| src_pdf_da695344 | `Autonomous Upgrade Framework Design` (`sha256:da69534492b3512f001aba24274374ea20df625d9045aacbf959fda57cbff0fe`) | archive-mined PDF | C2, C4, C5 | source_input_only |
| src_rpt_fourpack_20260308 | `reports/upgrade_continuity_ground_truth_fourpack_synthesis_2026-03-08.md` | historical report | C1, C2, C3, C4 | source_input_only |
| src_rpt_checkpoint_blueprint_20260308 | `reports/continuity_checkpoint_os_blueprint_2026-03-08.md` | historical report | C1, C3, C6 | source_input_only |
| src_rpt_gt_connectors_20260308 | `reports/ground_truth_connectors_plan_2026-03-08.md` | historical report | C3, C5 | source_input_only |
| src_rpt_batch4_20260308 | `reports/p0_upgrade_batch4_continuity_buildout_2026-03-08.md` | historical implementation report | C4, C5 | source_input_only |
| src_rpt_batch5_20260308 | `reports/p0_upgrade_batch5_continuity_reconcile_2026-03-08.md` | historical implementation report | C3, C5 | source_input_only |
| src_rpt_batch6_20260308 | `reports/p0_upgrade_batch6_continuity_blocker_reconcile_unification_2026-03-08.md` | historical implementation report | C2, C5 | source_input_only |
| src_rpt_batch7_20260308 | `reports/p0_upgrade_batch7_continuity_verify_reconcile_history_2026-03-08.md` | historical implementation report | C4, C5, C6 | source_input_only |
| src_rpt_handover_proof_20260320 | `reports/successor_safe_handover_proof_contract_roadmap_integration_2026-03-20.md` | support roadmap integration report | C4, C5 | source_input_only |
| src_rpt_sot_map_20260320 | `reports/openclaw_system_source_of_truth_map_2026-03-20.md` | active canonical map | C1, C3, C4 | subordinate_active_contract |
| src_rpt_publish_lock_20260326 | `reports/continuity_current_publish_lock_ownership_slice_2026-03-26.md` | support/runtime evidence report | C5 | source_input_only |
| src_rpt_runtime_failclose_20260327 | `reports/continuity_current_runtime_failclose_boundary_2026-03-27.md` | support/runtime evidence report | C5 | source_input_only |
| src_rpt_recover_republish_20260327 | `reports/continuity_publish_recovery_then_republish_contract_2026-03-27.md` | support/runtime evidence report | C5, C6 | source_input_only |
| src_rpt_archive_mining_20260330 | `reports/openclaw_archive_mining_ground_truth_continuity_2026-03-30.md` | archive-mined synthesis report | C1, C2, C3, C4, C5, C6 | source_input_only |
| src_rpt_exec_queue_20260330 | `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md` | canonical queue directive | C1, C3, C4, C5 | subordinate_active_contract |

---

## 7) Roadmap fold-ins captured in v1

1. **A1 canonicalization fold-in**: one active continuity integration contract now exists.
2. **Archive-mined doctrine fold-in**: checkpoint/reset OS, append-oriented lineage, and deterministic ground-truth capture are integrated as explicit clauses.
3. **Runtime hardening fold-in**: verify/reconcile/fail-close/publish-lock evidence is integrated as policy constraints, not ad hoc report behavior.
4. **Historical reclassification fold-in**: prior continuity reports remain source inputs; they are not standalone active contracts.

---

## 8) Supersession and historical handling rule

- Historical continuity reports and archive syntheses remain valuable evidence and design input.
- They MUST NOT be treated as active authority when they conflict with this contract.
- Active subordinate contracts (for example queue state model, verify-before-resume checklist, source-of-truth map) remain in force within their module boundaries, but continuity integration precedence for Program A is defined here.

---

## 9) Change control

Any continuity integration change that alters canonical clauses C1–C6 MUST:
1. update this contract,
2. update schema/template if artifact shape changes,
3. preserve source-input classification discipline,
4. record fold-in/reclassification rationale in a dated report.

---

## 10) Acceptance criteria for A1

A1 is considered complete when:
- one active continuity integration contract exists (this file),
- machine-readable schema/template companions exist,
- archived continuity/ground-truth materials are explicitly mapped as source inputs,
- fragmented historical references are no longer treated as active continuity authority by default.
