# Promotion Trace Manifest Contract v1

Date: 2026-03-21  
Status: active (Wave 7 artifact/evidence hardening)

## Purpose
Require a deterministic manifest for each successful promotion queue decision so promoted outputs remain traceable end-to-end.

## Canonical contract
- Schema: `docs/ops/schemas/promotion_trace_manifest.schema.json`
- Template: `docs/ops/templates/promotion_trace_manifest.template.json`
- Runtime producer: `scripts/knowledge_promotion_queue.py` (on successful `promote`)

## Required chain
Each manifest must include:
1. promotion + queue identity (`promotion_id`, `queue_entry_id`),
2. candidate artifact path + hash,
3. workflow decision artifact path + hash + decision,
4. verified source refs with declared vs observed hashes,
5. decision refs,
6. optional shared-memory object ref,
7. implementation refs when available.

## Fail-closed rule
If promotion gate PASS is reached but trace manifest generation fails, queue promotion must fail-closed (`BLOCKED`) with explicit block reason.

## Operational location
Default path:
- `state/continuity/knowledge_promotion_queue/manifests/<queue_entry_id>.json`

## Scope boundary
This contract governs promoted knowledge artifacts. It does not replace broader continuity evidence bundles in Column A; it extends cross-system provenance discipline into B1/B2 promotions.

## XR-007 promoted asset checklist (B5 canonical promotion pack)

Archive/runtime asset promoted to canonical B5 checklist:

1. **Operator evidence trace viewer projection (`evidence_trace_viewer`)**
   - Runtime surface: `state/continuity/latest/evidence_trace_viewer_latest.json`
   - Slice evidence: `reports/core_roadmap_slice29_strict_closeout_2026-03-28.md`
   - Mission-control producer: `ops/openclaw/continuity/operator_mission_control.sh`
   - Required verification refs:
     - `tests/test_operator_mission_control_evidence_trace_viewer.py`
     - `tests/test_operator_surface_schema_failclose.py`

Promotion rule (fail-closed): B5 explainability claims are canonical only when the viewer artifact is emitted and mission-control schema checks pass with deterministic trace projection.
