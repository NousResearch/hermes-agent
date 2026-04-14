# Source-of-Truth and Subagent Bootstrap Doctrine (v1)

Date: 2026-03-20  
Status: active (canonical workflow doctrine)  
Scope: OpenClaw system-upgrade work in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

This doctrine makes roadmap execution deterministic by defining:
- what to read first,
- what is canonical vs support vs historical,
- how to use the source-of-truth map,
- and how to update source-of-truth metadata without drift.

This is workflow doctrine, not a replacement for technical lane specs.

---

## 2) Mandatory bootstrap sequence (for main work and subagents)

Before proposing or executing non-trivial work slices:

1. **Scope guard first**
   - Confirm task is OpenClaw core system upgrade work.
   - Exclude unrelated downstream/domain work unless explicitly requested.

2. **Load canonical bootstrap set (in this exact order)**
   - `reports/openclaw_system_source_of_truth_map_2026-03-20.md`
   - `docs/ops/source_of_truth_and_subagent_bootstrap_doctrine_v1.md`
   - `reports/openclaw_full_roadmap_2026-03-20.md`
   - `reports/openclaw_full_roadmap_execution_table_2026-03-20.md`
   - `docs/ops/unified_operating_doctrine_v1.md`

3. **Load canonical resume/freshness gate**
   - `docs/ops/verify_before_resume_gate_checklist_v1.md`

4. **Load only lane-specific specs from the source-of-truth map**
   - Read only specs/files mapped to the target lane(s).
   - Avoid broad speculative reading outside lane boundaries.
   - For core-roadmap execution queue/executor slices (A2/A6/C1 surfaces), include:
     - `docs/ops/core_roadmap_queue_layer_doctrine_v1.md`
     - `docs/ops/subagent_slot_fill_protocol_v1.md`
     - `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`
     - `docs/ops/execution_meaningful_event_reporting_checklist_v1.md`
     - `ops/openclaw/architecture/schemas/core_roadmap_queue_layer.schema.json`

5. **Optional context only after canonical set is loaded**
   - Use support/audit docs if needed for rationale or gap confirmation.

6. **Classify execution shape before spawning work**
   - Use `docs/ops/model_routing_no_llm_matrix_v1.md` to classify `task_class`, `risk_tier`, `scope_shape`, and `verification_class`.
   - Decide `worker_topology` explicitly: `single`, `parallel_fanout`, or `staged_serial`.
   - For non-trivial slices, main lane is orchestration-only and default worker lane is `subagent_default`.
   - `main_session_tiny_exception` requires explicit low-risk bounded justification (`delegation_basis`) and cannot be used for fan-out/stale-supervision work.
   - Declare fold-in target up front: `canonical_doctrine`, `roadmap_pair`, `queue_continuity`, or `support_only`.
   - Declare stale/closeout posture up front: expected stale decision path and closeout bundle evidence target.

---

## 3) Canonical vs support vs historical classification rules

## Canonical (decision authority)
Use these to make decisions and define execution order:
- roadmap pair (`openclaw_full_roadmap_2026-03-20.md` + execution table)
- source-of-truth map
- unified operating doctrine
- lane specs explicitly listed in the source-of-truth map

## Support (context/evidence)
Use for evidence, audits, and implementation notes; do not override canonical docs:
- gap audit, mature inventory checklist, progress reports
- B6 runtime assignment snapshots (for example `reports/openclaw_current_model_routing_matrix_2026-03-26.md`) remain support-only unless explicitly promoted into canonical policy/spec surfaces

## Historical (reference only)
Do not treat as current authority:
- 2026-03-13 roadmap snapshots (`system_master_roadmap_2026-03-13.md`, `system_prioritized_roadmap_table_2026-03-13.md`)
- evaluation-era multipool/Kimi routing explorations (`docs/ops/model_routing_multipool_doctrine_v1.md`, `reports/kimi_k25_integration_synthesis_openclaw_2026-03-21.md`)

Rule: if canonical and historical disagree, canonical wins without debate.

---

## 4) How to use the source-of-truth map during execution

For each work slice:
1. Identify target lane in the map.
2. Pull:
   - canonical specs,
   - implementation files,
   - validation entrypoints,
   - operator surfaces.
3. Build slice contract against those references.
4. Validate with mapped tests/entrypoints before claiming completion.
5. Report evidence using mapped operator surfaces/artifacts.

If requested change spans multiple lanes, enumerate lanes explicitly and run per-lane validation.

---

## 5) Drift-avoidance rules

1. **No orphan specs**
   - New normative spec/schema/template must be attached to a lane in the map.

2. **No orphan implementation paths**
   - New runtime/script path that affects canonical behavior must be referenced in the map lane entry.

3. **No unverified claims**
   - Every claim of “implemented” must cite at least one mapped validation entrypoint result.

4. **No silent canon changes**
   - If roadmap priority/maturity changes, update canonical roadmap docs and map together.

5. **No historical re-promotion by accident**
   - Historical docs may be cited for context only; do not execute from them as authority.

---

## 6) Scope discipline for support research vs active implementation docs

- Support research can:
  - explain why a lane matters,
  - identify missing lanes/gaps,
  - suggest sequencing improvements.
- Support research cannot:
  - redefine canonical priority,
  - replace lane specs,
  - bypass lane-scoped validation.

Use support docs to inform; use canonical docs to decide and execute.

---

## 7) Source-of-truth update protocol

Update requirements when system truth changes:

1. **Lane mapping changes**
   - Update `reports/openclaw_system_source_of_truth_map_2026-03-20.md` in the same slice.

2. **Roadmap authority changes**
   - Update both canonical roadmap docs:
     - `reports/openclaw_full_roadmap_2026-03-20.md`
     - `reports/openclaw_full_roadmap_execution_table_2026-03-20.md`

3. **Workflow/bootstrap changes**
   - Update this doctrine and any direct bootstrap pointers in:
     - `AGENTS.md`
     - `docs/ops/unified_operating_doctrine_v1.md`

4. **Historical labeling discipline**
   - Keep old snapshots explicitly marked historical; do not rewrite them into fake-canonical docs.

---

## 8) Practical default for new subagents

Subagent kickoff packets should include:
- target lane(s),
- canonical map path,
- canonical roadmap pair,
- required lane specs,
- required validation entrypoints,
- explicit scope guard (OpenClaw system upgrade only).

For intake-heavy work (docs/PDF/research bundles), include additionally:
- absolute paths for all inbound materials,
- expected output artifact path,
- required output structure (per-item value, cross-cutting themes, lane impact, promote now/later/reference),
- explicit instruction to propose **minimal** canonical edits only when low-risk and clearly useful.

This keeps execution fast while preserving Column A momentum and full-system roadmap fidelity.

---

## 9) Default intake doctrine for new user-sent docs/PDFs (subagent-first)

When the user sends new documents/PDFs, default behavior is:

1. **Main/control lane orchestrates only**
   - Set scope guard and target lanes.
   - Delegate intake/analysis/integration to a subagent lane.

2. **Subagent lane executes intake**
   - Read/analyze the full batch.
   - Classify findings into `promote-now`, `promote-later`, `reference-only`.
   - Map each promoted recommendation to canonical artifacts (roadmap/table/map/doctrine/spec).

3. **Canonical edits are minimal and explicit**
   - Preserve roadmap shape.
   - Prefer wording/contract sharpening over structural churn.
   - Skip edits that are speculative, redundant, or high-risk.

4. **Required closeout evidence**
   - One synthesis note in `reports/`.
   - Clear list of files created/updated.
   - Concise list of integrations promoted now vs deferred.
   - One closeout packet conforming to `docs/ops/schemas/document_intake_batch_integration.schema.json` and validated via `scripts/document_intake_batch_integration_gate.py` (or `continuity.sh doc-intake-closeout`).

Exception rule: only handle docs/PDFs in main lane directly when trivially small and non-architectural; otherwise keep subagent-first default.
