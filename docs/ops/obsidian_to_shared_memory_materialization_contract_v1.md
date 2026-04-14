# Obsidian → Shared Memory Materialization Contract v1 (`XK-202`)

Date: 2026-03-28  
Status: active (canonical for `XK-202`)  
Owner: Architect  
Scope: deterministic Obsidian/research-case materialization into shared-memory compatible typed records

---

## 1) Purpose

This contract closes `XK-202` by defining a deterministic mapping from Obsidian/research-case artifacts into promotable shared-memory objects with:

1. immutable provenance anchors,
2. typed understanding-object projections,
3. promotion-gate compatibility,
4. conflict/demotion lifecycle compatibility.

`XK-202` does **not** set freshness/retrieval SLO thresholds (`XK-203`).

---

## 2) Canonical inputs

- `docs/ops/obsidian_knowledge_lane_contract_v1.md` (`XK-201` boundary)
- `docs/ops/shared_memory_fabric_lifecycle_contract_v1.md`
- `docs/ops/promotion_protocol_contract_v1.md`
- `docs/ops/knowledge_review_approval_promotion_queue_v1.md`
- `docs/ops/research_case_pipeline_mvp_2026-03-10.md`
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XK-202` slice contract)

---

## 3) Deterministic mapping profile

### 3.1 Source lineage

Allowed materialization lineage for `XK-202`:

1. research/Obsidian synthesis artifacts (`SYNTH/*.json|*.md`, governed lane notes),
2. promotion candidate (`promotion_candidate.json`) with review metadata,
3. promotion-gate decision (`PASS`) and queue entry id,
4. shared-memory materialization object (`smo_*`).

### 3.2 Typed understanding-object classes

`XK-202` canonicalizes these understanding classes for mapping:

- `claim`
- `principle`
- `framework`
- `tradeoff`
- `contradiction`
- `decision`

These are mapped deterministically into existing promotion/shared-memory enums:

| understanding_type | promotion `insight.kind` | shared-memory `memory.object_type` |
|---|---|---|
| `claim` | `fact` | `fact` |
| `principle` | `rule` | `rule` |
| `framework` | `procedure` | `procedure` |
| `tradeoff` | `heuristic` | `heuristic` |
| `contradiction` | `heuristic` | `heuristic` |
| `decision` | `rule` | `decision` |

### 3.3 Immutable provenance anchors (required)

Every materialized object MUST retain these immutable anchors:

1. `source_refs[].path` + `source_refs[].content_hash` (`sha256:*`),
2. `promotion.promotion_id` (`prom_*`),
3. `promotion.queue_entry_id` (`kpq_*`),
4. `promotion.candidate_sha256`,
5. `promotion.workflow_decision_sha256`.

Fail-closed rule: if any required anchor is missing/invalid, promotion blocks.

---

## 4) Promotion-gate compatibility requirements

For `target.surface=memory`, candidate must pass all six gates:

1. schema
2. provenance
3. confidence (>= memory threshold)
4. review (approved by allowed role)
5. leakage
6. publish traceability

`XK-202` compatibility is valid only when:

- gate decision is `PASS`,
- shared-memory promote succeeds,
- generated object validates against `shared_memory_object.schema.json`.

---

## 5) Conflict and demotion compatibility

`XK-202` materialized objects must support lifecycle compatibility with:

- `shared_memory_fabric.py conflict` (emit `smc_*`, transition to `CONFLICTED` on pending),
- `shared_memory_fabric.py demote` (emit `smd_*`, transition to deterministic demoted state).

No silent overwrite is allowed for contradictions.

---

## 6) Required artifacts for strict DONE

- mapping profile:
  - `state/continuity/latest/xk_202_research_case_to_memory_mapping_profile_2026-03-28.json`
- typed understanding-object samples:
  - `state/continuity/latest/xk_202_typed_understanding_object_samples_2026-03-28.json`
- promotion-gate compatibility packet:
  - `state/continuity/latest/xk_202_promotion_gate_compatibility_test_packet_2026-03-28.json`
- materialization sample artifacts:
  - `state/continuity/latest/xk_202_materialization_sample_artifacts_2026-03-28.json`
- closeout report:
  - `reports/xk_202_obsidian_to_shared_memory_materialization_contract_closeout_2026-03-28.md`

---

## 7) Validation entrypoints

- `python3 scripts/promotion_gate_runner.py --candidate <candidate.json> --repo-root /home/yeqiuqiu/clawd-architect --json`
- `python3 scripts/shared_memory_fabric.py --repo-root /home/yeqiuqiu/clawd-architect --json promote --candidate <candidate.json> --queue-entry-id <kpq_*> --workflow-decision-path <decision.json>`
- `python3 scripts/shared_memory_fabric.py --repo-root /home/yeqiuqiu/clawd-architect --json conflict ...`
- `python3 scripts/shared_memory_fabric.py --repo-root /home/yeqiuqiu/clawd-architect --json demote ...`

---

## 8) Closeout condition for `XK-202`

`XK-202` is complete only when:

1. deterministic mapping contract is canonicalized,
2. research-case → memory mapping profile exists,
3. typed understanding-object samples are published,
4. promotion-gate compatibility packet proves PASS materialization path,
5. conflict/demotion compatibility evidence is present,
6. queue slice `XK-202` is updated to `DONE` with evidence refs and bounded status reason.
