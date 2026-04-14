# Operator UX Information Architecture v2 (`XU-501`)

Date: 2026-03-28  
Status: active (canonical foundation for XU lane IA/state model)  
Owner: Architect  
Scope: Frontend/operator UX productization lane (`XU-*`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

`XU-501` defines the canonical IA foundation for downstream operator UX surfaces beyond mission-control primitives.

It establishes:
- a cross-lane IA v2 navigation model,
- a unified state model for required expanded slices,
- evidence-source bindings for each operator view,
- and governance alignment to XG ownership/risk posture.

This slice is IA/state-contract only.
It does **not** implement lane action runtime (`XU-502`) or explainability panel runtime (`XU-503`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json`
- `docs/ops/cockpit_action_card_design_v1.md`
- `docs/ops/human_first_observability_v1.md`
- `docs/ops/low_noise_interaction_policy_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/c3_activation_governance_contract_v1.md`
- `state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`

### Canonical outputs for `XU-501`
- IA/state doctrine artifact (this file),
- machine-readable navigation/state model artifact,
- operator walkthrough evidence artifact,
- queue-layer closeout evidence.

Deferred to follow-on slices:
- `XU-502` composable lane action runtime,
- `XU-503` explainability/provenance panel runtime.

---

## 3) IA v2 navigation model (contract)

Operator UX IA v2 is a 4-layer navigation hierarchy:

1. **Program Overview**
   - Shows global expanded-queue posture and blocked/ready concentration.
   - Primary source: `true_expanded_roadmap_queue_layer.json`.

2. **Lane Board**
   - Shows one row per lane (`XD/XK/XP/XB/XU/XT/XH/XG/XR/XE/XO`) with state distribution and risk class.
   - Primary sources: queue layer + `xg_801` risk/owner artifacts.

3. **Slice Workspace**
   - Focused slice card: objective, dependencies, closeout condition, evidence refs, and next deterministic action class.
   - Primary source: queue-layer slice object (`id`, `state`, `dependencies`, `evidence_expectations`, `evidence_refs`).

4. **Evidence & Walkthrough Panel**
   - Read-only evidence trace for slice closeout narratives and operator runbook path.
   - Primary sources: closeout reports + lane evidence artifacts.

Low-noise requirements:
- no alert spam for passive `DONE` refresh events,
- immediate blocker cards for dependency/risk/ownership failures,
- remediation hints are mandatory for blocked transitions.

---

## 4) Unified state model (contract)

IA v2 uses one normalized state envelope for all expanded slices:

### 4.1 Canonical slice states
- `DONE`
- `READY`
- `DEPENDENCY_BLOCKED`
- `QUEUED_OPTIONAL`

### 4.2 Derived operator state classes
- `ACTIONABLE_NOW` = canonical `READY`
- `WAITING_ON_DEPENDENCY` = canonical `DEPENDENCY_BLOCKED`
- `CLOSED_VERIFIED` = canonical `DONE`
- `OPTIONAL_BACKLOG` = canonical `QUEUED_OPTIONAL`
- `FAIL_CLOSED_REVIEW` = any risk/ownership/scope integrity failure

### 4.3 State transition constraints
1. `DEPENDENCY_BLOCKED -> READY` requires all dependencies `DONE` in queue truth.
2. `READY -> DONE` requires closeout artifacts and explicit evidence refs.
3. Any missing risk/owner tuple for lane-level activation context maps to `FAIL_CLOSED_REVIEW`.
4. Unknown queue state or malformed slice payload fails closed.

### 4.4 Owner/risk overlays
For each lane card, IA v2 overlays:
- `risk_class` from `xg_801_c3_activation_risk_matrix_2026-03-28.json`,
- owner tuple (`governance_owner`, `lane_owner`, `release_owner`, `incident_owner`) from `xg_801_c3_activation_owner_registry_2026-03-28.json`.

This ensures XU views are governance-coupled before runtime action surfaces are introduced.

---

## 5) Evidence-source mapping matrix

| IA view | Required canonical source(s) | Fail-close condition |
|---|---|---|
| Program overview | `state/continuity/latest/true_expanded_roadmap_queue_layer.json` | Queue artifact missing or unparsable |
| Lane board | Queue layer + `xg_801` risk matrix + `xg_801` owner registry | Missing lane risk/owner tuple |
| Slice workspace | Queue-layer slice object + closeout report refs | Missing slice id/state/dependency list |
| Evidence/walkthrough panel | Slice `evidence_refs` + closeout report | Broken artifact path or unreadable closeout file |

---

## 6) Bounded non-goals for `XU-501`

Out of scope in this slice:
- no runtime UI renderer implementation,
- no action-dispatch execution wiring,
- no provenance panel runtime queries,
- no release-ladder extension claim (`XG-802` remains separate).

---

## 7) Validation entrypoints for this slice

- `python -m json.tool state/continuity/latest/xu_501_navigation_state_model_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xu_501_operator_walkthrough_evidence_2026-03-28.json`
- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`

---

## 8) Closeout criteria for `XU-501`

`XU-501` is complete only when:
1. this IA v2 doctrine is canonical,
2. navigation/state model + walkthrough evidence artifacts are published,
3. evidence-source mapping is explicit and fail-closed,
4. queue-layer slice `XU-501` is transitioned to `DONE` with evidence refs.
