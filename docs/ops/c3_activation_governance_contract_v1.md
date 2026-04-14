# C3 Activation Governance Contract v1 (`XG-801`)

Date: 2026-03-28  
Status: active (canonical foundation for XG lane activation governance)  
Owner: Architect  
Scope: Cross-domain activation governance for downstream domain lanes (`XD/XK/XP/XT/XH`) in `/home/yeqiuqiu/clawd-architect`

---

## 1) Purpose

Define the mandatory governance foundation required before any downstream domain lane runtime activation.

`XG-801` establishes:
- shared entry criteria for C3 domain-lane activation,
- risk classes and activation posture constraints,
- ownership and escalation responsibilities,
- dependency-gate semantics used to unblock downstream XR2+ slices.

This slice is governance-foundation only.
It does **not** implement release evidence ladder extension (`XG-802`) or incident contract runtime/schema depth (`XG-803`).

---

## 2) Canonical inputs and bounded outputs

### Canonical inputs
- `state/continuity/latest/true_expanded_roadmap_queue_layer.json` (`XG-801..XG-803`, downstream dependency graph)
- `docs/ops/release_evidence_ladder_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`
- `docs/ops/lane_boundary_contract_v1.md`
- `docs/ops/true_expanded_out_of_core_scope_filter_rules_v1.md`
- `docs/ops/core_roadmap_dependency_unblock_policy_pack_v1.md`

### Canonical outputs for `XG-801`
- governance contract artifact (this file),
- cross-domain activation risk matrix artifact,
- cross-domain owner registry artifact,
- queue-layer state closeout evidence.

Deferred to follow-on slices:
- `XG-802`: release evidence ladder extension for domain lanes,
- `XG-803`: fail-close domain incident contract and remediation loops.

---

## 3) Governance boundary and non-goals

### In-boundary
- Activation governance for downstream C3 lane bring-up.
- Deterministic entry criteria and risk-class mapping.
- Ownership and escalation model for activation decisions.
- Mandatory fail-closed posture when criteria are unmet.

### Out-of-boundary for `XG-801`
- No runtime producer implementation for domain lanes.
- No new release bundle schema/template updates.
- No incident artifact schema or simulator wiring.
- No direct activation of blocked slices without dependency satisfaction.

---

## 4) Activation entry criteria (mandatory)

A downstream lane activation candidate is eligible only when all criteria pass:

1. **Canonical scope fit**
   - Candidate artifacts classify `IN_SCOPE_CORE_OR_EXPANDED` under
     `docs/ops/true_expanded_out_of_core_scope_filter_rules_v1.md`.
   - Unknown classification defaults to quarantine (fail-closed).

2. **Lane boundary declaration**
   - Candidate declares lane identity, domain objective, allowed write paths, and crossover assumptions compatible with
     `docs/ops/lane_boundary_contract_v1.md`.

3. **Bridge integrity posture**
   - Any cross-lane data flow is reference-first and compatible with
     `docs/ops/controlled_cross_lane_bridge_contract_v1.md`.

4. **Risk class assignment**
   - Candidate is assigned exactly one risk class from the `XG-801` risk matrix.
   - Assigned risk class controls minimum activation mode ceiling.

5. **Ownership assignment**
   - Candidate has mapped owner tuple: governance owner, lane owner, release owner, incident owner.
   - Missing ownership tuple is an automatic block.

6. **Dependency parity**
   - Upstream dependencies in `true_expanded_roadmap_queue_layer` are `DONE`.
   - Dependency mismatch or stale queue status is an automatic block.

---

## 5) Risk classes and activation posture

`XG-801` defines four risk classes:

- `RG0_LOW`: documentary/governance-only slices with no runtime mutation.
- `RG1_MODERATE`: bounded non-safety runtime slices with deterministic rollback path.
- `RG2_HIGH`: domain runtime slices with user-impacting behavior; stronger release and incident obligations.
- `RG3_CRITICAL`: safety-sensitive and/or irreversible-impact slices; highest review depth and hard fail-close thresholds.

Activation posture constraints:
- `RG0_LOW`: may proceed under documented governance checks only.
- `RG1_MODERATE`: requires release ladder progression at least through integration-replay (full extension in `XG-802`).
- `RG2_HIGH`: requires canary-grade release evidence and explicit rollback path.
- `RG3_CRITICAL`: requires progressive/broad activation prohibition until `XG-802` + `XG-803` close and domain-specific safety contracts are active.

Until `XG-802` lands, all `RG2_HIGH` and `RG3_CRITICAL` runtime activations remain blocked.

---

## 6) Ownership and escalation contract

Each domain-lane activation candidate must declare:

- `governance_owner`: authority for policy fit and entry-criteria decision.
- `lane_owner`: authority for lane-local contract implementation quality.
- `release_owner`: authority for release posture/go-no-go coupling.
- `incident_owner`: authority for fail-close remediation and escalation loop.

Escalation levels:
1. `L1_BOUNDARY_FAILURE` — boundary/scope mismatch, unresolved lane identity, or write-path ambiguity.
2. `L2_RELEASE_GAP` — required release-stage coverage or rollback evidence missing.
3. `L3_INCIDENT_READINESS_GAP` — missing deterministic incident handling obligations.
4. `L4_SAFETY_BLOCK` — high-risk activation attempt without mandatory safety controls.

Fail-close rule: any unresolved L2+ escalation blocks activation.

---

## 7) Dependency gate obligations (queue-layer unblocking)

`XG-801` is the mandatory prerequisite for downstream slice families:
- `XP-301`, `XT-601`, `XH-701`, `XU-501`, `XB-401`, and subsequent slices that depend on cross-domain activation governance.

Unblock condition from `XG-801`:
1. Contract artifact exists and is canonical.
2. Risk matrix + owner registry artifacts exist and are machine-readable.
3. Source-of-truth map references the contract/artifacts under XG lane.
4. Queue-layer state for `XG-801` is `DONE` with evidence refs.

---

## 8) Validation entrypoints for this slice

- `python ops/openclaw/continuity/check_source_of_truth_map_regressions.py --repo-root . --map-path reports/openclaw_system_source_of_truth_map_2026-03-20.md --json`
- `python -m json.tool state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json`
- `python -m json.tool state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json`

---

## 9) Closeout criteria for `XG-801`

`XG-801` is complete only when:
1. this governance contract is canonical,
2. risk matrix and owner registry artifacts are published,
3. source-of-truth map includes XG-801 references,
4. queue-layer slice `XG-801` is transitioned to `DONE` with evidence refs,
5. no claims are made about `XG-802`/`XG-803` completion.
