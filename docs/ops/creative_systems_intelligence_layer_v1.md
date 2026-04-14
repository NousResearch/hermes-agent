# Creative Systems Intelligence Layer v1

Date: 2026-03-30  
Status: active (Program D / D1 canonical contract)  
Owner: Architect  
Scope: OpenClaw system-wide CSI layer contract (contract-first baseline)

---

## 1) Purpose

Define the **Creative Systems Intelligence (CSI) Layer** as a bounded, buildable system layer rather than a loose roadmap idea.

CSI is responsible for six subfunctions:
1. gap detection
2. pattern formation
3. pattern triage
4. idea/discovery mining
5. concept reframing
6. improvement generation

This contract sets clear boundaries, required outputs, and fail-closed guardrails so CSI contributes high-leverage insight without becoming vibes-only roadmap mutation.

---

## 2) Authority and scope boundary

### 2.1 Canonical authority
This file is the canonical D1 contract authority for Program D / slice D1.

### 2.2 Scope-in
This contract governs:
- CSI layer role and placement in OpenClaw architecture,
- normative behavior for C1-C6 subfunctions,
- required CSI output classes (pre-schema baseline before D2),
- triage and promotion handoff expectations,
- guardrails for evidence, reviewability, and promotion gating.

### 2.3 Scope-out
This contract does **not** replace:
- Program B promotion/eval authority (`evaluation_gated_promotion_loop_v1` and related contracts),
- Program C model routing/qualification authority,
- continuity/ground-truth authority.

CSI proposes and prioritizes candidates; CSI does not self-promote canon.

---

## 3) Layer position and hard dependency posture

CSI is a strategic upper layer that sits above:
- Ground-Truth / Continuity Integration,
- Orchestrator Contract + Evaluation-Gated Improvement,
- Model Routing / Qualification + Pattern Harvest.

Hard posture:
1. CSI outputs MUST be artifact-bound and machine-tractable.
2. CSI outputs MUST remain candidate-state until validator/eval gate outcomes are explicit.
3. CSI MUST consume prior layers as inputs; it MUST NOT bypass their controls.

---

## 4) Non-negotiable invariants (v1)

1. **Evidence-linked when possible**: candidate claims cite concrete artifacts/signals.
2. **Artifact-aware**: every CSI conclusion references source artifacts or explicit uncertainty notes.
3. **Operator-reviewable**: outputs are inspectable and auditable by humans.
4. **Promotion-gated**: no silent canonical mutation from CSI outputs.
5. **Fail-closed on ambiguity**: uncertain/high-risk recommendations default to `defer`/`hold`, not silent promotion.
6. **Bounded creativity**: novelty is allowed; unsupported canonization is not.

---

## 5) CSI execution model (v1)

Canonical loop:
1. **Ingest** candidate evidence from deterministic input classes.
2. **Detect gaps** and recurring pain points.
3. **Form patterns** across incidents, projects, and archives.
4. **Triage patterns** into actionable priority states.
5. **Reframe concepts** into cleaner architectural/problem formulations.
6. **Generate improvements** as candidate packets.
7. **Handoff to validator/eval/promotion lanes** (no self-promotion).

This v1 contract is loop-definition and behavioral governance. D2-D6 add formal schemas, state machines, and runtime orchestration.

---

## 6) Subfunction contracts (bounded definitions)

## 6.1 C1 — Gap Detection

### Required behavior
- detect missing capabilities/layers/contracts,
- detect recurring unresolved pain points and blind spots,
- detect evidence gaps blocking promotion confidence.

### Required inputs
- roadmap inventories, source-of-truth maps, backlog status,
- incident/failure/regression history,
- repeated operator requests and unresolved asks.

### Required outputs
- `gap_findings[]` with: `gap_id`, missing primitive, impact, recurrence, urgency, evidence refs.

### Scope-out
- C1 does not prescribe final implementation details; it flags structurally missing or weak areas.

## 6.2 C2 — Pattern Formation

### Required behavior
- detect recurring structures across artifacts,
- form candidate abstractions/classes from repeated signals,
- separate one-off anomalies from plausible reusable patterns.

### Required outputs
- `candidate_patterns[]` with pattern statement, supporting evidence set, recurrence estimate, leverage hypothesis.

### Scope-out
- C2 does not promote patterns to canon without C3/C5/C6 handoff and downstream gates.

## 6.3 C3 — Pattern Triage

### Required behavior
- score/rank candidates by leverage, evidence quality, recurrence, urgency, risk,
- classify state using bounded decision states,
- explicitly separate `now` vs `later` vs `reject`.

### Required triage states (v1 baseline)
- `candidate`
- `reviewed`
- `validated`
- `promoted`
- `deferred`
- `rejected`

### Required outputs
- triage queue entries with rationale, confidence, and next-action owner.

### Scope-out
- C3 ranking is advisory for promotion; final canonization remains validator/eval-gated.

## 6.4 C4 — Idea/Discovery Mining

### Required behavior
Mine for high-leverage opportunities across:
- internal archives and prior reports,
- past sessions and recurring asks,
- donor/repo research,
- incidents/regressions/failures,
- cross-project recurring needs.

### Required mining posture
- use selective donor extraction (`keep | adapt | avoid | defer | reject`),
- prefer concrete reusable primitives over broad cargo-cult copying,
- preserve source traceability for each mined insight.

### Required outputs
- discovery findings mapped to candidate patterns/improvements with source provenance.

## 6.5 C5 — Concept Reframing

### Required behavior
- transform vague asks into sharper problem formulations,
- convert feature wishlists into architecture-layer options,
- produce bounded reframing alternatives with tradeoffs.

### Required outputs
- reframing packets: original framing, proposed reframing(s), expected leverage, risk notes, evidence links.

### Scope-out
- C5 does not mutate canonical roadmap directly; it proposes reframing candidates.

## 6.6 C6 — Improvement Generation

### Required behavior
- generate roadmap/doctrine/primitive improvement candidates from C1-C5 outputs,
- encode candidate scope, expected benefit, risk, and validation requirements,
- route high-confidence candidates to evaluation/promotion lanes.

### Required outputs
- candidate improvement packets with explicit gate requirements and rollback expectations.

### Scope-out
- C6 is generation + packaging, not ungated deployment.

---

## 7) Input classes and output artifact classes (pre-D2 baseline)

## 7.1 Allowed input classes
1. canonical contracts and schemas,
2. execution reports and incident history,
3. roadmap and queue artifacts,
4. curated donor/repo review packets,
5. continuity/event/replay artifacts,
6. repeated operator request patterns.

## 7.2 Output artifact classes (v1 placeholder contract)
1. `candidate_pattern`
2. `candidate_improvement`
3. `csi_triage_entry`
4. `csi_reframing_packet`
5. `cross_project_synthesis_note`

D2 and D6 will formalize these as machine schemas; v1 requires field discipline and explicit provenance now.

---

## 8) Guardrails and fail-closed policy

CSI MUST NOT become:
- random brainstorm spam,
- silent roadmap mutation,
- ungated speculative canon,
- evidence-free novelty assertions.

Fail-closed rules:
1. If evidence cannot be cited, mark confidence low and route to `deferred`/`review_required`.
2. If risk is high and validator path is absent, block promotion.
3. If candidate conflicts with continuity/truth constraints, reject or quarantine.
4. If donor-derived recommendation lacks adaptation rationale, classify `avoid` or `defer`.

---

## 9) Promotion handoff contract (CSI -> Program B)

For any candidate reaching `validated`:
1. attach evidence bundle,
2. attach risk + rollback notes,
3. attach evaluation requirements,
4. identify required reviewer roles (`validator_required` minimum for high-risk/canonical effects),
5. emit explicit promotion recommendation (`promote|hold|reject`) with rationale.

Final promotion authority remains in evaluation-gated improvement controls (Program B).

---

## 10) Source-input map (archive-mined CSI + roadmap fold-ins)

| Source ID | Artifact | Classification | Folded into this contract |
|---|---|---|---|
| src_exec_queue_d1 | `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md` | canonical_queue_directive | purpose, C1-C6 scope, done posture |
| src_csi_archive_mining | `reports/openclaw_archive_mining_creative_systems_intelligence_2026-03-30.md` | archive_mined_report | gap-audit protocol, donor mining posture, triage state orientation, validator-gated improvement |
| src_csi_formalization | `reports/openclaw_creative_systems_intelligence_layer_formalization_2026-03-30.md` | roadmap_formalization | subfunction naming, guardrails, layer role |
| src_missing_layers_foldin | `reports/openclaw_missing_layers_system_roadmap_foldin_2026-03-30.md` | roadmap_foldin | layer placement, dependency stack, evidence/review/promotion requirements |
| src_model_routing_doctrine | `docs/ops/model_routing_doctrine_v1.md` | active_support_contract | keep/adapt/avoid donor handling + validator/break-glass posture coupling |
| src_orchestrator_contract | `docs/ops/orchestrator_api_contract_v1.md` | active_support_contract | machine-readable artifact + fail-closed + replay-aware operating discipline |

---

## 11) D1 acceptance criteria

D1 is complete when:
1. one active CSI layer contract exists (this file),
2. all six CSI subfunctions are explicitly bounded,
3. guardrails and promotion handoff are explicit,
4. source archive/roadmap fold-ins are mapped,
5. CSI is defined as buildable contract authority (not just roadmap language).

---

## 12) Change control

Changes to CSI subfunction scope, invariants, or promotion handoff MUST:
1. update this contract,
2. update downstream schema/state-machine specs when D2-D6 land,
3. preserve fail-closed + promotion-gated behavior,
4. record rationale in a dated execution report.
