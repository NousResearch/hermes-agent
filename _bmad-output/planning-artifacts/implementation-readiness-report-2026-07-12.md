---
project: hermes-agent
workflow: bmad-check-implementation-readiness
invocation: hermes-workflow
date: 2026-07-12
status: NOT READY
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
selectedFiles:
  prd: _bmad-output/planning-artifacts/prd.md
  architecture: _bmad-output/planning-artifacts/architecture.md
  epics: _bmad-output/planning-artifacts/epics.md
  ux: null
duplicateFindings: []
missingFindings:
  - UX design document not found under _bmad-output/planning-artifacts using *ux*.md or */ux*/index.md patterns.
persistentFactIssues:
  - Configured persistent fact file missing: _bmad-output/planning-artifacts/cross-project-isolated-handoff-contract.md
---

# Implementation Readiness Assessment Report

**Date:** 2026-07-12
**Project:** hermes-agent

## Document Discovery

### PRD Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/prd.md` (15,908 bytes, modified 2026-07-12 06:00:00 +0700)

**Sharded Documents:**
- None found

**Selected for assessment:** `_bmad-output/planning-artifacts/prd.md`

### Architecture Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/architecture.md` (10,637 bytes, modified 2026-07-12 06:00:00 +0700)

**Sharded Documents:**
- None found

**Selected for assessment:** `_bmad-output/planning-artifacts/architecture.md`

### Epics & Stories Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/epics.md` (63,985 bytes, modified 2026-07-12 06:00:00 +0700)

**Sharded Documents:**
- None found

**Selected for assessment:** `_bmad-output/planning-artifacts/epics.md`

### UX Design Files Found

**Whole Documents:**
- None found

**Sharded Documents:**
- None found

**Selected for assessment:** None

### Discovery Issues

- **Missing artifact:** UX design document was not found. This reduces assessment completeness and may block readiness if UX obligations are referenced by PRD, architecture, or epics.
- **Missing persistent fact:** `_bmad-output/planning-artifacts/cross-project-isolated-handoff-contract.md` was configured as foundational context but does not exist in this worktree.

## PRD Analysis

### Functional Requirements

FR-1: Hermes can create, view, update, disable, and validate a Project Binding with profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, and display name. It rejects invalid cwd values, returns the active binding before BMAD/provider actions, persists enough metadata to reconstruct status after restart, and reports provider binding status as missing, valid, stale, disabled, rotated, or conflicting when available.

FR-2: Hermes can add the bound project's BMAD skill directory to the selected profile's `skills.external_dirs`, reload that profile's skill index, record the source directory, avoid using global skills as the primary BMAD mount for multi-project control, and detect missing or wrong-project BMAD mounts.

FR-3: Hermes runs BMAD and provider actions for a Project Binding from the Bound Project Cwd unless explicitly stated otherwise. BMAD artifacts created through Hermes land under the bound project's configured output location. Hermes blocks actions without a valid cwd, records the cwd used for each workflow action, and does not infer cwd from skill visibility alone.

FR-4: Hermes can invoke selected BMAD planning workflows for brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation from the Bound Project Cwd. It presents BMAD as a behind-the-scenes workflow engine, records produced artifact paths, preserves Project Binding context on failure, and continues orchestration from generated BMAD artifacts.

FR-5: Hermes can read `sprint-status.yaml` from the Bound Project Cwd and idempotently create or update Project Work Items for BMAD epics and stories. Re-runs update existing items instead of duplicating them. Hermes stores BMAD artifact references and observed planning status, and does not treat `sprint-status.yaml` as the runtime queue after materialization.

FR-6: Hermes persists operational backlog, selected story, phase metadata, workflow references, human gate metadata, and next action as operational project-work state. It exposes that state through structured command, agent, or API results while keeping canonical Kanban lifecycle values unchanged: `triage`, `todo`, `ready`, `running`, `blocked`, `done`, and `archived`.

FR-7: Hermes can register or inspect provider-side workflow bindings through generic `provider` and `name` vocabulary, detect disagreement between Project Binding and provider binding metadata, and surface missing, stale, disabled, rotated, and conflicting provider binding states as actionable diagnostics. Archon owns producer-side provider binding persistence and status production.

FR-8: Hermes can start, inspect, approve, reject, resume, retry, and cancel provider workflow runs through the adapter selected by the Project Binding. For `archon`, Hermes consumes parseable CLI JSON, captures stdout, stderr, exit code, cwd when applicable, timeout, correlation id, and parsed result, and fails closed on malformed JSON, incompatible schema version, timeout, unexpected exit code, or unexpected state. Hermes does not use provider HTTP APIs for the `archon` state-changing control path.

FR-9: Hermes receives signed workflow provider events on `/p/{profile}/webhooks/workflow-events/{provider}` and mutates project work only after schema, binding, replay, idempotency, provider, profile, and authorization validation pass. Hermes rejects unknown Project Binding, wrong profile, wrong codebase, stale timestamp, duplicate event id, invalid signature, unsupported provider, and schema failure before mutation, stores accepted event ids and idempotency keys, and maps completion, failure, approval-request, and artifact events only to the intended Project Work Item or Phase Task.

FR-10: Hermes can return structured provider event-delivery status identifying healthy, delayed, failed, duplicated, terminal failure, or reconciliation-pending state. It exposes this evidence through Story Status History and diagnostics and does not block provider workflow execution solely because event notification failed. Archon owns producer-side outbox and delivery-health status for provider `archon`.

FR-11: Hermes materializes each selected BMAD story as a single Phase Task linked to one Project Work Item and shared Story Status History evidence. Repeated materialization must not duplicate Phase Tasks.

FR-12: Hermes can run the configured combined story workflow for a selected BMAD story, record the provider workflow run reference on the Phase Task, and allow the workflow to run story creation through review without a Hermes-side pause between phases.

FR-13: Hermes blocks the Phase Task for a Done Verification Gate after provider completion evidence arrives. Human approval completes the Phase Task. Human rejection routes to rerun, resume, retry, or recovery without marking the story complete.

FR-14: Hermes can publish or return structured gate evidence, accept approval or rejection through an authorized command or agent interaction, persist the decision, and send the matching provider command when required. Each gate decision records actor, timestamp, gate kind, decision, reason when present, and evidence references. Hermes does not auto-continue past a HILT Gate unless an explicit persisted policy later permits it.

FR-15: Hermes can return one source-labeled Story Status History containing BMAD milestones, Project Work Item state, Phase Task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, provenance, and next action. The history distinguishes BMAD planning lifecycle from Hermes runtime lifecycle and GitHub PR merge state from Done Verification state.

FR-16: Hermes compares BMAD artifact state, provider workflow state, GitHub PR state, Hermes Project Work Item state, and HILT Gate state to detect drift. It may repair deterministic projection drift, but must not auto-approve HILT Gates or mark stories complete when evidence conflicts.

FR-17: Hermes surfaces binding conflicts, cwd problems, missing BMAD artifacts, unsupported sprint status, provider command contract gaps, event delivery failures, duplicate workflow events, outbox backlog, stale PR references, and unresolved gates. Diagnostics distinguish responsible domains, include recovery guidance, and redact secrets.

Total FRs: 17

### Non-Functional Requirements

NFR-1: Workflow events are delivery acceleration, not the sole source of truth.
NFR-2: Reconciliation handles event loss, duplicate delivery, gateway downtime, provider command failure, and manual PR merge.
NFR-3: Materialization is idempotent.
NFR-4: Gate decisions are replay-safe and auditable.
NFR-5: Event ingress fails closed on signature, schema, replay, binding, provider, profile, idempotency, or authorization failure.
NFR-6: Workflow event secrets are scoped to the correct profile.
NFR-7: Workflow actions cannot run outside the selected Bound Project Cwd.
NFR-8: Secrets are redacted from command logs, event logs, diagnostics, and status-history results.
NFR-9: Workflow commands, workflow events, reconciliation actions, gate decisions, and user-visible state transitions are persisted.
NFR-10: Story Status History explains why a story changed state.
NFR-11: Project-work changes retain source provenance.
NFR-12: Next actions use human-facing workflow language.
NFR-13: Done Verification approval remains separate from GitHub PR merge state.
NFR-14: Blocking issues return recovery options instead of raw stack traces alone.
NFR-15: Provider integration surfaces remain generic.
NFR-16: Isolated local handoffs are complete enough for subproject implementation agents.

Total NFRs: 16

### Additional Requirements

- Product boundary: v1 is headless and must not ship a dedicated Workflow Commander dashboard, graphical Kanban board, gate screen, timeline screen, desktop view, web application, or marketing surface.
- Ownership boundary: Hermes owns Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD invocation, materialization, Project Work Items, Phase Tasks, HILT Gates, provider adapter consumption, workflow event ingress, Story Status History, reconciliation, diagnostics, and headless validation guidance.
- Provider boundary: Archon is the first workflow provider and owns producer-side workflow execution, run state, provider binding, provider command JSON, event outbox, delivery status, and signed event production.
- Non-goals prohibit replacing BMAD, Archon, GitHub, or Hermes Kanban with a monolithic workflow database; requiring provider dashboard usage for normal control; using Archon HTTP APIs for state-changing control; adding Hermes-specific provider command vocabulary; treating `sprint-status.yaml`, GitHub Issues, or provider UI state as Hermes runtime queue truth; relying on global skills as the primary BMAD mount; auto-approving HILT Gates without persisted policy and evidence; writing implementation artifacts from parent planning; and shipping a dedicated graphical frontend in v1.
- Contract readiness rule: schema files under `contracts/workflow-commander/schemas/` and required example fixture families under `contracts/workflow-commander/examples/` must exist and pass compatibility tests before contract-gated downstream stories can complete.
- Candidate validation commands: `uv sync --extra dev`, `uv run pytest`, and `uv run ruff check .`.

### PRD Completeness Assessment

The PRD is clear about Hermes-owned scope, ownership boundaries, headless product boundary, and traceable FR/NFR identifiers. It also contains an explicit contract readiness rule. The main completeness risk is that the separately configured cross-project isolated handoff contract is missing from this worktree, and no UX artifact was found to validate any workflow or interaction assumptions beyond the headless boundary.

## Epic Coverage Validation

### Epic FR Coverage Extracted

- FR-1: Covered in Epic 2 by Stories 2.1a, 2.1b, and 2.1c.
- FR-2: Covered in Epic 2 by Story 2.2.
- FR-3: Covered in Epic 2 by Story 2.3.
- FR-4: Covered in Epic 2 by Story 2.4.
- FR-5: Covered in Epic 2 by Story 2.5.
- FR-6: Covered in Epic 2 by Story 2.6.
- FR-7: Covered in Epic 3 by Story 3.2.
- FR-8: Covered in Epic 3 by Stories 3.4a, 3.4b, and 3.4c.
- FR-9: Covered in Epic 3 by Stories 3.6a, 3.6b, 3.6c, 3.6d, and 3.6e.
- FR-10: Covered in Epic 3 by Story 3.8.
- FR-11: Covered in Epics 2 and 3 by Stories 2.6 and 3.6c.
- FR-12: Covered in Epics 3 and 4 by Stories 3.6c, 3.6e, and 4.1.
- FR-13: Covered in Epics 3 and 4 by Stories 3.6d and 4.2.
- FR-14: Covered in Epics 3 and 4 by Stories 3.4b, 4.1, 4.2, and 4.3.
- FR-15: Covered in Epic 5 by Story 5.1.
- FR-16: Covered in Epic 5 by Stories 5.2a, 5.2b, 5.2c, 5.2d, and 5.2e.
- FR-17: Covered in Epic 5 by Stories 5.3a, 5.3b, and 5.3c.

Total FRs in epics: 17

### Coverage Matrix

| FR Number | PRD Requirement | Epic Coverage | Status |
| --- | --- | --- | --- |
| FR-1 | Create and view Project Bindings | Epic 2, Stories 2.1a-2.1c | Covered |
| FR-2 | Mount project-local BMAD skills | Epic 2, Story 2.2 | Covered |
| FR-3 | Enforce Bound Project Cwd | Epic 2, Story 2.3 | Covered |
| FR-4 | Invoke BMAD planning workflows | Epic 2, Story 2.4 | Covered |
| FR-5 | Materialize sprint status into Project Work Items | Epic 2, Story 2.5 | Covered |
| FR-6 | Maintain Hermes-owned operational backlog | Epic 2, Story 2.6 | Covered |
| FR-7 | Consume generic workflow provider bindings | Epic 3, Story 3.2 | Covered |
| FR-8 | Control provider workflows through adapters | Epic 3, Stories 3.4a-3.4c | Covered |
| FR-9 | Receive typed workflow provider events | Epic 3, Stories 3.6a-3.6e | Covered |
| FR-10 | Return event delivery and outbox health | Epic 3, Story 3.8 | Covered |
| FR-11 | Create one Phase Task per BMAD story | Epic 2 Story 2.6; Epic 3 Story 3.6c | Covered |
| FR-12 | Run the combined story workflow | Epic 3 Stories 3.6c and 3.6e; Epic 4 Story 4.1 | Covered |
| FR-13 | Gate Done Verification | Epic 3 Story 3.6d; Epic 4 Story 4.2 | Covered |
| FR-14 | Collect human decisions from Hermes | Epic 3 Story 3.4b; Epic 4 Stories 4.1-4.3 | Covered |
| FR-15 | Return unified Story Status History | Epic 5, Story 5.1 | Covered |
| FR-16 | Reconcile cross-system state | Epic 5, Stories 5.2a-5.2e | Covered |
| FR-17 | Provide operational diagnostics | Epic 5, Stories 5.3a-5.3c | Covered |

### Missing Requirements

No PRD FRs are missing from the epics document.

### Coverage Statistics

- Total PRD FRs: 17
- FRs covered in epics: 17
- Coverage percentage: 100%
- FRs in epics but not in PRD: None

## UX Alignment Assessment

### UX Document Status

Not found. No whole UX document or sharded UX folder matched `_bmad-output/planning-artifacts/*ux*.md` or `_bmad-output/planning-artifacts/*ux*/index.md`.

### UX/UI Implication Assessment

The PRD, architecture, and epics all define Workflow Commander v1 as a headless product surface. The PRD says the feature ships through commands, agent interactions, structured API or command results, durable records, and optional existing notification transports, and explicitly excludes a dedicated dashboard, graphical Kanban board, gate screen, timeline screen, desktop view, web application, marketing surface, or dedicated graphical frontend. The architecture aligns with this by keeping Hermes as a human-facing command center and by using durable pending-gate queries, authorized decision commands, canonical `blocked` status plus `gate_kind=done_verification`, and structured records.

### Alignment Issues

No graphical UX alignment issue was found because the product and architecture explicitly exclude a dedicated UI. Headless interaction requirements are represented in PRD FRs and story acceptance criteria for structured command/API results, pending-gate evidence, notification-safe gate mirrors, human-facing next-action language, and redacted diagnostic output.

### Warnings

- No UX artifact exists. This is not a blocker for v1's explicitly headless scope, but implementation should preserve the command/API/notification interaction expectations captured in PRD FR-14, FR-15, FR-17 and Epic 4/5 acceptance criteria.

## Epic Quality Review

### Epic Structure Validation

The Hermes epics are user-value oriented rather than purely technical milestones:

- Epic 2 lets Kevin bind a project, mount BMAD skills, run planning from the correct cwd, and materialize BMAD stories into Hermes project work.
- Epic 3 lets Kevin connect to workflow providers, control workflow runs, and inspect event/outbox health without opening the provider dashboard.
- Epic 4 lets Kevin run the combined story workflow, review done-verification evidence, approve or reject, and route recovery.
- Epic 5 lets Kevin query Story Status History and resolve drift across BMAD, Hermes, workflow providers, GitHub, workflow events, and gates.

The missing Epic 1 in the local file is explained by the story ownership note: numbering is preserved from the parent workspace so cross-project dependency records remain stable. This is acceptable for a local handoff, provided parent contract dependencies remain explicit.

### Story Dependency Review

Hermes-to-Hermes dependencies are ordered backward through the story sequence. No direct forward dependency on a later Hermes story was found.

External dependencies are explicit and numerous. Stories reference parent contract Stories 1.3a, 1.3b, 1.3c and Archon producer Stories 3.1, 3.3a, 3.3b, 3.3c, 3.3d, 3.5, and 3.7. The contract package under `contracts/workflow-commander/` exists locally and validates through `uv run python _bmad-output/planning-artifacts/contracts/workflow-commander/validate_contracts.py`.

Contract validation result:

- Runtime used: CPython 3.11.15 via `uv run python`.
- Result: passed.
- Validated: 7 schemas, 16 command examples, 14 binding examples, 7 delivery examples, 6 generic event examples, 7 provider event examples, 9 callback rejection examples, 6 materialization examples.
- The same validator fails under bare `python3` because this worktree's default `python3` is Python 3.9.6.

### Acceptance Criteria Review

Most stories use testable Given/When/Then acceptance criteria with explicit fail-closed, idempotency, redaction, and diagnostic behavior. Error-path coverage is strong in binding validation, BMAD mount validation, provider command parsing, workflow event ingress, duplicate handling, materialization, gate decisions, and reconciliation.

### Critical Violations

None found in epic/story structure. No epic is merely a technical milestone, and no direct future Hermes story dependency was found.

### Major Issues

1. **Architecture and contract package disagree about example fixture reality.** The architecture says no example fixture files were observed under local `examples/`, but the contract package currently contains and validates 65 JSON examples. This stale architecture statement can cause implementation agents to keep stories blocked incorrectly or ignore available fixtures.

2. **NFR traceability numbering is inconsistent.** The PRD has 16 unlabeled NFR bullets, while the epics document contains a local NFR coverage map numbered NFR-1 through NFR-17. In particular, the epics map includes an NFR-15 for dependency records/bounded ownership that is not directly numbered in the PRD extraction, then maps provider-generic surfaces to NFR-16 and isolated handoffs to NFR-17. This should be reconciled before implementation stories rely on NFR ids.

3. **Story 2.3 risks forward-coupling Epic 2 to provider adapter behavior.** Story 2.3 requires proving provider adapter calls receive the Bound Project Cwd, but provider control/adapters are primarily introduced in Epic 3. This is implementable only if Story 2.3 explicitly creates a minimal provider-action port/test double, or if provider-specific cwd acceptance criteria move to Epic 3.

4. **Story 5.3a is broad relative to its acceptance criteria.** It covers diagnostic taxonomy, severity, affected-resource references, redacted evidence storage, next-action owner, recovery-option reference, and persistence rules across many diagnostic families, but has only two broad acceptance criteria. Add an explicit diagnostic family matrix or split the story if one implementation cycle cannot cover configuration, decision, external-delay, implementation-defect, duplicate-event, outbox, stale-PR, and unresolved-gate diagnostics.

5. **Contract validator command is underspecified for this repo's Python floor.** The contract README says to run `python3 .../validate_contracts.py`, but bare `python3` is 3.9.6 in this worktree and fails before validation. The project requires Python `>=3.11,<3.14`, and `uv run python ...` passes. The validation instruction should use the repo's Python resolution path.

### Minor Concerns

1. **No UX artifact exists.** This is acceptable for the explicitly headless v1 scope, but command/API/notification interaction details must stay covered by story acceptance criteria.

2. **External producer completion status is not locally proven.** The local contract package validates, but the readiness artifacts do not include status evidence that the referenced Archon producer stories are implemented. Hermes stories can be planned against fixtures, but completion of provider-dependent stories remains externally gated.

### Best Practices Compliance Checklist

| Check | Result |
| --- | --- |
| Epics deliver user value | Pass |
| Epics can progress sequentially without future Hermes epic dependencies | Pass with Story 2.3 caveat |
| Stories appropriately sized | Mostly pass; Story 5.3a needs tightening |
| No forward Hermes dependencies | Pass |
| Data/storage creation appears tied to first use | Pass |
| Clear acceptance criteria | Mostly pass; Story 5.3a needs a diagnostic matrix |
| FR traceability maintained | Pass |
| NFR traceability maintained | Fails until numbering is reconciled |

## Summary and Recommendations

### Overall Readiness Status

**NOT READY**

Implementation should not proceed as a broad readiness handoff until the blocking artifact/fact gap and traceability inconsistencies are corrected. FR coverage is strong, and the local contract package validates through the repo's Python resolution path, but the readiness package is not internally consistent enough to be considered implementation-ready.

### Issue Counts

- Critical: 1
- Major: 5
- Minor: 2
- Duplicate whole/sharded artifact findings: 0
- Total issues requiring attention: 8

### Critical Issues Requiring Immediate Action

1. **Missing configured persistent fact:** `_bmad-output/planning-artifacts/cross-project-isolated-handoff-contract.md` was configured as foundational workflow context but is absent from the worktree. Under the automation contract, missing required facts make the result NOT READY.

### Major Issues

1. Architecture and contract package disagree about example fixture reality: architecture says no examples were observed, while the contract package contains and validates 65 examples.
2. NFR traceability numbering is inconsistent: PRD extraction yields 16 NFRs, while epics map NFR-1 through NFR-17.
3. Story 2.3 risks forward-coupling Epic 2 to provider adapter behavior unless it explicitly owns a minimal provider-action port/test double or moves provider-specific cwd criteria to Epic 3.
4. Story 5.3a is too broad relative to its two acceptance criteria and needs a diagnostic family matrix or a split.
5. Contract validator instructions are brittle: bare `python3` fails in this worktree, while `uv run python` passes with Python 3.11.15.

### Minor Issues

1. No UX artifact exists. This is acceptable for v1's headless scope, but command/API/notification interaction expectations must remain explicit in stories.
2. External Archon producer completion status is not locally proven. Hermes stories can plan against validated fixtures, but provider-dependent story completion remains externally gated.

### Positive Readiness Evidence

- Required whole PRD, architecture, and epics artifacts are present.
- No whole/sharded duplicates were found.
- PRD FR coverage is complete: 17 of 17 FRs are explicitly mapped to stories.
- Epics are user-value oriented and not merely technical milestones.
- Hermes-to-Hermes story dependencies are ordered backward; no direct future Hermes story dependency was found.
- Contract validation passed through `uv run python _bmad-output/planning-artifacts/contracts/workflow-commander/validate_contracts.py`.

### Recommended Next Steps

1. Restore or regenerate `_bmad-output/planning-artifacts/cross-project-isolated-handoff-contract.md`, or remove it from workflow persistent facts if it is no longer required.
2. Update `architecture.md` to reflect the current validated contract package and fixture inventory.
3. Reconcile PRD and epics NFR numbering, then update the NFR coverage map.
4. Tighten Story 2.3 around provider cwd validation ownership.
5. Split or strengthen Story 5.3a with a concrete diagnostic family acceptance matrix.
6. Update contract validation instructions to use `uv run python` or otherwise require Python 3.11+ explicitly.

### Final Note

This assessment identified 8 issues across artifact completeness, traceability, story quality, validation reproducibility, UX documentation, and external dependency evidence. Address the critical issue and major traceability/artifact inconsistencies before treating the Hermes Workflow Commander handoff as implementation-ready.

Assessor: Codex running `bmad-check-implementation-readiness`
Completed: 2026-07-12
