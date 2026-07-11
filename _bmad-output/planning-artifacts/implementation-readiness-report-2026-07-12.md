---
project: hermes-agent
workflow: bmad-check-implementation-readiness
invocation: hermes-workflow
automationMode: non-interactive
date: 2026-07-12
status: NOT READY
issueCounts:
  critical: 0
  major: 1
  minor: 0
  warnings: 0
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
  ux: _bmad-output/planning-artifacts/ux-headless-interaction-contract.md
duplicateFindings: []
missingDocumentFindings: []
blockingEvidenceFindings:
  - External Archon producer runtime output is not present in the isolated local handoff.
---

# Implementation Readiness Assessment Report

**Date:** 2026-07-12
**Project:** hermes-agent

## Document Discovery

### PRD Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/prd.md` (16,253 bytes, modified 2026-07-12 06:13:53 +07)

**Sharded Documents:**
- None found.

**Selected for assessment:** `_bmad-output/planning-artifacts/prd.md`

### Architecture Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/architecture.md` (11,708 bytes, modified 2026-07-12 06:14:16 +07)

**Sharded Documents:**
- None found.

**Selected for assessment:** `_bmad-output/planning-artifacts/architecture.md`

### Epics And Stories Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/epics.md` (68,177 bytes, modified 2026-07-12 06:53:28 +07)

**Sharded Documents:**
- None found.

**Selected for assessment:** `_bmad-output/planning-artifacts/epics.md`

### UX Design Files Found

**Whole Documents:**
- `_bmad-output/planning-artifacts/ux-headless-interaction-contract.md` (3,533 bytes, modified 2026-07-12 06:15:33 +07)

**Sharded Documents:**
- None found.

**Selected for assessment:** `_bmad-output/planning-artifacts/ux-headless-interaction-contract.md`

### Discovery Issues

- No whole/sharded duplicates were found under `_bmad-output/planning-artifacts`.
- No required PRD, architecture, epics, or UX artifact is missing.

## PRD Analysis

### Functional Requirements

FR-1: Hermes can create, view, update, disable, and validate a Project Binding with profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, and display name. Hermes rejects invalid cwd values and returns the active binding before starting any BMAD or provider action. Hermes persists enough binding metadata to reconstruct status after restart. Hermes reports provider binding status as missing, valid, stale, disabled, rotated, or conflicting when that evidence is available.

FR-2: Hermes can add the bound project's BMAD skill directory to the selected profile's `skills.external_dirs` and reload that profile's skill index. Hermes records the source directory for mounted BMAD skills. Hermes does not use global skills as the primary BMAD mount for multi-project control. Hermes detects missing and wrong-project BMAD mounts.

FR-3: Hermes runs BMAD and provider actions for a Project Binding from the Bound Project Cwd unless a requirement explicitly states otherwise. BMAD artifacts created through Hermes land under the bound project's configured output location. Hermes blocks actions when the selected Project Binding lacks a valid cwd. Hermes audit records include the cwd used for each workflow action. Hermes does not infer cwd from skill visibility alone.

FR-4: Hermes can invoke selected BMAD planning workflows for brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation from the Bound Project Cwd. Hermes presents BMAD as a behind-the-scenes workflow engine, records produced artifact paths, and preserves Project Binding context on failure. Hermes can continue orchestration from generated BMAD artifacts in the bound project.

FR-5: Hermes can read `sprint-status.yaml` from the Bound Project Cwd and idempotently create or update Project Work Items for BMAD epics and stories. Re-running materialization updates existing Project Work Items instead of duplicating them. Hermes stores BMAD artifact references and observed planning status on each Project Work Item. Hermes does not treat `sprint-status.yaml` as the runtime queue after materialization.

FR-6: Hermes persists operational backlog, selected story, phase metadata, workflow references, human gate metadata, and next action as operational project-work state. Hermes exposes that state through structured command, agent, or API results. Hermes keeps canonical Kanban lifecycle values unchanged: `triage`, `todo`, `ready`, `running`, `blocked`, `done`, and `archived`.

FR-7: Hermes can register or inspect provider-side workflow bindings through generic `provider` and `name` vocabulary. Hermes detects disagreement between the Project Binding and provider binding metadata. Hermes surfaces missing, stale, disabled, rotated, and conflicting provider binding states as actionable diagnostics. Archon owns producer-side provider binding persistence and status production.

FR-8: Hermes can start, inspect, approve, reject, resume, retry, and cancel provider workflow runs through the adapter selected by the Project Binding. For provider `archon`, Hermes consumes parseable CLI JSON through the provider adapter. Hermes captures stdout, stderr, exit code, cwd when applicable, timeout, correlation id, and parsed result. Hermes fails closed on malformed JSON, incompatible schema version, timeout, unexpected exit code, or unexpected state. Hermes does not use provider HTTP APIs for the `archon` state-changing control path.

FR-9: Hermes receives signed workflow provider events on `/p/{profile}/webhooks/workflow-events/{provider}` and mutates project work only after schema, binding, replay, idempotency, provider, profile, and authorization validation pass. Hermes rejects unknown Project Binding, wrong profile, wrong codebase, stale timestamp, duplicate event id, invalid signature, unsupported provider, and schema failure before mutation. Hermes stores accepted event ids and idempotency keys so duplicate delivery is safe. Hermes maps completion, failure, approval-request, and artifact events only to the intended Project Work Item or Phase Task.

FR-10: Hermes can return structured provider event-delivery status identifying healthy, delayed, failed, duplicated, terminal failure, or reconciliation-pending state. Hermes exposes this evidence through Story Status History and diagnostics. Hermes does not block provider workflow execution solely because event notification failed. Archon owns producer-side outbox and delivery-health status for provider `archon`.

FR-11: Hermes materializes each selected BMAD story as a single Phase Task. The Phase Task links to one Project Work Item and shares Story Status History evidence. Repeated materialization must not duplicate Phase Tasks.

FR-12: Hermes can run the configured combined story workflow for a selected BMAD story. Hermes records the provider workflow run reference on the Phase Task. The workflow can run story creation through review without a Hermes-side pause between phases.

FR-13: Hermes blocks the Phase Task for a Done Verification Gate after provider completion evidence arrives. Human approval completes the Phase Task. Human rejection routes to rerun, resume, retry, or recovery without marking the story complete.

FR-14: Hermes can publish or return structured gate evidence, accept approval or rejection through an authorized command or agent interaction, persist the decision, and send the matching provider command when required. Each gate decision records actor, timestamp, gate kind, decision, reason when present, and evidence references. Hermes does not auto-continue past a HILT Gate unless an explicit persisted policy later permits it.

FR-15: Hermes can return one source-labeled Story Status History containing BMAD milestones, Project Work Item state, Phase Task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, provenance, and next action. The history distinguishes BMAD planning lifecycle from Hermes runtime lifecycle. The history distinguishes GitHub PR merge state from Done Verification state.

FR-16: Hermes compares BMAD artifact state, provider workflow state, GitHub PR state, Hermes Project Work Item state, and HILT Gate state to detect drift. Hermes may repair deterministic projection drift. Hermes must not auto-approve HILT Gates or mark stories complete when evidence conflicts.

FR-17: Hermes surfaces binding conflicts, cwd problems, missing BMAD artifacts, unsupported sprint status, provider command contract gaps, event delivery failures, duplicate workflow events, outbox backlog, stale PR references, and unresolved gates. Diagnostics distinguish user action, configuration action, Hermes automation, provider action, BMAD action, GitHub action, implementation-agent action, and external delay. Diagnostics include recovery guidance and redact secrets.

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
NFR-15: Dependency records and handoff validation preserve bounded ownership between Hermes, BMAD, workflow providers, GitHub, and parent shared contracts.
NFR-16: Provider integration surfaces remain generic.
NFR-17: Isolated local handoffs are complete enough for subproject implementation agents.

Total NFRs: 17

### Additional Requirements

- Workflow Commander v1 is headless and explicitly excludes a dedicated dashboard, graphical Kanban board, gate screen, timeline screen, desktop view, web application, or marketing surface.
- Hermes owns user and project orchestration, Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD invocation, materialization, Project Work Items, Phase Tasks, HILT Gates, provider adapter consumption, workflow event ingress, Story Status History, reconciliation, diagnostics, and headless validation guidance.
- Archon is the first workflow provider and owns producer-side workflow execution, run state, provider binding, provider command JSON, event outbox, delivery status, and signed event production.
- Required local contract schemas and example fixture families must exist under `contracts/workflow-commander/` and pass compatibility validation before contract-gated downstream stories can complete.
- Candidate validation commands are `uv sync --extra dev`, `uv run pytest`, and `uv run ruff check .`.

### PRD Completeness Assessment

The PRD is complete enough for traceability assessment. It defines product boundary, ownership boundary, 17 Hermes-owned FRs, 17 NFRs, non-goals, contract readiness constraints, implementation root, and candidate validation commands. It also makes clear that producer-side Archon evidence is outside the Hermes-owned artifact set and must be supplied before provider-dependent completion claims.

## Epic Coverage Validation

### Epic FR Coverage Extracted

- FR-1: Covered by Stories 2.1a, 2.1b, and 2.1c.
- FR-2: Covered by Story 2.2.
- FR-3: Covered by Story 2.3.
- FR-4: Covered by Story 2.4.
- FR-5: Covered by Story 2.5.
- FR-6: Covered by Story 2.6.
- FR-7: Covered by Story 3.2.
- FR-8: Covered by Stories 3.4a, 3.4b, and 3.4c.
- FR-9: Covered by Stories 3.6a, 3.6b, 3.6c, 3.6d, and 3.6e.
- FR-10: Covered by Story 3.8.
- FR-11: Covered by Stories 2.6 and 3.6c.
- FR-12: Covered by Stories 3.6c, 3.6e, and 4.1.
- FR-13: Covered by Stories 3.6d and 4.2.
- FR-14: Covered by Stories 3.4b, 4.1, 4.2, and 4.3.
- FR-15: Covered by Story 5.1.
- FR-16: Covered by Stories 5.2a, 5.2b, 5.2c, 5.2d, and 5.2e.
- FR-17: Covered by Stories 5.3a, 5.3b, and 5.3c.

Total FRs in epics: 17

### Coverage Matrix

| FR Number | PRD Requirement | Epic Coverage | Status |
| --- | --- | --- | --- |
| FR-1 | Project Bindings | Stories 2.1a-2.1c | Covered |
| FR-2 | Project-local BMAD skills | Story 2.2 | Covered |
| FR-3 | Bound Project Cwd enforcement | Story 2.3 | Covered |
| FR-4 | BMAD planning workflow invocation | Story 2.4 | Covered |
| FR-5 | Sprint status materialization | Story 2.5 | Covered |
| FR-6 | Hermes operational backlog | Story 2.6 | Covered |
| FR-7 | Generic workflow provider bindings | Story 3.2 | Covered |
| FR-8 | Provider workflow control | Stories 3.4a-3.4c | Covered |
| FR-9 | Signed workflow provider events | Stories 3.6a-3.6e | Covered |
| FR-10 | Event delivery and outbox health | Story 3.8 | Covered |
| FR-11 | One Phase Task per BMAD story | Stories 2.6 and 3.6c | Covered |
| FR-12 | Combined story workflow | Stories 3.6c, 3.6e, and 4.1 | Covered |
| FR-13 | Done Verification Gate | Stories 3.6d and 4.2 | Covered |
| FR-14 | Human gate decisions | Stories 3.4b, 4.1, 4.2, and 4.3 | Covered |
| FR-15 | Unified Story Status History | Story 5.1 | Covered |
| FR-16 | Cross-system reconciliation | Stories 5.2a-5.2e | Covered |
| FR-17 | Operational diagnostics | Stories 5.3a-5.3c | Covered |

### Missing Requirements

No PRD functional requirements are missing from the epics document. No extra FR identifiers outside PRD FR-1 through FR-17 were found in story coverage.

### Coverage Statistics

- Total PRD FRs: 17
- FRs covered in epics: 17
- Coverage percentage: 100%

## UX Alignment Assessment

### UX Document Status

Found: `_bmad-output/planning-artifacts/ux-headless-interaction-contract.md`.

### Alignment Issues

No UX alignment issues were found. The UX contract, PRD, architecture, and epics all define Workflow Commander v1 as a headless operator experience delivered through Hermes commands, agent interactions, structured command/API results, durable records, Story Status History, diagnostics, and optional existing notification transports.

The UX contract response shapes are supported by architecture and story coverage:

- Project Binding inspection maps to FR-1, AD-2, and Stories 2.1a-2.1c.
- BMAD workflow invocation maps to FR-3/FR-4, AD-4, and Stories 2.3-2.4.
- Provider workflow control maps to FR-7/FR-8, AD-3, and Stories 3.2 and 3.4a-3.4c.
- Done Verification Gate maps to FR-13/FR-14 and Stories 4.2-4.3.
- Story Status History maps to FR-15, AD-5, and Story 5.1.
- Operational diagnostics map to FR-17 and Stories 5.3a-5.3c.

### Warnings

None. A dedicated graphical UX is explicitly out of scope for v1, and the UX contract supplies validation expectations for commands, structured results, gates, Story Status History, diagnostics, and redaction.

## Epic Quality Review

### Epic Structure Validation

The epics are user-outcome oriented rather than technical milestones:

- Epic 2 lets Kevin bind a project, mount BMAD skills, run planning from the correct cwd, and materialize BMAD stories into Hermes project work.
- Epic 3 lets Kevin connect Hermes to workflow providers, control workflow runs, and inspect event/outbox health without opening the provider dashboard.
- Epic 4 lets Kevin run the combined story workflow, review Done Verification evidence, approve or reject, and route recovery.
- Epic 5 lets Kevin query Story Status History, understand drift, and resolve operational diagnostics.

The local handoff intentionally starts at Epic 2 because parent/shared-contract work owns earlier contract stories. This numbering gap is documented in the epics frontmatter and the cross-project isolated handoff contract, so it is not a quality defect.

### Story Structure Validation

- Story headings found in `epics.md`: 30.
- Story entries in `_bmad-output/implementation-artifacts/sprint-status.yaml`: 30.
- Stories with `Requirements Covered`: 30.
- Stories with `Implementation Scope`: 30.
- Stories with `Acceptance Criteria`: 30.
- Acceptance criteria bullets using Given/When/Then form: all checked bullets comply.
- Error and failure paths are represented for invalid cwd, binding conflicts, mount failure, malformed provider JSON, schema mismatch, duplicate events, unresolved mappings, delayed delivery, rejected gates, redaction, diagnostics, and reconciliation conflicts.
- Database/entity creation is scoped to the story that first needs the data. The epics do not ask for all tables or models to be created upfront.
- Brownfield constraints are represented: architecture keeps the existing Hermes runtime, uses ports/adapters, preserves Kanban lifecycle values, and avoids new runtime infrastructure or a dedicated graphical frontend.

### Dependency Analysis

No forward dependency between Hermes-owned stories was found. Hermes-owned story dependencies point to earlier same-project stories, parent contract stories, or Archon producer stories. External parent/Archon dependencies are explicit and use the required `Depends on`, `Contract needed`, `Blocking behavior`, and `Integration validation` records.

The implementation tracker is current with the epics:

- `_bmad-output/planning-artifacts/epics.md` defines 30 story headings.
- `_bmad-output/implementation-artifacts/sprint-status.yaml` tracks 30 matching story keys, generated 2026-07-12T06:27:13+0700.
- Provider-dependent story completion constraints are preserved in tracker notes and epics dependency records.

### Contract Package Validation

The local contract package validation passed:

```text
Workflow Commander 1.3a/1.3b/1.3c contract validation passed
Validated 7 schemas
Validated 16 command examples
Validated 14 binding examples
Validated 7 delivery examples
Validated 6 generic event examples
Validated 7 provider event examples
Validated 9 callback rejection examples
Validated 6 materialization examples
Validated isolated local package without parent workspace traversal
```

### Critical Violations

None.

### Major Issues

1. **External Archon producer runtime output is not present in this isolated local handoff.**
   - Evidence: `_bmad-output/planning-artifacts/cross-project-isolated-handoff-contract.md` states external Archon producer completion status is not proven by this local handoff and provider-dependent Hermes stories must keep Archon dependency records until compatible producer output is available and validated.
   - Required evidence: provider binding lifecycle output for create/update/status/rotate/disable/remove, workflow command output for start/status/approve/reject/resume/retry/cancel and failure paths, workflow event output for started/completed/failed/approval/artifact/delivery paths, and delivery/outbox status output for healthy/delayed/retrying/failed/duplicated/terminal/reconciliation-pending paths.
   - Current mitigation: local schemas and 65 local examples validate, epics preserve explicit dependency records, and `sprint-status.yaml` warns that provider-dependent stories must not be marked done from local fixtures alone.
   - Impact: Hermes-side implementation work can proceed where each story allows fixture-driven work, but full implementation readiness is blocked and provider-dependent stories cannot be completed from the local handoff alone.
   - Recommendation: capture compatible Archon producer runtime output and validate it against the local contract package before marking provider-dependent Hermes stories complete.

### Minor Concerns

None.

### Best Practices Checklist

| Check | Result |
| --- | --- |
| Epics deliver user value | Pass |
| Epics avoid pure technical milestones | Pass |
| Hermes-owned stories avoid forward dependencies | Pass |
| Stories are sized around one implementation responsibility | Pass |
| Acceptance criteria are present and testable | Pass |
| Traceability to FRs is maintained | Pass |
| Local contract fixtures validate | Pass |
| Implementation tracking matches current stories | Pass |
| External producer runtime evidence is present | Fail |

## Summary And Recommendations

### Overall Readiness Status

NOT READY.

The local planning package is strong but not fully implementation-ready:

- PRD, architecture, epics, and UX artifacts are present as whole documents.
- PRD FR coverage is 100%.
- UX and architecture agree on the headless v1 boundary.
- `sprint-status.yaml` matches the current 30-story epics structure.
- The local Workflow Commander contract package validates successfully with 7 schemas and 65 examples.

The readiness blocker is external evidence: this isolated local handoff does not include validated runtime output from the external Archon producer. Per the persistent handoff contract, facts are insufficient to claim full readiness or complete provider-dependent Hermes stories from local fixtures alone.

### Issue Counts

- Critical: 0
- Major: 1
- Minor: 0
- Warnings: 0

### Critical Issues Requiring Immediate Action

None.

### Major Issues Requiring Resolution

1. Supply and validate compatible external Archon producer runtime output.
   - Validate provider binding lifecycle output.
   - Validate workflow command output.
   - Validate workflow event output.
   - Validate delivery and outbox status output.
   - Keep provider-dependent Hermes stories from `done` until this evidence is available and passes the local contract validator or story-specific compatibility tests.

### Recommended Next Steps

1. Proceed only with stories whose dependencies are locally satisfiable, starting from Epic 2 in sprint-status order.
2. Preserve Archon dependency records on provider-dependent stories and prevent completion claims from local fixtures alone.
3. Capture the required external Archon producer outputs and validate them against `_bmad-output/planning-artifacts/contracts/workflow-commander/`.
4. Re-run this readiness workflow after external producer evidence is added.

### Final Note

This assessment identified 1 issue in 1 category. The issue is not a local document-coverage failure; it is a missing external evidence blocker. Address the Archon producer runtime evidence gap before claiming full implementation readiness.

Assessor: Codex using `bmad-check-implementation-readiness` in non-interactive automation mode.
