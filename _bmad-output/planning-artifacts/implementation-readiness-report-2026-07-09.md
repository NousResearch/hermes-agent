---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
includedFiles:
  prd:
    - /Users/agent/Desktop/workspace/opensource/hermes-agent/_bmad-output/planning-artifacts/prd.md
  architecture:
    - /Users/agent/Desktop/workspace/opensource/hermes-agent/_bmad-output/planning-artifacts/architecture.md
  epics:
    - /Users/agent/Desktop/workspace/opensource/hermes-agent/_bmad-output/planning-artifacts/epics.md
  ux: []
---

# Implementation Readiness Assessment Report

**Date:** 2026-07-09
**Project:** hermes-agent

## Document Inventory

### PRD Files Found

**Whole Documents:**
- `prd.md` (12,857 bytes, modified 2026-07-09 09:40:15 +07)

**Sharded Documents:**
- None found

### Architecture Files Found

**Whole Documents:**
- `architecture.md` (12,545 bytes, modified 2026-07-09 09:40:15 +07)

**Sharded Documents:**
- None found

### Epics & Stories Files Found

**Whole Documents:**
- `epics.md` (60,595 bytes, modified 2026-07-09 09:40:15 +07)

**Sharded Documents:**
- None found

### UX Design Files Found

**Whole Documents:**
- None found

**Sharded Documents:**
- None found

### Discovery Issues

- No duplicate whole/sharded document formats found.
- Warning: UX design document not found. This may reduce assessment completeness unless UX requirements are captured in PRD, Architecture, or Epics.

## PRD Analysis

### Functional Requirements

FR-1: Create and view Project Bindings — create, view, update, disable, and validate a Project Binding (profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, display name). Rejects invalid cwd; shows active binding before any workflow; persists enough to reconstruct status after restart; shows whether provider binding is missing/valid/stale/conflicting.

FR-2: Mount project-local BMAD skills — add the bound project's BMAD skill directory to `skills.external_dirs`, reload skill index. Skills appear in discovery; records source directory; does not use global `~/.hermes/skills` as primary mount; detects missing or wrong-project mount.

FR-3: Enforce Bound Project Cwd for workflow actions — run BMAD and provider actions from the Bound Project Cwd unless stated otherwise. Artifacts land under the bound project's `_bmad-output`; blocks actions when cwd missing; audit records include cwd used; never infers cwd from skill visibility alone.

FR-4: Invoke BMAD planning workflows from Hermes — invoke brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, dev-story preparation from the Bound Project Cwd. Presents BMAD as behind-the-scenes; records artifact paths; surfaces failure output without losing binding context.

FR-5: Materialize BMAD sprint status into Project Work Items — read `sprint-status.yaml`, idempotently create/update Project Work Items. Re-running updates instead of duplicating; stores artifact references; reports unsupported/missing/malformed data before mutating; never treats the file as the runtime queue.

FR-6: Maintain Hermes-owned operational backlog — store operational backlog, selected story, phase metadata, workflow references, gate metadata, next action. User chooses next story without BMAD auto-picking; facade lanes (Ideas, Backlog, Active Runs, Review Test Cases, Verify Done) over canonical Kanban status; canonical lifecycle stays `triage`/`todo`/`ready`/`running`/`blocked`/`done`/`archived`.

FR-7: Register generic workflow provider bindings (consumer side) — create/update the provider-side binding using generic `provider`/`name` vocabulary, detect disagreement with Archon's stored binding, surface rotation/removal/stale/missing states.

FR-8: Control provider workflows through provider adapters (consumer side) — start, check status, approve, reject, resume, retry, cancel through the adapter. Uses parseable JSON; captures stdout/stderr/exit code/cwd/timeout/correlation id; fails closed on malformed JSON/unexpected exit code; no HTTP for the `archon` state-changing path.

FR-9: Receive typed workflow provider events (consumer side) — receive signed events at `/p/{profile}/webhooks/workflow-events/{provider}`, validate schema/binding/replay/idempotency/authorization before mutating. Rejects unknown binding/wrong profile/wrong codebase/stale timestamp/duplicate id/invalid signature/unsupported provider/schema failure; stores accepted event ids for duplicate-safety; maps events to the correct Project Work Item.

FR-10: Surface provider event delivery and outbox health (consumer side) — show whether delivery is healthy/delayed/failed/duplicated/waiting-for-reconciliation. Exposes status on Story Timeline; shows terminal failures as actionable diagnostics; never blocks workflow execution solely on notification failure.

FR-11: Create one phase task per BMAD story — each BMAD story has exactly one Phase Task sharing a Story Timeline; repeated materialization never duplicates it.

FR-12: Run the combined story workflow — start the configured combined workflow for a selected story; record the run reference on the Phase Task; runs story creation through review without a Hermes-side pause in between.

FR-13: Gate done verification — block the phase task with gate kind `done_verification` after workflow provider reports completion; human approval completes the task; rejection reruns the fix loop or routes recovery.

FR-14: Collect human decisions from Hermes — notify user, present gate evidence, capture approval/rejection, store the decision, send the matching command when required. Each decision records actor/timestamp/gate kind/decision/reason/evidence; approval and rejection are visibly distinct; never auto-continues past a HILT Gate without explicit persisted policy.

FR-15: Show a unified Story Timeline — show BMAD milestones, Project Work Item state, phase task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, next action in one place. Distinguishes BMAD artifact status from Hermes Kanban status; distinguishes GitHub PR merge state from Done Verification Gate state.

FR-16: Reconcile cross-system state — compare BMAD artifact state, provider workflow state, GitHub PR state, Project Work Item state, HILT Gate state to detect drift. Detects completed-but-unapplied provider runs, GitHub-merge-vs-unresolved-gate conflicts, unmaterialized BMAD changes; reports automatic repair vs. needs-human-action.

FR-17: Provide operational diagnostics — surface binding conflicts, cwd problems, missing artifacts, unsupported sprint status, provider command gaps, delivery failures, duplicate events, outbox backlog, stale PR references, unresolved gates. Diagnostics distinguish user/configuration/implementation-defect/external-delay; never silently mark work complete on conflicting evidence.

Total FRs: 17

### Non-Functional Requirements

NFR-1: Workflow events are delivery acceleration, not sole source of truth.

NFR-2: Reconcile after event loss/duplicate/gateway downtime/command failure/manual PR merge.

NFR-3: Materialization must be idempotent.

NFR-4: Gate decisions replay-safe and auditable.

NFR-5: Reject events failing signature/schema/replay/binding/provider/authorization checks.

NFR-6: Scope event secrets to correct profile.

NFR-7: Prevent workflow actions outside Bound Project Cwd.

NFR-8: Redact secrets in command logs, event logs, diagnostics, timeline views.

NFR-9: Persist workflow commands, events, reconciliation actions, gate decisions, state transitions.

NFR-10: Story Timeline sufficient to understand why a story changed state.

NFR-11: State changes carry enough provenance to distinguish BMAD/provider/GitHub/Hermes/workflow event/human decision sources.

NFR-12: Phrase next actions in user-facing language, not backend state names.

NFR-13: Distinguish Done Verification Gate approval from GitHub PR merge state.

NFR-14: Surface blocking issues with recovery options, not raw stack traces.

NFR-15: Preserve bounded ownership between Hermes/BMAD/providers/GitHub.

NFR-16: New provider integration surfaces stay generic.

NFR-17: Cross-project handoffs complete enough for isolated agents.

Total NFRs: 17

### Additional Requirements

- UX/frontend work is explicitly out of scope for this slice. Existing dashboard shell, Kanban plugin, task drawer, and comment/attachment surfaces are reused as-is; new data surfaces through existing generic components.
- The product must not replace BMAD, Archon, GitHub, or Hermes Kanban with a monolithic workflow database.
- The product must not require Archon's dashboard for normal workflow control.
- Hermes must not use Archon HTTP APIs for the Hermes-to-Archon control path.
- Hermes must not treat `sprint-status.yaml`, GitHub Issues, or Archon UI as runtime queue state.
- Hermes must not rely on global `~/.hermes/skills` as the primary BMAD mount mechanism.
- Hermes must not auto-approve HILT Gates without explicit persisted policy and evidence requirements.
- Story-level cross-project dependency records must include dependency, contract needed, blocking behavior, and integration validation fields.
- The Archon producer-side obligations for FR-7 through FR-10 live outside this local PRD and must be validated against the separate Archon handoff before integration completion.

### PRD Completeness Assessment

The PRD is implementation-oriented and clear about ownership, boundaries, and anti-goals. It provides a strong FR/NFR inventory for traceability against epics. The main completeness risk is that several NFRs are terse one-line requirements without quantified thresholds or concrete acceptance tests, and FR-7 through FR-10 depend on a separate Archon producer-side handoff that is not part of this local artifact set. UX is intentionally out of scope, so the missing UX document is not automatically blocking, but story and architecture coverage must preserve the stated reuse-only UI decision.

## Epic Coverage Validation

### Coverage Matrix

| FR Number | PRD Requirement | Epic Coverage | Status |
| --------- | --------------- | ------------- | ------ |
| FR-1 | Create and view Project Bindings | Story 2.1 | Covered |
| FR-2 | Mount project-local BMAD skills | Story 2.2 | Covered |
| FR-3 | Enforce Bound Project Cwd for workflow actions | Story 2.3 | Covered |
| FR-4 | Invoke BMAD planning workflows from Hermes | Story 2.4 | Covered |
| FR-5 | Materialize BMAD sprint status into Project Work Items | Story 2.5 | Covered |
| FR-6 | Maintain Hermes-owned operational backlog | Story 2.6 | Covered |
| FR-7 | Register generic workflow provider bindings | Story 3.2 | Covered |
| FR-8 | Control provider workflows through provider adapters | Stories 3.4a, 3.4b, 3.4c | Covered |
| FR-9 | Receive typed workflow provider events | Stories 3.6a, 3.6b, 3.6c | Covered |
| FR-10 | Surface provider event delivery and outbox health | Story 3.8 | Covered |
| FR-11 | Create one phase task per BMAD story | Stories 2.6, 3.6c | Covered |
| FR-12 | Run the combined story workflow | Stories 3.6c, 4.1 | Covered |
| FR-13 | Gate done verification | Stories 3.6c, 4.2 | Covered |
| FR-14 | Collect human decisions from Hermes | Stories 3.4b, 4.1, 4.2, 4.3 | Covered |
| FR-15 | Show a unified Story Timeline | Story 5.1 | Covered |
| FR-16 | Reconcile cross-system state | Stories 5.2a, 5.2b, 5.2c | Covered |
| FR-17 | Provide operational diagnostics | Story 5.3 | Covered |

### Missing Requirements

No PRD functional requirements are missing from the epics coverage map.

### Coverage Statistics

- Total PRD FRs: 17
- FRs covered in epics: 17
- Coverage percentage: 100%
- Extra FRs claimed in epics but absent from PRD: None

### Coverage Observations

- The epics handoff states a global blocked dependency: all stories reference shared contract fixtures from parent Stories 1.3a/1.3b/1.3c, and those fixtures do not exist in this local handoff as of 2026-07-02. This does not create FR coverage loss, but it does block implementation readiness.
- Story 3.6a additionally claims NFR-6. NFR coverage is not part of this step's FR coverage calculation.

## UX Alignment Assessment

### UX Document Status

No standalone UX design document was found under `_bmad-output/planning-artifacts`.

### Alignment Issues

No direct PRD-to-architecture UX misalignment was found in the available artifacts. The PRD explicitly states that UI/frontend work is out of scope for this feature and that existing Hermes dashboard shell, Kanban plugin, task drawer, and comment/attachment surfaces are reused. The architecture matches this scope by omitting a new frontend source-tree seed and by defining reused display surfaces such as dashboard status, Hermes Kanban project work, Story Timeline projection, diagnostics, and a Hermes dashboard gate prompt.

### Warnings

- UX is still implied because this is a human-facing command center with visible Project Binding state, gate evidence, Story Timeline entries, diagnostics, status display, and user decisions.
- The missing UX document is not a blocking gap under the current product-scope decision because the PRD explicitly says no new UI/component story is needed.
- If the scope changes to add dedicated UI components, the local handoff will need a UX artifact or imported UX delta contract before implementation proceeds.

## Epic Quality Review

### Epic Structure Validation

| Epic | User Value Focus | Independence Assessment | Result |
| ---- | ---------------- | ----------------------- | ------ |
| Epic 2: Project-Bound Planning And Work Backlog | User can bind a project, mount BMAD skills, run planning from the right cwd, and materialize work. | Depends on parent/shared contract fixtures, but does not depend on later local epics. | Structurally valid, blocked by missing contracts |
| Epic 3: Workflow Provider Control And Event Delivery | User can connect Hermes to providers, control runs, and see event delivery health. | Correctly builds on Epic 2 outputs and Archon producer-side contracts. | Structurally valid, blocked by external producer/contracts |
| Epic 4: Human-Gated Story Execution | User can run combined story workflow and approve/reject done verification. | Correctly builds on Epics 2 and 3. | Structurally valid |
| Epic 5: Story Timeline, Reconciliation, And Diagnostics | User can inspect status, understand drift, and recover safely. | Correctly builds on Epics 2-4. | Structurally valid |

### Critical Violations

1. Shared contract fixtures are missing and explicitly block every story.

   Evidence: the epics overview states that every story references shared contract fixtures from parent Stories 1.3a/1.3b/1.3c and that those fixtures do not exist yet as of 2026-07-02. It also states no story should move to implementation-ready until the fixtures exist locally or are regenerated into the handoff.

   Impact: implementation agents cannot reliably validate provider command envelopes, workflow event envelopes, provider binding schema, Project Work Item identity, Phase Task identity, or materialization idempotency. This blocks implementation readiness even though FR coverage is complete.

   Recommendation: generate or import the shared contract fixtures into this local handoff before starting Phase 4 implementation, then update the story dependency notes from blocked to satisfied.

2. Archon producer-side dependencies are required but not locally verified.

   Evidence: multiple stories depend on Archon producer stories, including provider binding lifecycle, command envelopes, start/status JSON, approve/reject JSON, recovery JSON, signed workflow events, delivery health CLI JSON, and diagnostic categories.

   Impact: Hermes consumer stories can be implemented only to local stubs unless the Archon contracts are present and compatible. Integration completion cannot be claimed from the hermes-agent artifact set alone.

   Recommendation: validate the separate Archon handoff and attach/import the producer-side schemas, examples, and compatibility tests referenced by the Hermes stories.

### Major Issues

1. Several stories remain too large unless split before sprint commitment.

   Evidence: Stories 2.1, 3.6c, 5.2b, and 5.3 contain explicit sprint-slicing guard language. Story 2.1 spans binding creation, viewing, updating, disabling, validation, conflict detection, migration, uniqueness tests, and display state. Story 5.3 spans all operational diagnostic categories, recovery options, redaction, resolution history, and timeline linking.

   Impact: these stories may fail independent-completion expectations if taken into implementation unchanged.

   Recommendation: split any guarded story that cannot be implemented, tested, linted, and validated in one implementation cycle. Do this before sprint commitment, not during implementation.

2. NFR traceability is incomplete in the epics document.

   Evidence: the PRD defines NFR-1 through NFR-17, but the epics explicitly tag only NFR-6 in Story 3.6a. Many NFRs are clearly represented implicitly in acceptance criteria, but they are not systematically mapped.

   Impact: reliability, auditability, security, usability, and maintainability requirements may be missed during implementation or QA planning.

   Recommendation: add an NFR coverage map or add explicit NFR tags to stories whose acceptance criteria implement each NFR.

3. Some acceptance criteria depend on terms whose exact contracts are deferred.

   Evidence: diagnostics, recovery options, command envelopes, event envelopes, idempotency keys, provider delivery states, and identity derivation are repeatedly named, while the architecture defers their exact schemas and examples.

   Impact: many ACs are directionally testable but not fully executable until the deferred contract artifacts exist.

   Recommendation: resolve deferred schemas/examples first, then tighten AC language where needed to reference exact fixture names, enum values, and compatibility tests.

### Minor Concerns

- Some story titles are implementation-heavy even though their user stories are valid. Examples: "Persist Workflow Event Idempotency Receipts And Duplicate Diagnostics" and "Map Accepted Workflow Events To Project Work." This is acceptable for a backend orchestration slice, but the titles could be reframed around user-visible safety outcomes.
- Some AC phrases remain qualitative, such as "user-actionable diagnostic," "appropriate recovery option," and "enough diagnostic context." These become acceptable only if the diagnostic vocabulary and fixture expectations are supplied before implementation.

### Dependency Analysis

- No within-local-epic forward dependencies were found.
- Epic 2 stories depend only on earlier Epic 2 stories plus missing parent fixtures.
- Epic 3 stories depend on Epic 2 and earlier Epic 3 stories; no dependency on Epic 4 or 5 was found.
- Epic 4 stories depend on Epic 2, Epic 3, and earlier Epic 4 stories; no dependency on Epic 5 was found.
- Epic 5 stories depend on Epics 2-4 and earlier Epic 5 stories; no forward references were found.
- External Archon dependencies are explicit and traceable, but readiness requires proof those contracts exist.

### Database And Entity Creation Timing

No all-tables-up-front violation was found. The stories generally create persistence where first needed: Project Binding in Story 2.1, Project Work Items in Story 2.5, Phase Tasks in Story 2.6, workflow event receipts in Story 3.6b, done-verification gates in Story 4.2, and shared gate decision hardening in Story 4.3.

### Starter Template And Brownfield Check

The architecture ratifies a brownfield stack and explicitly requires no new runtime infrastructure for v1. No starter template is specified, and no greenfield initial setup story is required.

### Best Practices Compliance Checklist

- Epic delivers user value: Pass
- Epic can function independently in sequence: Pass with external fixture blockers
- Stories appropriately sized: Conditional failure until guarded large stories are split or explicitly accepted
- No forward dependencies: Pass
- Database tables created when needed: Pass
- Clear acceptance criteria: Pass with contract-dependent caveats
- Traceability to FRs maintained: Pass

## Summary and Recommendations

### Overall Readiness Status

NOT READY

The planning set is directionally strong and internally coherent, but it is not ready for Phase 4 implementation. The deciding issue is not FR coverage: all 17 PRD functional requirements are covered. The blocker is that the epics and architecture explicitly require shared contract fixtures and producer-side Archon contracts that are not present or locally verified.

### Critical Issues Requiring Immediate Action

1. Shared contract fixtures are missing.

   Required before implementation: workflow command envelopes, workflow event envelopes, Workflow Provider Binding shape, Project Work Item identity, Phase Task identity, materialization idempotency examples, and duplicate/rejection fixtures.

2. Archon producer-side dependencies are not locally verified.

   Required before integration completion: provider binding lifecycle, command JSON for start/status/approve/reject/resume/retry/cancel, signed event fixtures, delivery health CLI JSON, and diagnostic category contracts.

3. NFR coverage is incomplete as explicit traceability.

   The PRD defines NFR-1 through NFR-17, but the epics explicitly tag only NFR-6. The rest may be implicit, but implementation readiness requires traceable coverage.

4. Large guarded stories need pre-sprint disposition.

   Stories 2.1, 3.6c, 5.2b, and 5.3 explicitly warn that they may need splitting. They should not enter sprint execution until they are either split or explicitly accepted as one-cycle stories with evidence.

### Recommended Next Steps

1. Generate or import the shared contract fixtures into this local handoff and update the blocked dependency notes.

2. Validate the Archon handoff and attach the producer-side schemas/examples/compatibility tests referenced by Hermes stories.

3. Add an NFR coverage map linking NFR-1 through NFR-17 to specific stories, acceptance criteria, or test fixtures.

4. Review guarded large stories and split any story that cannot be implemented, tested, linted, and validated in one implementation cycle.

5. Resolve deferred schema and vocabulary decisions, then tighten ACs that currently depend on qualitative phrases such as "user-actionable diagnostic" or "appropriate recovery option."

### Positive Findings

- No duplicate whole/sharded planning documents were found.
- PRD functional requirement coverage is complete: 17 of 17 FRs are mapped in epics.
- No within-local-epic forward dependencies were found.
- The PRD and architecture are aligned on the no-new-frontend UX scope.
- No all-tables-up-front database/entity creation violation was found.

### Final Note

This assessment identified 8 issues requiring attention across 4 categories: 2 critical blockers, 3 major issues, 2 minor concerns, and 1 UX/documentation warning. Address the critical blockers before proceeding to implementation. Proceeding as-is would push unresolved contract design and producer integration risk into implementation, which is exactly what the planning phase is supposed to prevent.

**Assessment Date:** 2026-07-09
**Assessor:** Codex using `bmad-check-implementation-readiness`
