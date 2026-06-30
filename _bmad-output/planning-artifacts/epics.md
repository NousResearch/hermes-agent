---
title: hermes-agent Epics Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-06-27'
updated: '2026-06-29'
---

# hermes-agent Epics: Hermes Agent Workflow Commander

## Overview

This file contains the Hermes-owned subset of the parent Hermes Agent Workflow Commander epics.
It is local planning input for implementation inside `hermes-agent`.
It excludes workflow provider implementation work except where dependency notes are required for integration.
Archon is the first provider implementation under provider key `archon`.
Shared contract fixtures are planned under `_bmad-output/planning-artifacts/contracts/workflow-commander/` and must be regenerated into this local handoff before consumer code depends on them.
Workflow Commander UI work follows the approved UX requirements embedded in this handoff and `hermes-agent/web/README.md`.
No Hermes story may require traversal out of `hermes-agent` to read parent workspace planning files during implementation.

## Local Contract Validation Rule

Hermes consumer stories that parse workflow commands, workflow events, Workflow Provider Binding, delivery-health, materialization, phase-task, gate, reconciliation, or diagnostic payloads cannot move to implementation-ready until the relevant schema and example fixtures exist locally or as regenerated local equivalents.
Validation must prove the fixtures parse and that missing, malformed, duplicate, wrong-profile, wrong-binding, timeout, and conflicting evidence cases fail closed where applicable.

## Hermes NFR Coverage

| NFR | Hermes Story Coverage | Required Validation Evidence |
| --- | --- | --- |
| NFR-1 | Stories H2.4 and H4.2b | Workflow event health and reconciliation fixtures prove events accelerate delivery but reconciliation remains authoritative. |
| NFR-2 | Stories H4.2a, H4.2b, H4.2c, and H4.3 | Drift fixtures cover workflow event loss, duplicate delivery, gateway downtime, provider command failure, manual PR merge, and conflicting completion evidence. |
| NFR-3 | Stories H1.5, H1.6, and H4.2a | Materialization fixtures prove idempotent Project Work Item and Phase Task identity. |
| NFR-4 | Stories H2.2b, H3.1, H3.2, and H3.3 | Gate fixtures prove replay-safe decision records, separate command results, and auditable phase transitions. |
| NFR-5 | Stories H2.3a and H2.3b | Workflow event rejection and receipt fixtures cover signature, schema, replay, binding, provider, profile, idempotency, and authorization failures. |
| NFR-6 | Story H2.3a | Wrong-profile-secret fixtures prove profile-scoped workflow event secret enforcement. |
| NFR-7 | Stories H1.1, H1.3, and H2.2a | Cwd guard tests prove actions cannot run outside the selected Project Binding cwd. |
| NFR-8 | Stories H3.3, H4.1, and H4.3 | Gate, timeline, command, workflow event, and diagnostic fixtures prove secret redaction. |
| NFR-9 | Stories H1.3, H2.2a, H2.2b, H2.2c, H2.3b, H3.3, H4.2a, H4.2b, H4.2c, and H4.3 | Persistence and audit fixtures prove workflow commands, workflow events, reconciliation actions, gate decisions, and state transitions are recorded. |
| NFR-10 | Story H4.1 | Story Timeline fixtures explain every visible state change and next action. |
| NFR-11 | Stories H4.1, H4.2a, H4.2b, H4.2c, and H4.3 | Provenance fixtures distinguish BMAD, workflow provider, GitHub, Hermes, workflow event, reconciliation, implementation agent, and human decision sources. |
| NFR-12 | Stories H3.3, H4.1, and H4.3 | UI and diagnostic fixtures phrase next actions in user-facing workflow language. |
| NFR-13 | Stories H1.6, H3.1, H3.2, and H4.1 | Product Work and gate fixtures distinguish Review Test Cases from Verify Done. |
| NFR-14 | Stories H1.1, H1.2, H1.4, H2.1, H2.2a, H2.2b, H2.2c, H2.4, and H4.3 | Diagnostic fixtures show recovery options rather than raw stack traces alone. |
| NFR-15 | Stories H1.1 through H4.3 | Dependency records and local handoff boundaries preserve bounded ownership between Hermes, BMAD, workflow providers, and GitHub. |
| NFR-16 | Stories H2.1, H2.2a, H2.2b, and H2.2c | Hermes consumes generic provider binding and workflow surfaces without requiring Hermes-specific provider fields. |
| NFR-17 | Local handoff overview and all cross-project dependency records | Local `prd.md`, `architecture.md`, and `epics.md` are sufficient for isolated Hermes implementation without parent traversal. |

## Epic 1: Project-Bound Planning And Work Backlog

Hermes can bind a project, mount project-local BMAD skills, invoke planning from the correct cwd, and materialize BMAD stories into project work.

### Story H1.1: Create And Validate Project Bindings

As a workflow operator,
I want Hermes to create, view, update, disable, and validate explicit Project Bindings,
So that every BMAD and workflow provider action is tied to the correct profile, local cwd, GitHub context, BMAD mount, and provider metadata.

**Requirements Covered:** FR-1.

Sprint-slicing guard: before sprint commitment, split this story by persistence, adapter, projection, or UI display work if it cannot be implemented, tested, linted, and validated in one implementation cycle.
Any split must preserve dependency order, fixture prerequisites, and NFR coverage from the parent story.

Depends on: parent Story 1.2 and parent Story 1.3a.
Contract needed: Project Binding persistence rules, workflow provider binding metadata shape, and uniqueness constraints.
Blocking behavior: Project Binding implementation cannot complete until migration and uniqueness tests prevent ambiguous bindings.
Integration validation: Hermes tests validate profile, cwd, GitHub context, BMAD mount, and provider metadata against shared fixtures.

**Acceptance Criteria:**

**Given** no Project Binding exists for a selected profile
**When** the user creates one with required metadata
**Then** Hermes persists it and shows it after restart
**And** Hermes marks it active only after validation passes.

**Given** cwd is missing or outside allowed workspace roots
**When** validation runs
**Then** Hermes rejects the binding and shows an actionable diagnostic.

**Given** binding metadata conflicts with an existing binding
**When** the user creates or updates a binding
**Then** Hermes blocks ambiguous automation and shows the conflict.

**Given** an existing Project Binding has valid updated metadata
**When** the user updates display name, GitHub reference, BMAD mount path, or workflow provider binding metadata
**Then** Hermes persists the updated binding
**And** Hermes preserves the binding id, audit history, and validation state transition.

**Given** an existing Project Binding is disabled
**When** Hermes restarts and the user views project bindings
**Then** Hermes still shows the binding as disabled
**And** BMAD and workflow provider actions remain blocked for that binding.

**Given** a disabled Project Binding is repaired and re-enabled
**When** validation passes for cwd, GitHub reference, BMAD mount, and workflow provider metadata
**Then** Hermes marks the binding enabled and valid
**And** workflow actions become eligible again.

**Given** the user views Project Bindings
**When** binding status is displayed
**Then** Hermes shows display name, profile identity, Bound Project Cwd, GitHub reference, BMAD mount status, workflow provider binding status, enabled state, and validation state.

### Story H1.2: Mount Project-Local BMAD Skills

As a workflow operator,
I want Hermes to mount project-local BMAD skills for a selected Project Binding,
So that Hermes discovers and runs the correct BMAD workflows for that project.

**Requirements Covered:** FR-2.

Depends on: Story H1.1.
Contract needed: Project Binding BMAD skill directory field, profile `skills.external_dirs` update rule, and mount validation state.
Blocking behavior: BMAD mounting cannot complete until Hermes can associate a mounted directory with a valid enabled Project Binding.
Integration validation: Hermes reloads profile skill index, distinguishes project-local skills from global skills, and blocks execution for missing or wrong-project mounts.

**Acceptance Criteria:**

**Given** a valid enabled Project Binding with a BMAD skill directory
**When** the user mounts BMAD skills
**Then** Hermes adds the directory to the selected profile's `skills.external_dirs`
**And** Hermes reloads skill discovery.

**Given** the directory is missing or points to a different project
**When** validation runs
**Then** Hermes marks the mount invalid and prevents BMAD workflow execution.

**Given** the selected profile already contains the Project Binding BMAD skill directory in `skills.external_dirs`
**When** the user mounts BMAD skills again
**Then** Hermes leaves a single normalized entry for that directory
**And** Hermes does not duplicate `skills.external_dirs` entries.

**Given** profile skill-index reload fails after `skills.external_dirs` is changed
**When** Hermes reports the mount result
**Then** Hermes records the reload failure as a mount diagnostic
**And** Hermes does not mark the mount valid until skill discovery confirms the project-local BMAD skills are visible.

### Story H1.3: Enforce Bound Project Cwd

As a workflow operator,
I want Hermes to run BMAD and workflow provider actions from the Project Binding cwd,
So that workflow artifacts and execution belong to the intended project.

**Requirements Covered:** FR-3.

Depends on: Story H1.1 and Story H1.2.
Contract needed: Bound Project Cwd authority rule, provider adapter cwd expectation when applicable, and workflow action audit shape.
Blocking behavior: Cwd enforcement cannot complete until BMAD and workflow provider adapters receive the selected Project Binding cwd.
Integration validation: Hermes adapter tests prove valid cwd use and reject missing or invalid cwd before external invocation.

**Acceptance Criteria:**

**Given** an enabled Project Binding
**When** the user starts a BMAD or workflow provider action
**Then** Hermes runs the action from the Bound Project Cwd
**And** Hermes records the cwd used.

**Given** no Bound Project Cwd exists
**When** the user attempts a workflow action
**Then** Hermes blocks the action and explains the missing cwd requirement.

### Story H1.4: Invoke BMAD Planning Workflows

As a workflow operator,
I want Hermes to invoke BMAD planning workflows for a bound project,
So that planning artifacts are created and tracked from the correct project.

**Requirements Covered:** FR-4.

Depends on: Story H1.1, Story H1.2, and Story H1.3.
Contract needed: Supported BMAD workflow list, mounted skill discovery result, Bound Project Cwd execution rule, and produced artifact path record.
Blocking behavior: BMAD invocation cannot complete until Hermes proves the workflow exists in the mounted project-local skills and runs from the bound cwd.
Integration validation: Hermes invokes supported workflows from the Bound Project Cwd and records produced artifact paths.
Implementation scope: generic BMAD workflow discovery, invocation, Bound Project Cwd execution, artifact path recording, and failure diagnostics.
This story does not implement bespoke workflow-specific UX or behavior for each named BMAD workflow.

**Acceptance Criteria:**

**Given** a valid binding, cwd, and BMAD mount
**When** the user selects a supported BMAD workflow
**Then** Hermes invokes it from the Bound Project Cwd
**And** Hermes records workflow name, cwd, Project Binding id, profile identity, and result state.

**Given** Hermes exposes supported BMAD planning workflows for a Project Binding
**When** the supported-workflow list is built
**Then** it includes brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation when those workflows are present in the mounted project-local BMAD skill directory
**And** Hermes reports each missing workflow by workflow name, mount source, and Project Binding.

**Given** two supported BMAD planning workflows use the same invocation contract
**When** Hermes invokes either workflow from a Project Binding
**Then** Hermes uses the same generic invocation adapter, cwd enforcement, result capture, artifact recording, and diagnostic path
**And** Hermes does not add workflow-specific branching unless the workflow declares a distinct contract.

### Story H1.5: Materialize Sprint Status Into Project Work Items

As a workflow operator,
I want Hermes to materialize BMAD `sprint-status.yaml` into Project Work Items,
So that BMAD planning becomes an operational backlog without becoming the runtime queue.

**Requirements Covered:** FR-5.

Depends on: Story H1.1 and Story H1.4.
Contract needed: Project Work Item identity fixture, supported `sprint-status.yaml` examples, Project Work Item persistence shape, and idempotent upsert rule.
Blocking behavior: Materialization cannot complete until unchanged, changed, malformed, missing, and duplicate-prevention fixtures pass.
Integration validation: Re-running materialization updates existing Project Work Items and records provenance.

**Acceptance Criteria:**

**Given** `sprint-status.yaml` contains BMAD epics and stories
**When** Hermes materializes it
**Then** Hermes creates or updates Project Work Items with artifact references and source metadata
**And** Hermes does not create duplicates on rerun.

**Given** a BMAD story changes in `sprint-status.yaml`
**When** materialization runs again
**Then** Hermes derives the same stable Project Work Item identity from bound project cwd, BMAD artifact path, and BMAD story identity
**And** Hermes does not include phase kind in Project Work Item identity.

**Given** sprint status is missing, malformed, or unsupported
**When** materialization runs
**Then** Hermes rejects it before mutating project work.

### Story H1.6: Create Phase Tasks And Product Work Lanes

As a workflow operator,
I want each BMAD story represented as linked Prepare Story and Implement Story phase tasks,
So that I can choose and track story work without confusing BMAD status with Hermes runtime state.

**Requirements Covered:** FR-6, FR-11.

Depends on: Story H1.1 and Story H1.5.
Contract needed: Project Work Item identity, Phase Task identity, phase kind, phase task link, reserved gate metadata, and canonical Kanban status vocabulary.
Blocking behavior: Phase task creation cannot complete until repeated materialization proves stable Prepare Story and Implement Story identities.
Integration validation: Idempotency tests prove duplicate materialization does not duplicate phase tasks or reserved gate metadata.

**Acceptance Criteria:**

**Given** a BMAD story has been materialized
**When** Hermes creates phase tasks
**Then** Hermes creates linked Prepare Story and Implement Story tasks
**And** Implement Story remains unreleased until Prepare Story completes.

**Given** a BMAD story has one Project Work Item identity
**When** Hermes creates Prepare Story and Implement Story phase tasks
**Then** each Phase Task identity is derived from the Project Work Item identity plus phase kind
**And** repeated materialization does not create duplicate phase tasks for either phase kind.

**Given** Hermes displays Product Work lanes
**When** Review Test Cases or Verify Done is shown
**Then** the lane is derived from canonical `blocked` plus gate metadata
**And** Hermes does not introduce a new canonical Kanban status for either facade lane.

## Epic 2: Workflow Provider Integration Consumers

Hermes consumes provider binding, workflow command, workflow event, and delivery health surfaces through explicit contracts.
Provider `archon` supplies the first producer fixtures.

### Story H2.1: Register And Diagnose Workflow Provider Bindings From Hermes

As a workflow operator,
I want Hermes to register and inspect workflow provider bindings using generic identity,
So that providers can route workflow events without Hermes-specific vocabulary.

**Requirements Covered:** FR-7.

Depends on: Story H1.1 and provider `archon` Story A2.1.
Contract needed: Workflow Provider Binding payload schema, generic `provider` and `name`, status result shape, and malformed JSON failure shape.
Blocking behavior: Hermes registration cannot complete until provider `archon` exposes the generic provider binding lifecycle surface.
Integration validation: Hermes parses workflow provider binding fixtures and blocks automation on metadata conflict.

**Acceptance Criteria:**

**Given** a valid Hermes Project Binding
**When** the user registers the workflow provider binding
**Then** Hermes invokes generic provider adapter commands
**And** Hermes records binding state and diagnostics.

### Story H2.2a: Start And Inspect Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to start and inspect provider workflow runs through parseable adapter results,
So that Hermes can create and refresh workflow references without relying on a provider dashboard.

**Requirements Covered:** FR-8.

Depends on: Story H1.3, Story H2.1, provider `archon` Story A3.1a, and provider `archon` Story A3.1b.
Contract needed: Workflow command start, status, timeout, success, and error envelope schemas.
Blocking behavior: Hermes start and status control cannot complete until provider workflow command fixtures exist and Hermes fails closed on malformed or incompatible results.
Integration validation: Hermes adapter tests parse start and status fixtures, invoke from Bound Project Cwd, and update only allowed workflow reference or diagnostic state.

**Acceptance Criteria:**

**Given** a valid Project Binding and workflow provider binding
**When** the user starts a provider workflow run
**Then** Hermes invokes the selected provider adapter from the Bound Project Cwd when required
**And** Hermes records stdout, stderr, exit code, timeout, correlation id, workflow name, workflow run reference, and parsed JSON.

**Given** Hermes needs workflow status
**When** status is requested by the user or reconciliation
**Then** Hermes invokes provider status through the strict provider adapter
**And** Hermes updates only workflow reference or diagnostic state allowed by the parsed schema.

### Story H2.2b: Send Provider Decision Commands From Hermes

As a workflow operator,
I want Hermes to send approve and reject commands to the selected provider through parseable adapter results,
So that human gate decisions can drive workflow progress without conflating user approval with command execution.

**Requirements Covered:** FR-8, FR-14.

Depends on: Story H2.2a, provider `archon` Story A3.1a, and provider `archon` Story A3.1c.
Contract needed: Workflow command approve, reject, timeout, success, and error envelope schemas.
Blocking behavior: Decision command handling cannot complete until Hermes records human decision state separately from provider command results.
Integration validation: Hermes adapter tests parse approve and reject fixtures and fail closed on malformed JSON.

**Acceptance Criteria:**

**Given** a workflow run is waiting for a decision
**When** the user approves or rejects from Hermes
**Then** Hermes sends the matching provider adapter command
**And** Hermes records the command result separately from the human gate decision.

### Story H2.2c: Resume, Retry, And Cancel Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to resume, retry, and cancel provider workflow runs through parseable adapter results,
So that recovery actions are recorded consistently and do not depend on human-readable output.

**Requirements Covered:** FR-8.

Depends on: Story H2.2a, provider `archon` Story A3.1a, and provider `archon` Story A3.1d.
Contract needed: Workflow command resume, retry, cancel, timeout, success, and error envelope schemas.
Blocking behavior: Recovery command handling cannot complete until resulting run state or diagnostic state is recorded without relying on human-readable output.
Integration validation: Hermes adapter tests parse resume, retry, cancel, timeout, and unexpected-state fixtures.

**Acceptance Criteria:**

**Given** a workflow run can be resumed, retried, or cancelled
**When** the user selects that action from Hermes
**Then** Hermes invokes the matching provider adapter command
**And** Hermes records the resulting run state or diagnostic without relying on human-readable output.

### Story H2.3a: Validate Profile-Routed Workflow Event Ingress

As a workflow operator,
I want Hermes to validate workflow provider events against the profile route and profile-scoped secret before mutation,
So that events cannot cross project, binding, provider, or profile boundaries.

**Requirements Covered:** FR-9, NFR-6.

Depends on: Story H1.1, Story H2.1, and provider `archon` Story A4.1.
Contract needed: Workflow event envelope, workflow event rejection examples, Project Binding identity, profile route, profile-scoped secret, provider identity, and rejection diagnostic shape.
Blocking behavior: Workflow event ingress cannot complete until invalid events are rejected before mutation and without exposing secrets.
Integration validation: Hermes accepts valid workflow event fixtures and rejects bad signature, stale timestamp, wrong binding, unknown project, schema mismatch, unsupported provider, and valid signature under the wrong profile secret.

**Acceptance Criteria:**

**Given** a provider delivers an outbox event
**When** Hermes receives it on `/p/{profile}/webhooks/workflow-events/{provider}`
**Then** Hermes validates schema, signature, replay, idempotency, profile route, profile-scoped secret, provider, codebase, binding, and authorization
**And** Hermes rejects invalid events before mutation.

**Given** Hermes receives a workflow event signed with a valid secret for the wrong profile
**When** validation runs
**Then** Hermes rejects the event before mutation
**And** Hermes records a redacted wrong-profile-secret diagnostic.

### Story H2.3b: Persist Workflow Event Idempotency Receipts And Duplicate Diagnostics

As a workflow operator,
I want Hermes to persist workflow event idempotency receipts and duplicate diagnostics,
So that redelivery does not create duplicate workflow references, gates, comments, or project-work transitions.

**Requirements Covered:** FR-9.

Depends on: Story H2.3a.
Contract needed: Workflow event idempotency receipt shape, duplicate event id rule, duplicate idempotency key rule, duplicate-safe marker, and diagnostic shape.
Blocking behavior: Receipt handling cannot complete until duplicate workflow events are classified without applying duplicate mutations.
Integration validation: Duplicate workflow event fixtures are accepted as duplicate-safe receipts without duplicate mutation.

**Acceptance Criteria:**

**Given** a duplicate workflow event arrives
**When** Hermes processes it
**Then** Hermes records duplicate-safe receipt state
**And** Hermes does not duplicate workflow references, gates, comments, or project-work transitions.

### Story H2.3c: Map Accepted Workflow Events To Project Work

As a workflow operator,
I want Hermes to map accepted workflow provider events to the correct project work,
So that workflow events update only the intended Project Work Item or phase task.

**Requirements Covered:** FR-9, FR-11, FR-12, FR-13.

Sprint-slicing guard: before sprint commitment, split this story by persistence, adapter, projection, or UI display work if it cannot be implemented, tested, linted, and validated in one implementation cycle.
Any split must preserve dependency order, fixture prerequisites, and NFR coverage from the parent story.

Depends on: Story H1.6, Story H2.1, Story H2.3a, Story H2.3b, and provider `archon` Story A4.1.
Contract needed: Workflow event envelope, Project Work Item identity, Phase Task identity, workflow reference, event-to-mutation map, and deliver-only notification marker.
Blocking behavior: Workflow event mutation cannot complete until accepted events map to existing Project Work Items or phase tasks without creating unintended work.
Integration validation: Accepted workflow completion, workflow failure, approval requested, and workflow artifact fixtures map to the correct Project Work Item or phase task.

**Acceptance Criteria:**

**Given** Hermes accepts a valid workflow event
**When** the event represents workflow completion, workflow failure, approval requested, or workflow artifact
**Then** Hermes maps the event to the correct Project Work Item, phase task, workflow reference, gate, comment, or artifact reference
**And** Hermes distinguishes typed state mutation events from deliver-only notifications.

### Story H2.4: Surface Workflow Event Delivery And Outbox Health

As a workflow operator,
I want Hermes to display workflow event delivery and outbox health,
So that delayed, failed, duplicated, or reconciliation-pending events are visible and actionable.

**Requirements Covered:** FR-10.

Depends on: Story H2.3a, Story H2.3b, and provider `archon` Story A4.2.
Contract needed: Workflow delivery status schema, retry state, terminal failure category, duplicate-safe marker, and reconciliation-needed marker.
Blocking behavior: Hermes health display cannot complete until provider delivery status fixtures exist.
Integration validation: Provider delivery status fixtures drive Hermes states without mutating project work incorrectly.

**Acceptance Criteria:**

**Given** a Project Work Item has a provider workflow reference
**When** Hermes displays workflow status
**Then** Hermes shows workflow event delivery state and links it to the workflow run and Project Binding.

## Epic 3: Human-Gated Story Execution

Hermes runs Prepare Story and Implement Story workflows while blocking for the correct human gates.

### Gate Decision Ownership Convention

- Story H2.2b owns provider approve and reject command transport, strict adapter parsing, fail-closed command-result handling, and command-result persistence only.
- Story H2.2b does not own HILT Gate decision persistence or phase-task gate transitions.
- Story H3.1 owns the Prepare Story transition from blocked `test_case_adequacy` to Implement Story release or Prepare Story recovery.
- Story H3.1 does not define a general HILT Gate decision model beyond the minimum phase-specific transition data needed for Prepare Story flow.
- Story H3.2 owns the Implement Story transition from blocked `done_verification` to story completion or Implement Story recovery.
- Story H3.2 does not define a general HILT Gate decision model beyond the minimum phase-specific transition data needed for Implement Story flow.
- Story H3.3 owns shared HILT Gate decision record hardening, evidence display, audit fields, rejection reason capture, recovery-action selection, delayed prompt behavior, and separation between human decision records and provider command results.
- The persisted human decision record is authoritative for approval or rejection.
- Provider command success is transport evidence only and must not be treated as proof that gate evidence was sufficient.

### Story H3.1: Run Prepare Story And Block For Test-Case Adequacy

As a workflow operator,
I want Hermes to run Prepare Story and block for test-case adequacy,
So that implementation does not begin until tests are human-approved.

**Requirements Covered:** FR-12, FR-14.

Depends on: Story H1.6, Story H2.2a, Story H2.2b, Story H2.3a, Story H2.3b, Story H2.3c, and Story H2.4.
Contract needed: Prepare Story workflow result, workflow completion event, phase task identity, test-case adequacy gate record, and gate decision command result.
Blocking behavior: Prepare Story cannot complete until provider workflow evidence can produce a blocked test-case adequacy gate.
Integration validation: Prepare Story starts through the provider adapter, completes through workflow event or reconciliation, blocks on `test_case_adequacy`, and releases Implement Story only after approval.

**Acceptance Criteria:**

**Given** a Project Work Item has linked phase tasks
**When** the user starts Prepare Story
**Then** Hermes starts the configured provider workflow and records the run reference.

**Given** Prepare Story completes
**When** evidence arrives
**Then** Hermes blocks on `test_case_adequacy` and does not release Implement Story until approval.

**Given** a phase task is blocked on the test-case adequacy gate
**When** gate evidence is displayed
**Then** Hermes distinguishes test-case adequacy from done verification and shows affected project work, phase task, BMAD story, provider run, workflow event receipt when available, evidence references, and recovery action.

### Story H3.2: Run Implement Story And Block For Done Verification

As a workflow operator,
I want Hermes to run Implement Story and block for done verification,
So that implementation is only complete after evidence and human review support it.

**Requirements Covered:** FR-13, FR-14.

Depends on: Story H3.1.
Contract needed: Implement Story workflow result, fix-loop result, workflow completion event, PR reference, done verification gate record, and gate decision command result.
Blocking behavior: Implement Story cannot start until Prepare Story is approved and cannot complete until done verification is approved.
Integration validation: Implement Story blocks on `done_verification`, rejects GitHub merge alone as completion proof, and routes rejection to rerun or recovery.

**Acceptance Criteria:**

**Given** Prepare Story is approved
**When** the user starts Implement Story
**Then** Hermes starts the implementation workflow and fix loop
**And** Hermes records the workflow run reference.

**Given** implementation completes
**When** evidence arrives
**Then** Hermes blocks on `done_verification` until human approval.

**Given** a phase task is blocked on the done verification gate
**When** gate evidence is displayed
**Then** Hermes distinguishes done verification from test-case adequacy and shows affected project work, phase task, BMAD story, provider run, workflow event receipt when available, evidence references, and recovery action.

### Story H3.3: Capture Human Gate Decisions And Evidence

As a workflow operator,
I want Hermes to present evidence and capture approval or rejection,
So that every HILT Gate is auditable and can drive required provider control actions.

**Requirements Covered:** FR-14.

Depends on: Story H2.2b, Story H2.2c, Story H3.1, and Story H3.2.
Contract needed: Gate decision record, evidence reference shape, provider approve, reject, resume, and retry command result schemas.
Blocking behavior: Gate capture cannot complete until Hermes can persist replay-safe decisions and send required provider commands separately from the decision record.
Integration validation: Approval and rejection fixtures persist actor, timestamp, gate kind, decision, evidence references, reason, selected recovery action, command result, and resulting phase state.

**Acceptance Criteria:**

**Given** a phase task is blocked on a gate
**When** Hermes displays the gate
**Then** Hermes shows gate kind, affected work, evidence references, workflow run reference, and available decisions.

**Given** the user approves or rejects
**When** Hermes records the decision
**Then** Hermes stores decision evidence and routes the phase task accordingly.

**Given** Hermes prompts for a gate decision
**When** the prompt is displayed in the dashboard or sent through an existing notification channel
**Then** the prompt identifies Project Binding, BMAD story, phase task, gate kind, and required decision
**And** the prompt excludes secrets, raw workflow event signatures, and unredacted command output.

## Epic 4: Timeline, Reconciliation, And Diagnostics

Hermes explains story state and repairs or surfaces drift across BMAD, Hermes, workflow providers, GitHub, workflow events, and gates.

### Story H4.1: Render Unified Story Timeline

As a workflow operator,
I want one Story Timeline for each BMAD story,
So that I can understand status, evidence, ownership, and next action.

**Requirements Covered:** FR-15.

Depends on: Story H1.6, Story H2.2a, Story H2.2b, Story H2.2c, Story H2.3a, Story H2.3b, Story H2.3c, Story H2.4, Story H3.1, Story H3.2, and Story H3.3.
Contract needed: Project Work Item state, phase task state, provider run reference, workflow event receipt, GitHub PR reference, HILT Gate decision record, and next-action vocabulary.
Blocking behavior: Timeline cannot complete until it can render provenance without collapsing source-specific lifecycle state.
Integration validation: Timeline fixture shows BMAD, Hermes, workflow provider, GitHub, workflow event, and gate state with redacted sensitive details.

**Acceptance Criteria:**

**Given** a Project Work Item links to a BMAD story
**When** the user opens its Story Timeline
**Then** Hermes shows milestones, phase states, workflow runs, workflow events, PR references, gate decisions, and next action.

**Given** a Story Timeline contains events from multiple systems
**When** the timeline is displayed
**Then** each entry is source-labeled as BMAD, Hermes, workflow provider, GitHub, workflow event, reconciliation, or human decision.

### Story H4.2a: Reconcile BMAD Materialization Drift

As a workflow operator,
I want Hermes to reconcile BMAD artifact changes against Project Work Items and phase tasks,
So that changed planning artifacts update existing project work without duplicating work items or phase tasks.

**Requirements Covered:** FR-16.

Depends on: Story H1.5, Story H1.6, and Story H4.1.
Contract needed: BMAD source-state adapter, reconciliation result record, deterministic materialization repair rule, unresolved conflict marker, Project Work Item identity, and Phase Task identity.
Blocking behavior: BMAD materialization reconciliation cannot complete until missing sprint status, malformed sprint status, changed stories, unchanged stories, and duplicate-safe Project Work Item updates are represented as fixtures.
Integration validation: BMAD drift fixtures repair deterministic materialization gaps, preserve unresolved conflicts, avoid duplicate Project Work Items or phase tasks, and never auto-approve a HILT Gate.

**Acceptance Criteria:**

**Given** BMAD source state differs from materialized project work
**When** reconciliation runs
**Then** Hermes records checked source, detected drift, repair action when deterministic, and unresolved conflict when not deterministic
**And** Hermes does not duplicate Project Work Items or phase tasks.

### Story H4.2b: Reconcile Provider Workflow And Event Drift

As a workflow operator,
I want Hermes to reconcile provider workflow state and workflow event evidence against phase task state,
So that workflow event loss, duplicate delivery, gateway downtime, and provider command failures are detected and safely repaired or surfaced.

**Requirements Covered:** FR-16.

Sprint-slicing guard: before sprint commitment, split this story by persistence, adapter, projection, or UI display work if it cannot be implemented, tested, linted, and validated in one implementation cycle.
Any split must preserve dependency order, fixture prerequisites, and NFR coverage from the parent story.

Depends on: Story H1.6, Story H2.2a, Story H2.2c, Story H2.3a, Story H2.3b, Story H2.3c, Story H2.4, Story H3.1, Story H3.2, Story H3.3, Story H4.1, and Story H4.2a.
Contract needed: Provider workflow source-state adapter, workflow event receipt source-state adapter, reconciliation result record, deterministic repair rule, unresolved conflict marker, and event-loss diagnostic shape.
Blocking behavior: Provider workflow and event reconciliation cannot complete until workflow event loss, duplicate delivery, gateway downtime, provider command failure, and terminal run state versus Hermes phase task state are represented as fixtures.
Integration validation: Provider workflow and event drift fixtures repair deterministic projection gaps, preserve unresolved conflicts, avoid duplicate workflow references or gates, and never auto-approve a HILT Gate.

**Acceptance Criteria:**

**Given** provider workflow state or workflow event evidence differs from Hermes phase task state
**When** reconciliation runs
**Then** Hermes records checked sources, drift, deterministic repair action when available, and unresolved conflict when needed
**And** Hermes does not apply duplicate project-work mutation.

### Story H4.2c: Reconcile GitHub And Done Verification Conflicts

As a workflow operator,
I want Hermes to reconcile GitHub PR state against Done Verification Gate state,
So that merged code never counts as completed story work without human done verification.

**Requirements Covered:** FR-16.

Depends on: Story H3.2, Story H3.3, Story H4.1, Story H4.2a, and Story H4.2b.
Contract needed: GitHub PR source-state adapter, done verification gate state, reconciliation result record, unresolved completion conflict marker, and recovery option vocabulary.
Blocking behavior: GitHub and done-verification reconciliation cannot complete until merged PR, unresolved Done Verification Gate, rejected Done Verification Gate, and completion diagnostic fixtures exist.
Integration validation: GitHub and done-verification fixtures preserve completion conflicts, show recovery options, and never mark a story complete without done verification approval.

**Acceptance Criteria:**

**Given** GitHub PR state indicates merged but Done Verification Gate is unresolved or rejected
**When** reconciliation evaluates completion
**Then** Hermes records the conflict
**And** Hermes does not mark the story complete without done verification approval.

### Story H4.3: Surface Operational Diagnostics And Recovery Paths

As a workflow operator,
I want Hermes to surface actionable diagnostics,
So that I can distinguish configuration issues, user decisions, external delays, and implementation defects.

**Requirements Covered:** FR-17.

Sprint-slicing guard: before sprint commitment, split this story by persistence, adapter, projection, or UI display work if it cannot be implemented, tested, linted, and validated in one implementation cycle.
Any split must preserve dependency order, fixture prerequisites, and NFR coverage from the parent story.

Depends on: Story H1.1, Story H1.3, Story H1.5, Story H2.1, Story H2.2a, Story H2.2b, Story H2.2c, Story H2.3a, Story H2.3b, Story H2.3c, Story H2.4, Story H3.1, Story H3.2, Story H3.3, Story H4.1, Story H4.2a, Story H4.2b, and Story H4.2c.
Contract needed: Diagnostic category vocabulary, affected-resource reference shape, recovery option vocabulary, redaction rule, and diagnostic resolution record.
Blocking behavior: Diagnostics cannot complete until planned configuration, user-decision, external-delay, implementation-defect, duplicate-event, outbox-backlog, stale-PR, and unresolved-gate cases map to recovery paths.
Integration validation: Diagnostic fixtures show category, severity, affected reference, redacted evidence, owner of next action, recovery path, resolution source, timestamp, and resulting state.

**Acceptance Criteria:**

**Given** Hermes detects a workflow orchestration problem
**When** diagnostics are generated
**Then** Hermes records category, severity, affected reference, owner of next action, recovery path, and redacted evidence.

**Given** Hermes displays a diagnostic
**When** the diagnostic is shown
**Then** Hermes includes severity, affected reference, redacted evidence, next action owner, recovery path, and resulting state when resolved.

## Validation

Run from inside `hermes-agent`.

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Hermes stories must prove idempotency, fail-closed adapters, duplicate-safe workflow events, replay-safe gates, redaction, reconciliation, and actionable diagnostics.
