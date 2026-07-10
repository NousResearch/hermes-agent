---
title: hermes-agent Epics Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-02'
updated: '2026-07-10'
localContractPackage: _bmad-output/planning-artifacts/contracts/workflow-commander/
storyOwnershipNote: >
  Story numbering is kept identical to the parent workspace's epics.md (Epics 2-5)
  so cross-references between hermes-agent, Archon, and the parent stay unambiguous.
  This file contains ONLY hermes-agent-owned stories. Parent Epic 1 (contract
  fixtures and handoff generation) is parent-workspace work, not something
  hermes-agent implements.
---

# hermes-agent Epics: Hermes Agent Workflow Commander

## Overview

This file contains the hermes-agent-owned subset of the 43 parent stories across Epics 2-5. It excludes Archon-owned producer work except where dependency notes are required for integration.

**Local contract package:** the shared Workflow Commander contract package now exists in `_bmad-output/planning-artifacts/contracts/workflow-commander/` as of 2026-07-09.
This resolves the local planning-artifact fixture presence blocker identified by the implementation-readiness report.
No story should move to implementation-ready until its implementation tests load the exact schemas and examples it consumes from that package and, for provider-facing stories, prove compatibility with the Archon producer-side story listed in the dependency note.

## NFR Coverage Map

The PRD defines NFR-1 through NFR-17.
This map makes the traceability explicit for implementation and QA planning.

| NFR | Primary Story Coverage | Contract Or Test Evidence |
| --- | --- | --- |
| NFR-1 workflow events are acceleration, not sole truth | Stories 3.8, 5.2b–5.2d | Delivery-status fixtures and reconciliation fixtures must show event loss does not complete work. |
| NFR-2 reconcile after loss, duplicates, downtime, command failure, or manual PR merge | Stories 5.2a–5.2e, 5.3a–5.3c | Materialization, delivery, provider-state, and PR/done-verification drift fixtures. |
| NFR-3 materialization is idempotent | Stories 2.5, 2.6, 5.2a | `materialization/unchanged-story.json`, `changed-story.json`, and `duplicate-phase-task-prevention.json`. |
| NFR-4 gate decisions are replay-safe and auditable | Story 4.3 | Gate decision tests must persist actor, timestamp, evidence references, decision, reason, and command result separation. |
| NFR-5 reject events failing signature, schema, replay, binding, provider, or authorization checks | Stories 3.6a, 3.6b | Workflow event rejection fixtures cover signature, schema, replay, binding, provider, and authorization failures. |
| NFR-6 scope event secrets to the correct profile | Story 3.6a | `wrong-profile-secret.json` and profile-routed webhook tests. |
| NFR-7 prevent workflow actions outside Bound Project Cwd | Story 2.3 | Adapter tests must assert BMAD and provider actions receive the Project Binding cwd. |
| NFR-8 redact secrets in logs, diagnostics, command output, events, and status-history results | Stories 3.6a, 4.3, 5.1, 5.3a, 5.3b | Rejection, gate, status-history, and diagnostic tests must assert redacted evidence. |
| NFR-9 persist commands, events, reconciliation actions, gate decisions, and state transitions | Stories 2.3, 3.4a–3.4c, 3.6b, 4.3, 5.2a–5.2e, 5.3a, 5.3c | Persistence tests cover audit records, command envelopes, receipts, decisions, reconciliation results, and diagnostic resolution. |
| NFR-10 Story Status History explains why state changed | Story 5.1 | Status-history fixtures return source-labeled BMAD, Hermes, provider, GitHub, workflow event, reconciliation, and human decision entries. |
| NFR-11 state changes carry provenance by source | Stories 5.1, 5.2a–5.2e, 5.3a–5.3c | Status-history, reconciliation, and diagnostic fixtures include source, affected reference, and resulting state. |
| NFR-12 next actions use user-facing language | Stories 4.3, 5.1, 5.3b | Gate, status-history, and diagnostic query tests avoid backend-only state names as the user-facing next action. |
| NFR-13 distinguish Done Verification Gate approval from GitHub PR merge | Stories 4.2, 5.1, 5.2e | Completion fixtures keep PR state and done-verification state separate. |
| NFR-14 surface blockers with recovery options, not raw stack traces | Stories 2.1b, 2.1c, 2.4, 3.4a–3.4c, 5.3a–5.3c | Diagnostic fixtures include category, severity, owner, recovery path, and redacted evidence. |
| NFR-15 preserve bounded ownership between Hermes, BMAD, providers, and GitHub | Stories 2.5, 3.2, 3.8, 5.2a–5.2e | Tests must keep BMAD artifacts, provider state, GitHub PR state, and Hermes project work as separate sources. |
| NFR-16 provider integration surfaces stay generic | Stories 3.2, 3.4a, 3.4b, 3.4c, 3.6a, 3.6b, 3.6c, 3.8 | Schemas use generic `provider` and `name` vocabulary with provider-specific examples under `providers/archon/`. |
| NFR-17 cross-project handoffs are complete enough for isolated agents | All provider-facing stories, especially 3.2 through 3.8 | The local contract package and Archon dependency notes are required inputs before implementation-ready status. |

## Epic 2: Project-Bound Planning And Work Backlog

Kevin can bind a project, mount its BMAD skills, run planning from the correct cwd, and see BMAD stories materialized into Hermes project work with linked phase tasks.

### Story 2.1a: Create And Persist Project Bindings

As a workflow operator, I want Hermes to create and retrieve stable Project Bindings so every later action has a persisted project identity.

**Requirements Covered:** FR-1.

**Implementation Scope:** Project Binding schema, migration, create and read operations, restart persistence, stable identity, and persistence-level uniqueness.

Contract needed: Project Binding persistence rules and shared Workflow Provider Binding metadata shape.
Blocking behavior: Migration and uniqueness tests must prove ambiguous bindings cannot be persisted.
Integration validation: A created binding returns after restart with the same identity; uniqueness violations fail without partial writes.

**Acceptance Criteria:**

**Given** no binding exists for a selected profile
**When** an authorized command creates one with all required fields
**Then** Hermes persists it and returns the same stable identity after restart.

**Given** persistence-level uniqueness would be violated
**When** creation runs
**Then** Hermes rejects the record without a partial write and returns a machine-readable conflict.

### Story 2.1b: Validate Project Binding Safety And Conflicts

As a workflow operator, I want Hermes to validate binding safety and conflicts so automation fails closed for unsafe or ambiguous project context.

**Requirements Covered:** FR-1.

**Implementation Scope:** Allowed-root cwd validation, GitHub and BMAD reference validation, provider-metadata validation, cross-binding conflict detection, validation-state transitions, and actionable diagnostics.

Depends on: Story 2.1a.
Blocking behavior: A binding cannot become enabled until validation passes.
Integration validation: Invalid cwd, missing reference, and conflicting metadata fixtures fail closed.

**Acceptance Criteria:**

**Given** cwd or required references are invalid
**When** validation runs
**Then** Hermes rejects activation and returns an actionable structured diagnostic.

**Given** another binding conflicts on governed metadata
**When** validation runs
**Then** Hermes blocks automation and returns both binding references plus the conflict category.

### Story 2.1c: Manage Project Binding Lifecycle And Status

As a workflow operator, I want Hermes to update, disable, repair, re-enable, and query Project Bindings so lifecycle changes remain durable and auditable.

**Requirements Covered:** FR-1.

**Implementation Scope:** Binding update, disable, repair, re-enable, audit preservation, and versioned structured status serialization.

Depends on: Story 2.1b.
Blocking behavior: Disabled or invalid bindings block BMAD and provider actions.
Integration validation: Lifecycle state survives restart and structured results return display name, profile, cwd, GitHub, BMAD mount, provider binding, enabled state, and validation state.

**Acceptance Criteria:**

**Given** an existing binding is updated, disabled, repaired, or re-enabled
**When** the authorized lifecycle command succeeds
**Then** Hermes preserves stable identity and audit history and returns the resulting lifecycle state.

**Given** a caller queries binding status
**When** at least one binding exists
**Then** Hermes returns all required binding and validation fields in a versioned structured result.

---

### Story 2.2: Mount Project-Local BMAD Skills For A Binding

As a workflow operator,
I want Hermes to mount project-local BMAD skills for a selected Project Binding,
So that Hermes can discover and run the correct BMAD workflows for that project without relying on global skills.

**Requirements Covered:** FR-2.

**Implementation Scope:** BMAD skill mount configuration, profile skill-index reload, and mount diagnostics.

Depends on: Story 2.1c (this file).
Contract needed: Project Binding BMAD skill directory field, profile `skills.external_dirs` update rule, and mount validation state.
Blocking behavior: Cannot be marked complete until Hermes can associate a mounted skill directory with a valid enabled Project Binding.
Integration validation: Hermes reloads the selected profile skill index, distinguishes project-local BMAD skills from global skills, and blocks workflow execution when the mount is missing or points at the wrong project.

**Acceptance Criteria:**

**Given** a valid enabled Project Binding with a BMAD skill directory reference
**When** the user mounts BMAD skills for that binding
**Then** Hermes adds the project BMAD skill directory to the selected profile's `skills.external_dirs`
**And** Hermes reloads the profile skill index so the mounted skills appear in skill discovery, skill view, and slash command discovery.

**Given** the selected profile already contains the Project Binding BMAD skill directory in `skills.external_dirs`
**When** the user mounts BMAD skills again
**Then** Hermes leaves a single normalized entry for that directory
**And** Hermes does not duplicate `skills.external_dirs` entries.

**Given** profile skill-index reload fails after `skills.external_dirs` is changed
**When** Hermes reports the mount result
**Then** Hermes records the reload failure as a mount diagnostic
**And** Hermes does not mark the mount valid until skill discovery confirms the project-local BMAD skills are visible.

**Given** a mounted BMAD skill is visible in Hermes
**When** the user inspects the skill source
**Then** Hermes shows the source skill directory and associated Project Binding
**And** Hermes distinguishes project-local BMAD skills from global `~/.hermes/skills`.

**Given** the configured BMAD skill directory is missing
**When** Hermes validates or reloads the Project Binding
**Then** Hermes marks the BMAD mount as invalid
**And** Hermes prevents BMAD workflow execution for that binding until the mount is repaired.

**Given** the configured BMAD skill directory points to a different project than the Bound Project Cwd
**When** Hermes validates the mount
**Then** Hermes reports a wrong-project mount diagnostic
**And** Hermes does not use skill visibility alone to infer artifact placement.

**Given** the user disables a Project Binding
**When** Hermes refreshes profile skill state
**Then** Hermes prevents disabled binding skills from being used as active project workflow actions
**And** Hermes preserves enough mount metadata to re-enable the binding safely later.

---

### Story 2.3: Enforce Bound Project Cwd For Workflow Actions

As a workflow operator,
I want Hermes to run BMAD and workflow provider actions from the Project Binding's explicit cwd,
So that planning artifacts and workflow execution always belong to the intended local project.

**Requirements Covered:** FR-3.

**Implementation Scope:** Workflow-action cwd guard and audit capture.

Depends on: Story 2.1c (this file).
Contract needed: Bound Project Cwd authority rule, provider adapter cwd expectation when applicable, and workflow action audit record shape.
Blocking behavior: Cannot be marked complete until Hermes can prove BMAD and workflow provider action adapters receive the selected Project Binding cwd.
Integration validation: Adapter tests prove BMAD actions and provider adapter calls use the Bound Project Cwd and reject missing or invalid cwd before invoking external workflow actions.

**Acceptance Criteria:**

**Given** an enabled Project Binding with a valid Bound Project Cwd
**When** the user starts a BMAD planning workflow from Hermes
**Then** Hermes runs the workflow with the Bound Project Cwd as the working directory
**And** any BMAD artifacts produced through Hermes land under that project's configured `_bmad-output` location.

**Given** an enabled Project Binding with a valid Bound Project Cwd
**When** the user starts a workflow provider action from Hermes
**Then** Hermes passes the Bound Project Cwd to the provider adapter when the provider requires cwd
**And** Hermes records the cwd used for the action.

**Given** the selected project has no Bound Project Cwd
**When** the user attempts to start any BMAD or workflow provider action
**Then** Hermes blocks the action
**And** Hermes shows a diagnostic that identifies the missing cwd requirement.

**Given** a mounted BMAD skill is visible from a profile
**When** Hermes determines where a workflow action should run
**Then** Hermes uses the active Project Binding cwd, not skill visibility, as the artifact placement and execution authority.

**Given** Hermes executes any project-bound workflow action
**When** the action completes, fails, or is cancelled
**Then** Hermes persists an audit record containing the Project Binding id, profile identity, cwd, command or workflow name, started time, completed time if available, result state, and correlation id if available.

---

### Story 2.4: Invoke BMAD Planning Workflows From Hermes

As a workflow operator,
I want Hermes to invoke BMAD planning workflows for a bound project,
So that I can create or update planning artifacts from Hermes and continue orchestration from those artifacts.

**Requirements Covered:** FR-4.

**Implementation Scope:** Generic BMAD workflow discovery, invocation, Bound Project Cwd execution, artifact path recording, and failure diagnostics. Does not implement bespoke workflow-specific UX or behavior for each named BMAD workflow.

Depends on: Story 2.1c, Story 2.2, Story 2.3 (this file).
Contract needed: Supported BMAD workflow list, Bound Project Cwd execution rule, mounted skill discovery result, and produced artifact path record.
Blocking behavior: Cannot be marked complete until Hermes can prove the selected workflow exists in the mounted project-local skills and runs from the bound cwd.
Integration validation: Hermes invokes supported BMAD planning workflows from the Bound Project Cwd, records produced artifact paths, and blocks unsupported, unavailable, or unmounted workflows.

**Acceptance Criteria:**

**Given** an enabled Project Binding with a valid Bound Project Cwd and valid BMAD mount
**When** the user selects a supported BMAD planning workflow from Hermes
**Then** Hermes invokes that workflow from the Bound Project Cwd
**And** Hermes records the workflow name, cwd, Project Binding id, profile identity, started time, and result state.

**Given** Hermes exposes supported BMAD planning workflows for a Project Binding
**When** the supported-workflow list is built
**Then** it includes brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation when those workflows are present in the mounted project-local BMAD skill directory
**And** Hermes reports each missing workflow by workflow name, mount source, and Project Binding.

**Given** two supported BMAD planning workflows use the same invocation contract
**When** Hermes invokes either workflow from a Project Binding
**Then** Hermes uses the same generic invocation adapter, cwd enforcement, result capture, artifact recording, and diagnostic path
**And** Hermes does not add workflow-specific branching unless the workflow declares a distinct contract.

**Given** a BMAD workflow produces planning artifacts
**When** the workflow completes successfully
**Then** Hermes records the produced artifact paths on the Project Binding or related project-work context
**And** Hermes can show those artifact paths to the user.

**Given** a BMAD workflow fails
**When** Hermes receives the failure result or process output
**Then** Hermes preserves the Project Binding context
**And** Hermes surfaces the failure output as a user-actionable diagnostic without losing the cwd used.

**Given** BMAD planning artifacts already exist for a bound project
**When** the user asks Hermes to continue orchestration from generated artifacts
**Then** Hermes can locate the artifacts from the bound project `_bmad-output` location
**And** Hermes does not search unrelated project roots.

**Given** the selected BMAD workflow is unsupported, unavailable, or missing from the mounted skills
**When** the user attempts to run it through Hermes
**Then** Hermes blocks the invocation
**And** Hermes shows whether the issue is missing skill mount, unsupported workflow, or invalid Project Binding state.

---

### Story 2.5: Materialize Sprint Status Into Project Work Items

As a workflow operator,
I want Hermes to materialize BMAD `sprint-status.yaml` into Hermes Project Work Items,
So that BMAD planning output becomes an operational backlog without making `sprint-status.yaml` the runtime queue.

**Requirements Covered:** FR-5.

**Implementation Scope:** `sprint-status.yaml` reader, Project Work Item identity, and Project Work Item upsert behavior.

Depends on: shared Project Work Item identity fixture in `_bmad-output/planning-artifacts/contracts/workflow-commander/`, Story 2.1c, Story 2.4 (this file).
Contract needed: Project Work Item identity fixture, supported `sprint-status.yaml` examples, Project Work Item persistence shape, and idempotent upsert rule.
Blocking behavior: Cannot be marked complete until unchanged, changed, malformed, missing, and duplicate prevention fixtures pass against the Hermes implementation.
Integration validation: Re-running materialization updates existing Project Work Items rather than creating duplicates and records provenance for each source artifact.

**Acceptance Criteria:**

**Given** an enabled Project Binding with a valid Bound Project Cwd
**When** the user asks Hermes to materialize planning output
**Then** Hermes reads `sprint-status.yaml` only from the bound project cwd
**And** Hermes rejects missing, malformed, or unsupported sprint status data before mutating project work.

**Given** `sprint-status.yaml` contains BMAD epics and stories
**When** Hermes materializes the file
**Then** Hermes creates or updates Project Work Items with BMAD artifact references, observed BMAD status, target BMAD status, and source artifact path metadata
**And** Hermes does not treat the file as the runtime queue after materialization.

**Given** Hermes has already materialized a BMAD story
**When** materialization runs again with unchanged source data
**Then** Hermes updates the existing Project Work Item or leaves it unchanged
**And** Hermes does not create duplicate cards.

**Given** a BMAD story changes in `sprint-status.yaml`
**When** materialization runs again
**Then** Hermes derives the same stable Project Work Item identity from bound project cwd, BMAD artifact path, and BMAD story identity
**And** Hermes updates the existing Project Work Item rather than creating a duplicate
**And** Hermes does not include phase kind in Project Work Item identity.

**Given** materialization partially fails after validation begins
**When** Hermes reports the failure
**Then** Hermes preserves enough provenance to identify which Project Binding, source file, epic, story, or Project Work Item was affected
**And** Hermes does not silently mark work complete.

**Given** Project Work Item persistence schema is created or changed
**When** materialization validation runs
**Then** idempotency tests prove repeated materialization does not duplicate Project Work Items
**And** this story does not create workflow event, gate, timeline, or reconciliation tables that are not needed for materialization.

---

### Story 2.6: Create Phase Task And Operational Work State

As a workflow operator,
I want Hermes to represent each BMAD story as one phase task,
So that I can choose and track story work from Hermes without confusing BMAD story status with Hermes runtime state.

**Requirements Covered:** FR-6, FR-11.

**Implementation Scope:** Phase-task persistence, operational project-work metadata, canonical Kanban status mapping, and structured state results.

Depends on: shared Project Work Item identity fixture in `_bmad-output/planning-artifacts/contracts/workflow-commander/`, Story 2.1c, Story 2.5 (this file).
Contract needed: Project Work Item identity, Phase Task identity, phase task link, reserved gate metadata, and canonical Kanban status vocabulary.
Blocking behavior: Cannot be marked complete until repeated materialization proves a stable, non-duplicating Phase Task identity per Project Work Item.
Integration validation: Idempotency tests prove duplicate materialization does not duplicate the phase task or reserved gate metadata.

**Acceptance Criteria:**

**Given** a BMAD story has been materialized into Hermes project work
**When** Hermes creates the phase task for that story
**Then** Hermes creates exactly one Phase Task linked to the Project Work Item.

**Given** a BMAD story has one Project Work Item identity
**When** Hermes creates its Phase Task
**Then** the Phase Task identity is derived from the Project Work Item identity
**And** repeated materialization does not create a duplicate phase task.

**Given** a phase task is waiting for human review in a later epic
**When** Hermes returns operational work state
**Then** the result uses canonical `blocked` status plus `gate_kind` metadata
**And** Hermes keeps the canonical Kanban lifecycle values unchanged: `triage`, `todo`, `ready`, `running`, `blocked`, `done`, and `archived`.

**Given** a Project Work Item has a phase task
**When** the user inspects project-work metadata
**Then** Hermes returns selected story, phase metadata, workflow references if available, human gate metadata if available, next action, observed BMAD status, target BMAD status, and artifact references in a versioned structured result.

**Given** phase task or gate metadata schema is created or changed
**When** validation runs
**Then** idempotency tests prove repeated materialization does not duplicate the phase task or reserved gate metadata
**And** this story creates only the phase-task and project-work metadata needed before gate execution begins.

## Epic 3 (hermes-agent Consumer Side): Workflow Provider Control And Event Delivery

Kevin can connect Hermes to workflow providers generically, control workflow runs through provider command JSON, and see event delivery or outbox health without opening the provider dashboard.
This is the consumer side only — Archon's producer side lives in `Archon/_bmad-output/planning-artifacts/epics.md`.

### Story 3.2: Register And Diagnose Workflow Provider Bindings From Hermes

As a workflow operator,
I want Hermes to register and inspect workflow provider bindings using generic provider identity,
So that the selected provider can route workflow events to the correct Hermes profile without Hermes-specific vocabulary.

**Requirements Covered:** FR-7.

**Implementation Scope:** Workflow provider binding registration flow, status display, conflict detection, and lifecycle command adapter.

Depends on: Story 2.1c (this file).
**Depends on Archon:** Story 3.1 (Workflow Provider Binding Lifecycle) — must exist for Hermes to register/diagnose bindings against.
Contract needed: Workflow Provider Binding payload schema, generic `provider`/`name` vocabulary, binding status result shape, and malformed JSON failure shape.
Blocking behavior: Cannot be marked complete until Hermes has a Project Binding and provider `archon` exposes the provider binding lifecycle surface.
Integration validation: Hermes parses provider binding fixtures, associates results with Project Bindings, and blocks automation when Hermes Project Binding metadata conflicts with provider binding metadata.

**Acceptance Criteria:**

**Given** a valid Hermes Project Binding with workflow provider metadata
**When** the user registers the workflow provider binding for provider `archon`
**Then** Hermes uses the provider adapter command that identifies the controller by generic `provider` and `name`
**And** Hermes does not require provider commands or models named specifically for Hermes.

**Given** the provider adapter returns binding status
**When** Hermes inspects the binding
**Then** Hermes can show whether the binding is missing, valid, stale, disabled, rotated, or conflicting
**And** Hermes links the diagnostic to the affected Project Binding.

**Given** the provider's stored binding disagrees with Hermes Project Binding metadata
**When** Hermes validates the relationship
**Then** Hermes marks the binding as conflicting
**And** Hermes blocks state-changing automation that depends on that binding until the conflict is resolved.

**Given** a workflow provider binding needs rotation, removal, or disabling
**When** the user performs the lifecycle action from Hermes
**Then** Hermes invokes the generic provider adapter command
**And** Hermes records the CLI result, correlation id, actor, timestamp, and resulting binding state.

**Given** the provider adapter returns malformed JSON or an unexpected exit code for a binding action
**When** Hermes receives the result
**Then** Hermes fails closed
**And** Hermes surfaces an actionable diagnostic without mutating Project Work state.

---

### Story 3.4a: Start And Inspect Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to start and inspect provider workflow runs through parseable adapter results,
So that Hermes can create and refresh workflow references without relying on a provider dashboard.

**Requirements Covered:** FR-8.

**Implementation Scope:** Strict workflow provider adapter for workflow start and status commands. Provider `archon` maps this adapter to Archon CLI JSON in v1.

Depends on: Story 2.1c, Story 2.3, Story 3.2 (this file).
**Depends on Archon:** Story 3.3a (shared command envelope), Story 3.3b (start/status CLI JSON).
Contract needed: Workflow command start, status, timeout, success, and error envelope schemas.
Blocking behavior: Cannot be marked complete until provider workflow command fixtures are available and Hermes can fail closed on malformed or incompatible results.
Integration validation: Adapter tests parse provider `archon` start and status fixtures, invoke from the Bound Project Cwd, record stdout, stderr, exit code, timeout, correlation id, and update only allowed workflow reference or diagnostic state.

**Acceptance Criteria:**

**Given** a valid Project Binding and valid workflow provider binding
**When** the user starts a provider workflow run from Hermes
**Then** Hermes invokes the selected provider adapter from the Bound Project Cwd
**And** Hermes records stdout, stderr, exit code, timeout, correlation id, workflow name, workflow run reference, and parsed JSON result.

**Given** Hermes needs workflow status
**When** the user or reconciliation process requests status for a provider workflow run
**Then** Hermes invokes the provider adapter status command
**And** Hermes updates only the workflow reference or diagnostic state allowed by the parsed schema.

**Given** a provider start or status command returns malformed JSON, a schema-version mismatch, timeout, or unexpected exit code
**When** Hermes processes the result
**Then** Hermes fails closed
**And** Hermes does not update Project Work state as if the command succeeded.

---

### Story 3.4b: Send Provider Decision Commands From Hermes

As a workflow operator,
I want Hermes to send approve and reject commands to the selected workflow provider through parseable adapter results,
So that human gate decisions can drive provider workflow progress without conflating user approval with command execution.

**Requirements Covered:** FR-8, FR-14.

**Implementation Scope:** Strict workflow provider adapter for workflow approve and reject commands, command-result persistence, and fail-closed diagnostics only. Does not persist the authoritative HILT Gate decision record and does not transition phase tasks after gate approval or rejection — those responsibilities remain with Epic 4 phase-specific gate stories and shared decision-record hardening. Provider `archon` maps this adapter to Archon CLI JSON in v1.

Depends on: Story 2.1c, Story 3.2, Story 3.4a (this file).
**Depends on Archon:** Story 3.3a, Story 3.3c (approve/reject CLI JSON).
Contract needed: Workflow command approve, reject, timeout, success, and error envelope schemas.
Blocking behavior: Cannot be marked complete until Hermes records human decision state separately from provider command results.
Integration validation: Adapter tests parse approve and reject fixtures, fail closed on malformed JSON, and keep command results separate from HILT Gate decision records.

**Acceptance Criteria:**

**Given** a workflow run is waiting for a decision
**When** the user approves or rejects from Hermes
**Then** Hermes sends the matching provider command
**And** Hermes records the decision command result separately from the human gate decision and any later resume or rerun action.

**Given** an approve or reject command succeeds
**When** Hermes records the result
**Then** Hermes stores the command result, correlation id, workflow run reference, actor when available, timestamp, and parsed result state
**And** Hermes does not treat command success alone as proof that the human gate evidence was sufficient.

**Given** an approve or reject command returns malformed JSON, a schema-version mismatch, timeout, or unexpected exit code
**When** Hermes processes the result
**Then** Hermes fails closed
**And** Hermes surfaces a diagnostic without mutating Project Work state as if the decision command succeeded.

---

### Story 3.4c: Resume, Retry, And Cancel Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to resume, retry, and cancel provider workflow runs through parseable adapter results,
So that recovery actions are recorded consistently and do not depend on human-readable provider output.

**Requirements Covered:** FR-8.

**Implementation Scope:** Strict workflow provider adapter for workflow resume, retry, cancel, timeout, and unexpected-state handling. Provider `archon` maps this adapter to Archon CLI JSON in v1.

Depends on: Story 2.1c, Story 3.2, Story 3.4a (this file).
**Depends on Archon:** Story 3.3a, Story 3.3d (recovery CLI JSON).
Contract needed: Workflow command resume, retry, cancel, timeout, success, and error envelope schemas.
Blocking behavior: Cannot be marked complete until resulting run state or diagnostic state is recorded without relying on human-readable output.
Integration validation: Adapter tests parse resume, retry, cancel, timeout, and unexpected-state fixtures and update only allowed workflow reference or diagnostic state.

**Acceptance Criteria:**

**Given** a workflow run can be resumed, retried, or cancelled
**When** the user selects that action from Hermes
**Then** Hermes invokes the matching provider adapter command
**And** Hermes records the resulting run state or diagnostic without relying on human-readable output.

**Given** a recovery command succeeds
**When** Hermes records the result
**Then** Hermes stores stdout, stderr, exit code, timeout, correlation id, workflow run reference, action name, and parsed result state
**And** Hermes links the action to the affected Project Binding and workflow reference.

**Given** a resume, retry, or cancel command returns malformed JSON, a schema-version mismatch, timeout, unexpected state, or unexpected exit code
**When** Hermes processes the result
**Then** Hermes fails closed
**And** Hermes surfaces an actionable diagnostic without marking the workflow action successful.

---

### Story 3.6a: Validate Profile-Routed Workflow Event Ingress

As a workflow operator,
I want Hermes to validate workflow provider events against the profile route and profile-scoped secret before mutation,
So that events cannot cross project, binding, provider, or profile boundaries.

**Requirements Covered:** FR-9, NFR-6.

**Implementation Scope:** Hermes-owned workflow event ingress validation before idempotency receipts or project-work mutation.

Depends on: shared workflow event envelope and rejection fixtures in `_bmad-output/planning-artifacts/contracts/workflow-commander/`, Story 2.1c, Story 3.2 (this file).
**Depends on Archon:** Story 3.5 (Archon must produce a valid signed event fixture to validate against).
Contract needed: Workflow event envelope schema, workflow event rejection examples, Project Binding identity, profile route, profile-scoped secret, provider identity, and rejection diagnostic shape.
Blocking behavior: Cannot be marked complete until invalid events are rejected before mutation and without exposing secrets.
Integration validation: Hermes accepts the valid provider `archon` event fixture and rejects bad signature, stale timestamp, wrong binding, unknown project, schema mismatch, unsupported provider, and valid signature under the wrong profile secret.

**Acceptance Criteria:**

**Given** an outbox event is delivered to Hermes
**When** Hermes receives the event on `/p/{profile}/webhooks/workflow-events/{provider}`
**Then** Hermes validates schema version, signature, replay window, provider, event id, event type, binding reference, project or codebase reference, workflow run reference, idempotency key, profile route, and profile-scoped secret
**And** Hermes rejects the event before mutation if any validation fails.

**Given** Hermes receives a workflow event signed with a valid secret for the wrong profile
**When** validation runs
**Then** Hermes rejects the event before mutation
**And** Hermes records a redacted wrong-profile-secret diagnostic.

**Given** Hermes receives a workflow event for an unknown Project Binding, wrong profile, wrong codebase, stale timestamp, invalid signature, wrong binding, unknown project, unsupported provider, or schema mismatch
**When** validation runs
**Then** Hermes rejects the event
**And** Hermes records enough diagnostic context to support investigation without exposing secrets.

---

### Story 3.6b: Persist Workflow Event Idempotency Receipts And Duplicate Diagnostics

As a workflow operator,
I want Hermes to persist workflow event idempotency receipts and duplicate-safe diagnostics,
So that redelivery does not create duplicate workflow references, gates, comments, or project-work transitions.

**Requirements Covered:** FR-9.

**Implementation Scope:** Workflow event receipt persistence, duplicate classification, and duplicate-safe diagnostics.

Depends on: shared workflow event fixtures in `_bmad-output/planning-artifacts/contracts/workflow-commander/`, Story 2.1c, Story 3.6a (this file).
Contract needed: Workflow event idempotency receipt shape, duplicate event id rule, duplicate idempotency key rule, duplicate-safe marker, and diagnostic shape.
Blocking behavior: Cannot be marked complete until duplicate events are classified without applying duplicate mutations.
Integration validation: Duplicate workflow event fixtures are accepted as duplicate-safe receipts without duplicating workflow references, gates, comments, or transitions.

**Acceptance Criteria:**

**Given** Hermes receives a workflow event with a new event id and idempotency key
**When** validation has passed
**Then** Hermes persists a workflow event receipt with Project Binding, provider, workflow run, event type, event id, idempotency key, received time, and processing state
**And** the receipt can be linked from diagnostics and Story Status History.

**Given** Hermes receives a duplicate workflow event id or idempotency key
**When** the event is processed
**Then** Hermes treats delivery as duplicate-safe
**And** Hermes does not create duplicate workflow references, gates, comments, or project-work transitions.

**Given** duplicate workflow event delivery is detected
**When** Hermes records diagnostics
**Then** Hermes records the duplicate-safe marker and affected receipt reference
**And** Hermes preserves enough context to prove no duplicate mutation was applied.

---

### Story 3.6c: Map Workflow Outcome Events To Project Work

As a workflow operator, I want completion and failure events mapped to existing project work so they update only the intended Phase Task.

**Requirements Covered:** FR-9, FR-11, FR-12.

**Implementation Scope:** Workflow-completed and workflow-failed mapping to Project Work Items, Phase Tasks, workflow references, and result state.

Depends on: shared Project Work Item and Phase Task identity fixtures in `_bmad-output/planning-artifacts/contracts/workflow-commander/`, Story 2.1c, Story 2.6, Story 3.2, Story 3.6a, and Story 3.6b.
**Depends on Archon:** Story 3.5.
Blocking behavior: Outcome mapping must not create unintended work.
Integration validation: Completion and failure fixtures map deterministically or preserve state with an unresolved diagnostic.

**Acceptance Criteria:**

**Given** Hermes accepts a workflow-completed or workflow-failed event
**When** it resolves to existing project work
**Then** Hermes updates that Phase Task without creating new work.

**Given** the outcome identity is unresolved
**When** mapping runs
**Then** Hermes records a diagnostic and leaves project-work state unchanged.

### Story 3.6d: Map Approval Request Events To Gate State

As a workflow operator, I want approval-request events mapped to reserved gate state so human decisions are never inferred from event delivery.

**Requirements Covered:** FR-9, FR-13.

Depends on: Story 3.6c.
Blocking behavior: Mapping must not create a duplicate gate or imply approval.
Integration validation: Approval-request fixtures are replay-safe and separate typed mutation from deliver-only notification.

**Acceptance Criteria:**

**Given** Hermes accepts an approval-request event
**When** it maps to an existing Phase Task
**Then** Hermes records the pending gate exactly once and does not record a human decision.

### Story 3.6e: Map Workflow Artifact Events And Diagnose Unresolved Mappings

As a workflow operator, I want artifact events mapped with provenance so evidence is available without unsafe mutation.

**Requirements Covered:** FR-9, FR-12.

Depends on: Story 3.6d.
Blocking behavior: Unresolved artifact mappings leave project-work state unchanged.
Integration validation: Artifact fixtures retain provenance, redact secrets, and create actionable unresolved-mapping diagnostics.

**Acceptance Criteria:**

**Given** an artifact event has resolvable identity
**When** Hermes applies it
**Then** Hermes records the artifact or comment reference with provenance and redaction.

**Given** artifact identity cannot be resolved
**When** mapping runs
**Then** Hermes records a diagnostic and leaves project-work state unchanged.

---

### Story 3.8: Return Workflow Event Delivery And Outbox Health Status

As a workflow operator,
I want Hermes to return workflow event delivery and provider outbox health for each relevant workflow run,
So that delayed, failed, duplicated, or reconciliation-pending events are queryable and actionable.

**Requirements Covered:** FR-10.

**Implementation Scope:** Structured workflow event health result, workflow diagnostic record, and reconciliation-needed marker.

Depends on: Story 2.1c, Story 3.6a, Story 3.6b (this file).
**Depends on Archon:** Story 3.7 (delivery health CLI JSON).
Contract needed: Workflow delivery status schema, retry state, terminal failure category, duplicate-safe marker, and reconciliation-needed marker.
Blocking behavior: Cannot be marked complete until it can return provider delivery states without treating event delivery as the sole source of truth.
Integration validation: Provider `archon` delivery status fixtures drive Hermes health states for healthy, delayed, failed, duplicated, terminal failure, and waiting for reconciliation without mutating project work incorrectly.

**Acceptance Criteria:**

**Given** a Project Work Item has a provider workflow reference
**When** an authorized caller queries workflow status
**Then** Hermes returns workflow event delivery state as healthy, delayed, failed, duplicated, terminal failure, or waiting for reconciliation when that state is known
**And** Hermes links the state to the affected workflow run and Project Binding.

**Given** workflow event delivery is delayed or retrying
**When** Hermes receives or polls delivery status through CLI JSON
**Then** Hermes returns the retry state, last attempt time if available, next action if available, and whether user action is required.

**Given** workflow event delivery reaches terminal failure
**When** Hermes records the failure
**Then** Hermes returns an actionable diagnostic with delivery status, last error category, affected event type, workflow run reference, and recovery option
**And** Hermes does not block workflow execution solely because event notification failed.

**Given** Hermes detects duplicate workflow event delivery
**When** an authorized caller queries event delivery health
**Then** Hermes identifies the event as duplicate-safe
**And** Hermes returns evidence that no duplicate project-work mutation was applied.

**Given** Hermes has incomplete workflow event evidence but other systems may have progressed
**When** reconciliation is needed
**Then** Hermes marks the workflow or Project Work Item as waiting for reconciliation
**And** Hermes exposes a user action or automated reconciliation path without silently completing work.

## Epic 4: Human-Gated Story Execution

Kevin can run the combined story workflow from Hermes, review the done-verification evidence, approve or reject the gate, and route reruns or recovery.

### Gate Decision Ownership Convention

- Story 3.4b owns provider approve and reject command transport, strict adapter parsing, fail-closed command-result handling, and command-result persistence only.
- Story 3.4b does not own HILT Gate decision persistence or phase-task gate transitions.
- Story 4.1 owns starting the combined story workflow and recording the workflow run reference on the Phase Task. Story 4.1 does not own any gate transition.
- Story 4.2 owns the Phase Task transition from blocked `done_verification` to story completion or recovery.
- Story 4.2 does not define a general HILT Gate decision model beyond the minimum transition data needed for this flow.
- Story 4.3 owns shared HILT Gate decision record hardening, structured evidence results, audit fields, rejection reason capture, recovery-action selection, delayed-decision behavior, and separation between human decision records and provider command results.
- The persisted human decision record is authoritative for approval or rejection.
- Provider command success is transport evidence only and must not be treated as proof that gate evidence was sufficient.

### Story 4.1: Run Story Workflow

As a workflow operator,
I want Hermes to start the combined story workflow for a selected BMAD story,
So that story creation, implementation, and review run as one provider-controlled sequence.

**Requirements Covered:** FR-12, FR-14.

**Implementation Scope:** Combined story workflow start and workflow run reference recording on the Phase Task. This story owns no gate transition.

Depends on: Story 2.6, Story 3.4a, Story 3.4b, Story 3.6a, Story 3.6b, Story 3.6c, Story 3.8 (this file).
Contract needed: Combined story workflow control result, workflow completion event, and phase task identity.
Blocking behavior: Cannot be marked complete until provider workflow control can start the combined workflow and Hermes workflow event ingestion can record its run reference for the selected BMAD story.
Integration validation: A workflow-start fixture starts through the provider adapter and records the run reference on the Phase Task.

**Acceptance Criteria:**

**Given** a Project Work Item has a Phase Task
**When** the user starts the story workflow from Hermes
**Then** Hermes starts the configured combined story workflow for the selected BMAD story
**And** Hermes records the workflow run reference on the Phase Task.

**Given** the combined story workflow is running
**When** the user checks status from Hermes
**Then** Hermes returns structured workflow progress without requiring the provider dashboard.

---

### Story 4.2: Block Story Workflow For Done Verification

As a workflow operator,
I want Hermes to block the phase task for real done verification once the combined story workflow completes,
So that the story is only marked complete when workflow evidence, PR state, and human review support it.

**Requirements Covered:** FR-13, FR-14.

**Implementation Scope:** Done-verification gate transition after the combined workflow's fix loop and PR creation complete. Owns completing the Phase Task after approval or routing it to recovery after rejection. Must not create a separate general HILT Gate decision model beyond the minimum transition data required for this flow.

Depends on: Story 2.6, Story 3.4a, Story 3.4b, Story 3.4c, Story 3.6a, Story 3.6b, Story 3.6e, Story 3.8, Story 4.1 (this file).
Contract needed: Combined workflow completion event, fix-loop result, PR reference if available, done verification gate record, and gate decision command result.
Blocking behavior: Cannot be marked complete until done verification is approved in Hermes.
Integration validation: A workflow-completion fixture completes through provider workflow evidence, blocks on `done_verification`, rejects GitHub merge alone as completion proof, and routes rejection to rerun or recovery.

**Acceptance Criteria:**

**Given** the combined story workflow completes successfully
**When** Hermes receives completion evidence through workflow event, provider status, or reconciliation
**Then** Hermes blocks the Phase Task with gate kind `done_verification`
**And** Hermes does not complete the BMAD story or Project Work Item until the gate is approved.

**Given** the Phase Task is blocked for done verification
**When** an authorized caller queries the pending gate
**Then** Hermes returns implementation evidence, workflow result, PR reference if available, unresolved issues if available, and next available actions.

**Given** the Phase Task is blocked for done verification
**When** an authorized caller queries gate evidence
**Then** Hermes returns affected Project Work Item, Phase Task, BMAD story, provider workflow run, workflow event receipt when available, evidence references, and recovery action.

**Given** the user approves the done verification gate
**When** Hermes records the decision
**Then** Hermes completes the Phase Task
**And** Hermes records that completion came from human done verification, not only Archon completion or GitHub merge state.

**Given** the user rejects the done verification gate
**When** Hermes records the rejection
**Then** Hermes routes the Phase Task to rerun the fix loop or another selected recovery path
**And** Hermes does not mark the story complete.

---

### Story 4.3: Capture Human Gate Decisions And Evidence

As a workflow operator,
I want Hermes to return gate evidence, accept my approval or rejection, and persist the decision,
So that every HILT gate is auditable and can drive the matching provider control action when needed.

**Requirements Covered:** FR-14.

**Implementation Scope:** Shared HILT Gate decision record hardening, structured evidence results, audit fields, rejection reason capture, recovery-action selection, delayed-decision behavior, and provider command-result association. Must keep human decision records separate from provider command results.

Depends on: Story 3.4b, Story 3.4c, Story 4.1, Story 4.2 (this file).
Contract needed: Gate decision record, evidence reference shape, provider approve, reject, resume, and retry command result schemas.
Blocking behavior: Cannot be marked complete until Hermes can persist replay-safe human decisions and send required provider control commands without conflating command result with human decision.
Integration validation: Approval and rejection fixtures persist actor, timestamp, gate kind, decision, evidence references, reason if provided, selected recovery action, command result when required, and resulting phase state.

**Acceptance Criteria:**

**Given** a phase task is blocked on a HILT Gate
**When** Hermes publishes a notification or returns the pending gate
**Then** Hermes returns gate kind, affected Project Work Item, Phase Task, BMAD story reference, workflow run reference, evidence references, and available decisions
**And** Hermes phrases the next action in user-facing workflow language.

**Given** the user approves a HILT Gate
**When** Hermes records the decision
**Then** Hermes stores actor, timestamp, gate kind, decision, evidence references, reason if provided, and resulting phase state
**And** Hermes distinguishes approval from rejection through explicit decision values.

**Given** the user rejects a HILT Gate
**When** Hermes records the decision
**Then** Hermes stores actor, timestamp, gate kind, decision, rejection reason, evidence references, and selected recovery action
**And** Hermes routes the phase task to rerun, resume, retry, or recovery according to the selected failure reason.

**Given** a gate decision requires a provider approval, rejection, resume, or retry command
**When** Hermes records the human decision
**Then** Hermes sends the matching command through the strict provider adapter
**And** Hermes records the command result separately from the human decision record.

**Given** the user ignores or delays a pending gate
**When** Hermes evaluates gate state
**Then** Hermes keeps the phase task blocked
**And** Hermes returns the pending gate and recovery options without auto-continuing unless an explicit persisted policy permits it.

**Given** Hermes has a pending gate decision
**When** the gate is queried or mirrored through an existing notification transport
**Then** the result identifies Project Binding, BMAD story, Phase Task, gate kind, and required decision
**And** it excludes secrets, raw workflow event signatures, and unredacted command output.

## Epic 5: Story Status History, Reconciliation, And Diagnostics

Kevin can query one Story Status History, understand why a story changed state, and resolve drift across BMAD, Hermes, workflow providers, GitHub, workflow events, and gates.

### Story 5.1: Produce Unified Story Status History

As a workflow operator,
I want Hermes to return one Story Status History for each BMAD story,
So that I can understand current status, evidence, ownership, and next action without opening a provider dashboard.

**Requirements Covered:** FR-15.

**Implementation Scope:** Versioned structured Story Status History projection and optional human-readable command formatting.

Depends on: Story 2.6, Story 3.4a, Story 3.4b, Story 3.4c, Story 3.6a, Story 3.6b, Story 3.6e, Story 3.8, Story 4.1, Story 4.2, Story 4.3 (this file).
Contract needed: Project Work Item state, phase task state, provider run reference, workflow event receipt, GitHub PR reference, HILT Gate decision record, and next-action vocabulary.
Blocking behavior: Cannot be marked complete until it can return provenance from BMAD, Hermes, workflow providers, GitHub, workflow events, and human decisions without collapsing source-specific lifecycle state.
Integration validation: Status-history fixture returns BMAD artifact milestones, Project Work Item state, phase states, provider references, workflow events, GitHub PR references, gate decisions, redacted sensitive details, and user-facing next action.

**Acceptance Criteria:**

**Given** a Project Work Item is linked to a BMAD story
**When** an authorized caller queries its Story Status History
**Then** Hermes returns BMAD artifact milestones, Project Work Item state, Phase Task state, provider workflow run references, workflow events, GitHub PR references if available, HILT Gate decisions, and next action.

**Given** a Story Status History contains events from multiple systems
**When** the history is returned
**Then** each entry is source-labeled as BMAD, Hermes, workflow provider, GitHub, workflow event, reconciliation, or human decision.

**Given** BMAD artifact status and Hermes Kanban status differ
**When** Hermes produces the Story Status History
**Then** Hermes labels each status with its source system
**And** Hermes does not collapse BMAD artifact lifecycle into Hermes runtime Kanban lifecycle.

**Given** GitHub PR state and Done Verification Gate state differ
**When** Hermes produces completion status
**Then** Hermes returns PR state separately from done verification state
**And** Hermes does not treat a merged PR as human done approval.

**Given** a story has pending work
**When** Hermes produces the next action
**Then** Hermes returns who owns the next action: user, Hermes, workflow provider, BMAD, GitHub, or implementation agent
**And** Hermes phrases the next action in user-facing workflow language.

**Given** the Story Status History includes sensitive command, workflow event, or diagnostic data
**When** Hermes returns the history
**Then** Hermes redacts secrets
**And** Hermes preserves enough provenance for the user to understand why the story changed state.

---

### Story 5.2a: Reconcile BMAD Materialization Drift

As a workflow operator,
I want Hermes to reconcile BMAD artifact changes against Project Work Items and phase tasks,
So that changed planning artifacts update existing project work without duplicating work items or phase tasks.

**Requirements Covered:** FR-16.

**Implementation Scope:** Reconciliation for BMAD artifacts, `sprint-status.yaml`, Project Work Item materialization, and phase-task identity.

Depends on: Story 2.5, Story 2.6, Story 5.1 (this file).
Contract needed: BMAD source-state adapter, reconciliation result record, deterministic materialization repair rule, unresolved conflict marker, Project Work Item identity, and Phase Task identity.
Blocking behavior: Cannot be marked complete until missing sprint status, malformed sprint status, changed stories, unchanged stories, and duplicate-safe Project Work Item updates are represented as fixtures.
Integration validation: BMAD drift fixtures repair deterministic materialization gaps, preserve unresolved conflicts, avoid duplicate Project Work Items or phase tasks, and never auto-approve a HILT Gate.

**Acceptance Criteria:**

**Given** BMAD story artifact state changes after materialization
**When** reconciliation runs
**Then** Hermes detects whether the corresponding Project Work Item needs materialization update
**And** Hermes does not duplicate Project Work Items or phase tasks.

**Given** `sprint-status.yaml` is missing, malformed, unsupported, changed, or unchanged
**When** BMAD materialization reconciliation runs
**Then** Hermes records checked source, detected drift, deterministic repair action if available, and unresolved conflict if needed
**And** Hermes uses Project Work Item identity without phase kind.

**Given** reconciliation performs an automatic BMAD materialization repair
**When** an authorized caller queries Story Status History or diagnostics
**Then** Hermes returns what was repaired, why it was considered deterministic, and which BMAD source records supported the repair
**And** Hermes never auto-approves a HILT Gate.

---

### Story 5.2b: Detect Provider Workflow State Drift

As a workflow operator, I want Hermes to compare provider workflow state with Phase Task state so drift is detected before repair.

**Requirements Covered:** FR-16.

**Implementation Scope:** Provider workflow source-state adapter, status comparison, unreliable-status handling, and reconciliation-result persistence without repair.

Depends on: Story 2.6, Story 3.4a, Story 3.4c, Story 3.6a, Story 3.6b, Story 3.6c, Story 3.8, Story 4.1, Story 4.2, Story 4.3, Story 5.1, and Story 5.2a.
Blocking behavior: Preserve the last known Hermes state when provider evidence is unreliable.
Integration validation: Provider fixtures cover completed, failed, unavailable, malformed, timed-out, and unexpected state without repair.

**Acceptance Criteria:**

**Given** provider workflow state differs from Phase Task state
**When** reconciliation compares them
**Then** Hermes persists detected drift and source references without applying repair.

**Given** provider status is unreliable
**When** reconciliation runs
**Then** Hermes preserves the last known Phase Task state and records a diagnostic.

### Story 5.2c: Reconcile Workflow Event Delivery Evidence

As a workflow operator, I want Hermes to compare event receipts and delivery health with provider state so delivery drift is classified safely.

**Requirements Covered:** FR-16.

Depends on: Story 5.2b.
Blocking behavior: Event-evidence reconciliation never applies duplicate project-work mutation.
Integration validation: Fixtures classify delayed, duplicated, lost, terminally failed, and reconciliation-pending delivery.

**Acceptance Criteria:**

**Given** workflow event delivery is duplicated, delayed, failed, or absent after provider completion
**When** reconciliation evaluates event evidence
**Then** Hermes records the delivery-drift category without duplicate mutation.

### Story 5.2d: Apply Deterministic Provider Projection Repairs

As a workflow operator, I want Hermes to apply only deterministic repairs so recoverable drift is fixed without hiding conflicts or bypassing gates.

**Requirements Covered:** FR-16.

Depends on: Story 5.2c.
Blocking behavior: Repairs must not duplicate workflow references or gates and must never auto-approve a HILT Gate.
Integration validation: Repair fixtures explain every change, preserve unresolved conflicts, and remain idempotent.

**Acceptance Criteria:**

**Given** detected drift has one deterministic repair
**When** reconciliation applies it
**Then** Hermes records the repair evidence and resulting state without duplicating references or gates.

**Given** no deterministic repair exists
**When** reconciliation evaluates the conflict
**Then** Hermes preserves state and records an unresolved diagnostic.

---

### Story 5.2e: Reconcile GitHub And Done Verification Conflicts

As a workflow operator,
I want Hermes to reconcile GitHub PR state against Done Verification Gate state,
So that merged code never counts as completed story work without human done verification.

**Requirements Covered:** FR-16.

**Implementation Scope:** Reconciliation for GitHub PR state, Done Verification Gate state, completion diagnostics, and conflict projection.

Depends on: Story 4.2, Story 4.3, Story 5.1, Story 5.2a, Story 5.2d (this file).
Contract needed: GitHub PR source-state adapter, done verification gate state, reconciliation result record, unresolved completion conflict marker, and recovery option vocabulary.
Blocking behavior: Cannot be marked complete until merged PR, unresolved Done Verification Gate, rejected Done Verification Gate, and completion diagnostic fixtures exist.
Integration validation: GitHub and done-verification fixtures preserve completion conflicts, show recovery options, and never mark a story complete without done verification approval.

**Acceptance Criteria:**

**Given** GitHub PR state indicates merged but Done Verification Gate is unresolved or rejected
**When** reconciliation evaluates completion
**Then** Hermes records the conflict
**And** Hermes does not mark the story complete without done verification approval.

**Given** GitHub PR evidence and Done Verification Gate evidence disagree
**When** Hermes returns diagnostics or Story Status History
**Then** Hermes labels both source states separately
**And** Hermes identifies the next action owner and recovery path.

**Given** Done Verification Gate is approved and source evidence is consistent
**When** reconciliation evaluates completion
**Then** Hermes records the source records that support completion
**And** Hermes keeps GitHub merge state separate from human done approval.

---

### Story 5.3a: Define And Persist Operational Diagnostics

As a workflow operator, I want Hermes to persist stable, source-linked diagnostics so failures remain auditable and safe to query.

**Requirements Covered:** FR-17.

**Implementation Scope:** Diagnostic taxonomy, severity, affected references, redacted evidence storage, next-action owner, recovery-option reference, and persistence rules.

Depends on: Story 2.1c, Story 2.3, Story 2.5, Story 3.2, Stories 3.4a–3.4c, Story 3.6e, Story 3.8, Stories 4.1–4.3, Story 5.1, and Stories 5.2a–5.2e.
**Depends on Archon:** Stories 3.1 and 3.3a–3.3d.
Blocking behavior: Every required diagnostic family must have a stable, source-linked, secret-safe record.
Integration validation: Fixtures cover configuration, decisions, external delay, implementation defects, duplicate events, outbox backlog, stale PR, and unresolved gates.

**Acceptance Criteria:**

**Given** Hermes detects a supported orchestration problem
**When** diagnostics are generated
**Then** Hermes persists category, severity, affected references, redacted evidence, next-action owner, and recovery reference.

### Story 5.3b: Query Operational Diagnostics And Recovery Guidance

As a workflow operator, I want structured diagnostic and recovery results so I can act without a graphical interface.

**Requirements Covered:** FR-17.

**Implementation Scope:** Versioned diagnostic query results, human-readable command formatting, redaction, next-action language, and recovery references.

Depends on: Story 5.1 and Story 5.3a.
Blocking behavior: Results must be actionable without raw stack traces or secret disclosure.
Integration validation: Queries return category, severity, affected reference, redacted evidence, owner, recovery path, timestamp, and state.

**Acceptance Criteria:**

**Given** an authorized caller queries an actionable diagnostic
**When** Hermes returns it
**Then** the versioned result includes redacted evidence, next-action ownership, and recovery guidance without raw secrets.

### Story 5.3c: Resolve Diagnostics And Preserve Audit History

As a workflow operator, I want diagnostic resolution to preserve evidence and outcome so recovery remains auditable.

**Requirements Covered:** FR-17.

Depends on: Story 3.4c, Story 5.2e, and Story 5.3b.
Blocking behavior: Clearing a diagnostic records why and how it resolved without deleting prior evidence.
Integration validation: Resolution fixtures append source, timestamp, recovery action, and resulting state while preserving original evidence.

**Acceptance Criteria:**

**Given** a diagnostic is resolved
**When** reconciliation or authorized action clears it
**Then** Hermes appends resolution source, timestamp, recovery action, and resulting state while preserving prior history.

## Validation Command

```text
cd hermes-agent
uv sync --extra dev
uv run pytest
uv run ruff check .
```
