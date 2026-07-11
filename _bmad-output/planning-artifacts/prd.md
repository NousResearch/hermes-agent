---
title: hermes-agent Planning Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-11'
updated: '2026-07-11'
source: workflow-engine parent planning package materialized as local hermes-agent inputs
---

# PRD: hermes-agent Slice For Hermes Agent Workflow Commander

## Purpose

Hermes Agent Workflow Commander makes Hermes Agent the human-facing, headless command surface for BMAD planning, provider workflow execution, GitHub PR state, and local project work.
This file is the local `hermes-agent` PRD handoff for implementation agents.
It contains the Hermes-owned product requirements and consumer-side provider duties needed for implementation inside `hermes-agent`.
Implementation agents should not read parent planning files to complete downstream Hermes stories.

## Product Boundary

Workflow Commander v1 is headless.
It ships through commands, agent interactions, structured API or command results, durable records, and optional existing notification transports.
It does not ship a dedicated Workflow Commander dashboard, graphical Kanban board, gate screen, timeline screen, desktop view, web application, or marketing surface.

Hermes owns user and project orchestration.
BMAD owns planning artifacts and story workflow artifacts.
Archon is the first workflow provider and owns producer-side workflow execution, run state, provider binding, provider command JSON, event outbox, delivery status, and signed event production.
GitHub owns pull request state.
Hermes owns Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD invocation, materialization, Project Work Items, Phase Tasks, HILT Gates, provider adapter consumption, workflow event ingress, Story Status History, reconciliation, diagnostics, and headless validation guidance.

## Target User Journeys

- **UJ-1: Create a project-bound commander workspace.**
Kevin creates or selects a Project Binding, points it at the local repo cwd, connects GitHub context, mounts local BMAD skills, and registers provider metadata.
Hermes blocks ambiguous or unsafe automation when the cwd, profile, GitHub context, BMAD mount, or provider metadata conflicts.

- **UJ-2: Materialize BMAD planning into Hermes project work.**
Hermes reads `sprint-status.yaml` from the Bound Project Cwd, derives stable identities, creates or updates Project Work Items, and links a single Phase Task to each selected BMAD story.
Hermes rejects missing, unsupported, or malformed planning input before mutating project work.

- **UJ-3: Start and monitor provider-controlled story execution.**
Hermes starts provider workflow runs through a strict provider adapter and stores provider command evidence without requiring the provider dashboard.
Delayed, duplicated, malformed, or failed provider evidence routes through idempotency, diagnostics, and reconciliation.

- **UJ-4: Verify done through a human gate.**
Hermes blocks Phase Tasks for Done Verification after implementation evidence arrives.
Human approval completes the Phase Task, while rejection routes to rerun, resume, retry, or recovery.
GitHub merge state never substitutes for human Done Verification approval.

## Glossary

- **Archon** - The first workflow provider implementation under provider key `archon`, owning workflow run state, execution context, retry state, event binding records, event outbox, and signed event production.
- **Workflow Provider Binding** - A persisted workflow provider binding from a project, codebase, or provider execution context to an external controller identity and workflow event route.
- **BMAD** - The planning and story workflow system that owns brainstorming, PRDs, architecture, epics, story artifacts, sprint status, create-story workflows, and dev-story workflows.
- **Bound Project Cwd** - The explicit local repository path Hermes uses as the working directory for BMAD and workflow provider actions.
- **Controller Identity** - The generic provider controller identity composed of `provider` and `name`.
- **Done Verification Gate** - A human-in-the-loop gate after implementation that decides whether the story is actually done.
- **Hermes** - The user-facing command, agent-interaction, notification, project-work, decision, and reconciliation surface.
- **HILT Gate** - A human-in-the-loop review point where Hermes blocks progress until a human approves, rejects, or routes recovery.
- **Materialization** - The operation that reads BMAD planning artifacts from the Bound Project Cwd and creates or updates Hermes project-work tasks idempotently.
- **Phase Task** - The single Hermes phase task that runs the combined story workflow for a BMAD story and blocks for done verification.
- **Project Binding** - The Hermes-owned record that binds a project profile, local cwd, GitHub context, BMAD skill mount, and workflow provider integration metadata.
- **Project Work Item** - A Hermes operational work item layered on a Kanban task with BMAD, workflow provider, GitHub, PR, artifact, workflow, and gate references.
- **Reconciliation** - The process that compares BMAD artifact state, workflow provider state, GitHub PR state, and Hermes gate state to detect drift and repair projections.
- **Story Status History** - A source-labeled structured history of current status, evidence, references, gate state, provenance, and next action for a BMAD story.

## Hermes-Owned Functional Requirements

### FR-1: Create And View Project Bindings

Hermes can create, view, update, disable, and validate a Project Binding with profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, and display name.
Hermes rejects invalid cwd values and returns the active binding before starting any BMAD or provider action.
Hermes persists enough binding metadata to reconstruct status after restart.
Hermes reports provider binding status as missing, valid, stale, disabled, rotated, or conflicting when that evidence is available.

### FR-2: Mount Project-Local BMAD Skills

Hermes can add the bound project's BMAD skill directory to the selected profile's `skills.external_dirs` and reload that profile's skill index.
Hermes records the source directory for mounted BMAD skills.
Hermes does not use global skills as the primary BMAD mount for multi-project control.
Hermes detects missing and wrong-project BMAD mounts.

### FR-3: Enforce Bound Project Cwd For Workflow Actions

Hermes runs BMAD and provider actions for a Project Binding from the Bound Project Cwd unless a requirement explicitly states otherwise.
BMAD artifacts created through Hermes land under the bound project's configured output location.
Hermes blocks actions when the selected Project Binding lacks a valid cwd.
Hermes audit records include the cwd used for each workflow action.
Hermes does not infer cwd from skill visibility alone.

### FR-4: Invoke BMAD Planning Workflows From Hermes

Hermes can invoke selected BMAD planning workflows for brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation from the Bound Project Cwd.
Hermes presents BMAD as a behind-the-scenes workflow engine, records produced artifact paths, and preserves Project Binding context on failure.
Hermes can continue orchestration from generated BMAD artifacts in the bound project.

### FR-5: Materialize Sprint Status Into Project Work Items

Hermes can read `sprint-status.yaml` from the Bound Project Cwd and idempotently create or update Project Work Items for BMAD epics and stories.
Re-running materialization updates existing Project Work Items instead of duplicating them.
Hermes stores BMAD artifact references and observed planning status on each Project Work Item.
Hermes does not treat `sprint-status.yaml` as the runtime queue after materialization.

### FR-6: Maintain Hermes-Owned Operational Backlog

Hermes persists operational backlog, selected story, phase metadata, workflow references, human gate metadata, and next action as operational project-work state.
Hermes exposes that state through structured command, agent, or API results.
Hermes keeps canonical Kanban lifecycle values unchanged: `triage`, `todo`, `ready`, `running`, `blocked`, `done`, and `archived`.

### FR-7: Consume Generic Workflow Provider Bindings

Hermes can register or inspect provider-side workflow bindings through generic `provider` and `name` vocabulary.
Hermes detects disagreement between the Project Binding and provider binding metadata.
Hermes surfaces missing, stale, disabled, rotated, and conflicting provider binding states as actionable diagnostics.
Archon owns producer-side provider binding persistence and status production.

### FR-8: Control Provider Workflows Through Provider Adapters

Hermes can start, inspect, approve, reject, resume, retry, and cancel provider workflow runs through the adapter selected by the Project Binding.
For provider `archon`, Hermes consumes parseable CLI JSON through the provider adapter.
Hermes captures stdout, stderr, exit code, cwd when applicable, timeout, correlation id, and parsed result.
Hermes fails closed on malformed JSON, incompatible schema version, timeout, unexpected exit code, or unexpected state.
Hermes does not use provider HTTP APIs for the `archon` state-changing control path.

### FR-9: Receive Typed Workflow Provider Events

Hermes receives signed workflow provider events on `/p/{profile}/webhooks/workflow-events/{provider}` and mutates project work only after schema, binding, replay, idempotency, provider, profile, and authorization validation pass.
Hermes rejects unknown Project Binding, wrong profile, wrong codebase, stale timestamp, duplicate event id, invalid signature, unsupported provider, and schema failure before mutation.
Hermes stores accepted event ids and idempotency keys so duplicate delivery is safe.
Hermes maps completion, failure, approval-request, and artifact events only to the intended Project Work Item or Phase Task.

### FR-10: Return Provider Event Delivery And Outbox Health

Hermes can return structured provider event-delivery status identifying healthy, delayed, failed, duplicated, terminal failure, or reconciliation-pending state.
Hermes exposes this evidence through Story Status History and diagnostics.
Hermes does not block provider workflow execution solely because event notification failed.
Archon owns producer-side outbox and delivery-health status for provider `archon`.

### FR-11: Create One Phase Task Per BMAD Story

Hermes materializes each selected BMAD story as a single Phase Task.
The Phase Task links to one Project Work Item and shares Story Status History evidence.
Repeated materialization must not duplicate Phase Tasks.

### FR-12: Run The Combined Story Workflow

Hermes can run the configured combined story workflow for a selected BMAD story.
Hermes records the provider workflow run reference on the Phase Task.
The workflow can run story creation through review without a Hermes-side pause between phases.

### FR-13: Gate Done Verification

Hermes blocks the Phase Task for a Done Verification Gate after provider completion evidence arrives.
Human approval completes the Phase Task.
Human rejection routes to rerun, resume, retry, or recovery without marking the story complete.

### FR-14: Collect Human Decisions From Hermes

Hermes can publish or return structured gate evidence, accept approval or rejection through an authorized command or agent interaction, persist the decision, and send the matching provider command when required.
Each gate decision records actor, timestamp, gate kind, decision, reason when present, and evidence references.
Hermes does not auto-continue past a HILT Gate unless an explicit persisted policy later permits it.

### FR-15: Return Unified Story Status History

Hermes can return one source-labeled Story Status History containing BMAD milestones, Project Work Item state, Phase Task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, provenance, and next action.
The history distinguishes BMAD planning lifecycle from Hermes runtime lifecycle.
The history distinguishes GitHub PR merge state from Done Verification state.

### FR-16: Reconcile Cross-System State

Hermes compares BMAD artifact state, provider workflow state, GitHub PR state, Hermes Project Work Item state, and HILT Gate state to detect drift.
Hermes may repair deterministic projection drift.
Hermes must not auto-approve HILT Gates or mark stories complete when evidence conflicts.

### FR-17: Provide Operational Diagnostics

Hermes surfaces binding conflicts, cwd problems, missing BMAD artifacts, unsupported sprint status, provider command contract gaps, event delivery failures, duplicate workflow events, outbox backlog, stale PR references, and unresolved gates.
Diagnostics distinguish user action, configuration action, Hermes automation, provider action, BMAD action, GitHub action, implementation-agent action, and external delay.
Diagnostics include recovery guidance and redact secrets.

## Non-Functional Requirements

- Workflow events are delivery acceleration, not the sole source of truth.
- Reconciliation handles event loss, duplicate delivery, gateway downtime, provider command failure, and manual PR merge.
- Materialization is idempotent.
- Gate decisions are replay-safe and auditable.
- Event ingress fails closed on signature, schema, replay, binding, provider, profile, idempotency, or authorization failure.
- Workflow event secrets are scoped to the correct profile.
- Workflow actions cannot run outside the selected Bound Project Cwd.
- Secrets are redacted from command logs, event logs, diagnostics, and status-history results.
- Workflow commands, workflow events, reconciliation actions, gate decisions, and user-visible state transitions are persisted.
- Story Status History explains why a story changed state.
- Project-work changes retain source provenance.
- Next actions use human-facing workflow language.
- Done Verification approval remains separate from GitHub PR merge state.
- Blocking issues return recovery options instead of raw stack traces alone.
- Provider integration surfaces remain generic.
- Isolated local handoffs are complete enough for subproject implementation agents.

## Non-Goals

- Workflow Commander will not replace BMAD, Archon, GitHub, or Hermes Kanban with one monolithic workflow database.
- Workflow Commander will not require the user to operate the provider dashboard for normal workflow control.
- Workflow Commander will not use provider HTTP APIs for the `archon` state-changing control path.
- Workflow Commander will not add Hermes-specific provider command vocabulary.
- Workflow Commander will not treat `sprint-status.yaml`, GitHub Issues, or provider UI state as Hermes runtime queue truth.
- Workflow Commander will not rely on global skills as the primary BMAD mount mechanism for multi-project control.
- Workflow Commander will not auto-approve HILT Gates without explicit persisted policy and evidence requirements.
- Workflow Commander will not write implementation artifacts from this parent planning story.
- Workflow Commander will not ship a dedicated graphical frontend in v1.

## Contract Readiness Rule

The local contract package contains schema files under `contracts/workflow-commander/schemas/`.
Required example fixture families remain required before contract-gated downstream stories can complete.
No downstream story may claim example fixture readiness unless the matching files exist under `contracts/workflow-commander/examples/` and pass the story's compatibility tests.

## Implementation Root And Candidate Validation Commands

Run downstream implementation workflows from the `hermes-agent` subproject root:

```text
hermes-agent/
```

Candidate validation commands:

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```
