---
title: hermes-agent Planning Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-06-27'
updated: '2026-06-29'
---

# PRD: hermes-agent Slice For Hermes Agent Workflow Commander

## Purpose

This file is the `hermes-agent` local PRD handoff for Hermes Agent Workflow Commander.
It contains only Hermes-owned product requirements and explicit dependency notes for Archon-owned producer surfaces.
Implementation agents should run BMad workflows from inside `hermes-agent` and should not read parent workspace planning files.

## Product Outcome

Hermes becomes the human-facing command center for project-bound BMAD planning, workflow provider control, human gates, Story Timeline, reconciliation, and diagnostics.
Hermes owns the Project Binding, BMAD mount, materialization, project-work records, phase tasks, HILT Gates, Story Timeline, reconciliation projection, workflow event ingress, and workflow provider adapters.
Archon is the first provider implementation under provider key `archon`.

## hermes-agent-Owned Functional Requirements

### FR-1: Project Binding

Hermes can create, view, update, disable, and validate a Project Binding with profile identity, Bound Project Cwd, GitHub context, BMAD skill directory reference, workflow provider binding metadata, and display name.
Hermes must fail closed on invalid cwd, binding conflicts, disabled bindings, and ambiguous automation.

### FR-2: Project-Local BMAD Mount

Hermes can add the bound project's BMAD skill directory to the selected profile's `skills.external_dirs`.
Hermes reloads the profile skill index and distinguishes project-local BMAD skills from global skills.

### FR-3: Bound Project Cwd Enforcement

Hermes runs BMAD and workflow provider actions for a Project Binding from the Bound Project Cwd.
Hermes records the cwd used for workflow actions and blocks actions without a valid cwd.

### FR-4: BMAD Planning Invocation

Hermes invokes supported BMAD planning workflows from the Bound Project Cwd.
Hermes records produced artifact paths and preserves Project Binding context on failure.
This is a generic workflow discovery, invocation, cwd enforcement, result capture, artifact recording, and diagnostic path.
Hermes does not add bespoke workflow-specific branching unless the workflow declares a distinct contract.

### FR-5: Materialization

Hermes reads `sprint-status.yaml` from the Bound Project Cwd and idempotently creates or updates Project Work Items.
Project Work Item identity is derived from bound project cwd, BMAD artifact path, and BMAD epic or story identity.
Project Work Item identity must not include phase kind.
Hermes must reject missing, malformed, or unsupported sprint status data before mutating project work.

### FR-6: Operational Backlog

Hermes stores operational backlog, selected story, phase metadata, workflow references, gate metadata, artifact refs, observed BMAD status, target BMAD status, and next action.
Hermes keeps canonical Kanban statuses unchanged and uses facade lanes for workflow language.
Phase Task identity is derived from Project Work Item identity plus phase kind.

### FR-8: Workflow Provider Control Adapter

Hermes starts, checks status, approves, rejects, resumes, retries, and cancels workflow provider runs through the provider adapter selected by the Project Binding.
Provider `archon` maps this adapter to Archon CLI JSON in v1.
Hermes captures stdout, stderr, exit code, timeout, cwd, correlation id, workflow name, workflow run reference, and parsed result.

### FR-9: Workflow Event Ingress

Hermes receives workflow provider events on `/p/{profile}/webhooks/workflow-events/{provider}`.
Hermes mutates project work only after schema, signature, replay, idempotency, profile route, profile-scoped secret, provider, codebase, binding, and authorization checks pass.
Hermes rejects workflow events signed with a valid secret for the wrong profile before mutation.

### FR-10: Workflow Event Delivery Health Display

Hermes displays workflow event delivery health from provider status fixtures and local workflow event receipts.
Hermes must not treat workflow event delivery as the only source of truth.

### FR-11: Phase Tasks

Hermes materializes each selected BMAD story into a Prepare Story Phase Task and an Implement Story Phase Task.
Implement Story remains blocked until Prepare Story completes.

### FR-12: Prepare Story Gate

Hermes runs the Prepare Story workflow and blocks for a Test-Case Adequacy Gate after completion evidence arrives.
Approval releases Implement Story.
Rejection routes rerun or recovery.

### FR-13: Implement Story Gate

Hermes runs the Implement Story workflow and fix loop only after Prepare Story is approved.
Hermes blocks for a Done Verification Gate after completion evidence arrives.

### FR-14: Human Gate Decisions

Hermes presents gate evidence, captures approval or rejection, persists the decision, and sends required provider decision commands through the strict provider adapter.
Human decision records must be replay-safe and auditable.
For v1, Hermes dashboard gate prompts and blocked phase-task state are the required notification surface.
Existing Hermes notification channels may mirror gate prompts when available, but out-of-dashboard notification delivery is optional for v1.
If notification delivery fails, the Story Timeline and phase-task blocked state remain canonical.

### FR-15: Story Timeline

Hermes renders one Story Timeline across BMAD artifact milestones, Project Work Item state, phase task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, and next action.

### FR-16: Reconciliation

Hermes compares BMAD artifact state, provider workflow state, GitHub PR state, Hermes Project Work Item state, workflow event receipts, and HILT Gate state to detect drift.
Deterministic projection gaps may be repaired.
Conflicting evidence remains visible.
Hermes never treats GitHub merge state as done verification approval.

### FR-17: Diagnostics

Hermes surfaces binding conflicts, cwd problems, missing BMAD artifacts, unsupported sprint status, provider command contract gaps, workflow event failures, duplicate workflow events, outbox backlog, stale PR references, unresolved gates, and recovery paths.

## Cross-Project Dependencies

Hermes implementation depends on workflow provider contracts for provider binding, workflow command JSON, workflow event production, and workflow event delivery health.
Every story that depends on a provider implementation must include `Depends on`, `Contract needed`, `Blocking behavior`, and `Integration validation`.
Hermes consumer tests must parse the same shared examples that provider producer tests emit.
Shared contract fixtures are planned under `_bmad-output/planning-artifacts/contracts/workflow-commander/` and must be regenerated into this local handoff before Hermes consumer code depends on them.

## UX Requirements

Workflow Commander reuses the Hermes dashboard shell, existing Kanban dashboard plugin, task drawer, task events, comments, runs, attachments, and parent-child task behavior.
Project Binding selection shows display name, profile identity, Bound Project Cwd, GitHub reference, BMAD mount status, workflow provider binding status, enabled state, and validation state.
Review Test Cases and Verify Done are facade lanes over canonical `blocked` plus gate metadata.
Gate evidence presentation shows gate kind, affected Project Work Item, affected phase task, BMAD story reference, provider workflow run reference, evidence references, available decisions, rejection reason capture, and recovery action.
Story Timeline entries are source-labeled across BMAD, Hermes, workflow provider, GitHub, workflow events, reconciliation, and human decisions.
Diagnostics show category, severity, affected reference, redacted evidence, next action owner, recovery path, resolution source, timestamp, and resulting state.
UI implementation follows `hermes-agent/web/README.md` typography, contrast, token, font-tier, responsive layout, and dashboard density rules.

## Non-Goals

- Hermes will not own provider workflow run state, event outbox delivery, or provider binding persistence.
- Hermes will not call provider HTTP APIs for state-changing control in v1.
- Hermes will not treat `sprint-status.yaml` as the runtime queue after materialization.
- Hermes will not auto-approve HILT Gates.
- Hermes will not add a new shared database or cloud queue for v1.

## Validation

Run implementation validation from inside `hermes-agent`.

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Validation must prove idempotent materialization, duplicate-safe workflow events, cwd enforcement, replay-safe gate decisions, redaction, and reconciliation behavior.
