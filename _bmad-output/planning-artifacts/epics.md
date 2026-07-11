---
title: hermes-agent Epics Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-11'
updated: '2026-07-12'
local_contract_package: contracts/workflow-commander/
story_ownership_note: >
  Story numbering is kept identical to the parent workspace story ids so cross-project dependency records remain stable.
  This file contains only hermes-agent-owned implementation stories.
---

# hermes-agent Epics: Hermes Agent Workflow Commander

## Overview

This file is the local Hermes-owned downstream story catalog for Workflow Commander.
It contains exactly thirty implementation story headings.
It excludes Archon producer stories, parent contract stories, parent handoff stories, and parent validation stories.
Hermes implementation agents should use this file from inside the `hermes-agent` subproject without reading parent planning files.

The implementation root is the `hermes-agent` subproject root:

```text
hermes-agent/
```

Candidate downstream validation commands:

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Ownership Boundary

Hermes owns Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD invocation, workflow materialization, Project Work Items, Phase Tasks, HILT Gates, workflow event ingress, Story Status History, reconciliation, diagnostics, provider adapter registration, provider command-result consumption, and Hermes validation guidance.
Archon owns Workflow Provider Binding storage, Archon CLI producer implementation, provider workflow run state, retry and cancel internals, provider event outbox, event delivery status production, and signed workflow event production.
Parent contract stories own shared schemas and example fixture families before downstream consumer or producer stories can complete.
Workflow Commander v1 remains headless and does not create a dedicated dashboard, desktop surface, web app, or marketing-style UX surface.

## Contract Readiness

The local contract package path is `contracts/workflow-commander/`.
Schema files are present under `contracts/workflow-commander/schemas/`.
Required example fixture families remain required before contract-gated stories can complete.
No story may claim example fixture readiness unless the matching files exist under `contracts/workflow-commander/examples/` and pass compatibility validation.

## Provider Completion Gate

The local contract package is Hermes-side readiness evidence, not proof that external Archon producer output is compatible.
Any Hermes story that depends on Archon producer output must keep its dependency record and must not be marked `done` from local fixtures alone.
Completion for those provider-dependent stories requires compatible external Archon producer output for every consumed command, provider binding, delivery-status, and workflow-event family, validated against the local `contracts/workflow-commander/` package.
Until that external evidence is available, Hermes implementation may proceed against local fixtures where the story allows it, but final done claims remain blocked.

Current external evidence status: no captured external Archon producer runtime output is present in this isolated handoff. Local contract validation is sufficient for Hermes-side implementation where a story allows fixture-driven work, but provider-dependent completion remains blocked until captured Archon producer output for provider binding lifecycle, workflow command, workflow event, and delivery/outbox status families is supplied and validated against the local package.

## Local NFR Coverage Map

This map keeps the parent NFR traceability local to the Hermes handoff.
The NFR ids match the numbered PRD NFRs.
It lists Hermes-owned story coverage and the validation evidence downstream implementation agents must preserve.
Parent shared contract stories and Archon producer stories remain dependencies, not Hermes implementation work.

| NFR | Hermes Coverage | Required Local Validation Evidence |
| --- | --- | --- |
| NFR-1 | Stories 3.8, 5.2b, 5.2c, and 5.2d | Workflow event and delivery-status examples prove events accelerate delivery but reconciliation remains authoritative. |
| NFR-2 | Stories 5.2a, 5.2b, 5.2c, 5.2d, 5.2e, 5.3a, 5.3b, and 5.3c | Drift examples cover workflow event loss, duplicate delivery, gateway downtime, provider command failure, and manual PR merge. |
| NFR-3 | Stories 2.5, 2.6, and 5.2a | Materialization examples prove idempotent Project Work Item and Phase Task identity. |
| NFR-4 | Stories 3.4b, 4.2, and 4.3 | Gate examples prove replay-safe decision records, separate command results, and audit fields. |
| NFR-5 | Stories 3.6a and 3.6b | Workflow event rejection examples cover signature, schema, replay, binding, provider, profile, idempotency, and authorization failures. |
| NFR-6 | Story 3.6a | Wrong-profile-secret examples prove profile-scoped workflow event secret enforcement. |
| NFR-7 | Stories 2.1b, 2.1c, 2.3, and 3.4a | Cwd guard tests prove actions cannot run outside the selected Project Binding cwd. |
| NFR-8 | Stories 4.3, 5.1, 5.3a, and 5.3b | Status-history, gate, command, workflow event, and diagnostic examples prove secret redaction. |
| NFR-9 | Stories 2.3, 3.4a, 3.4b, 3.4c, 3.6b, 4.3, 5.2a, 5.2b, 5.2c, 5.2d, 5.2e, 5.3a, and 5.3c | Persistence and audit examples prove workflow commands, workflow events, reconciliation actions, gate decisions, and state transitions are recorded. |
| NFR-10 | Story 5.1 | Story Status History examples explain every returned state change and next action. |
| NFR-11 | Stories 5.1, 5.2a, 5.2b, 5.2c, 5.2d, 5.2e, 5.3a, 5.3b, and 5.3c | Provenance examples distinguish BMAD, workflow provider, GitHub, Hermes, workflow event, reconciliation, implementation agent, and human decision sources. |
| NFR-12 | Stories 4.3, 5.1, and 5.3b | Command and diagnostic examples phrase next actions in user-facing workflow language. |
| NFR-13 | Stories 2.6, 4.2, and 5.1 | Structured project-work and gate examples distinguish Done Verification Gate approval from GitHub PR merge state. |
| NFR-14 | Stories 2.1b, 2.1c, 2.2, 2.4, 3.2, 3.4a, 3.4b, 3.4c, 3.8, 5.3a, 5.3b, and 5.3c | Diagnostic examples return recovery options rather than raw stack traces alone. |
| NFR-15 | All Hermes downstream stories with dependency records | Dependency records and handoff validation preserve bounded ownership between Hermes, BMAD, workflow providers, and GitHub. |
| NFR-16 | Stories 3.2, 3.4a, 3.4b, and 3.4c | Provider adapter examples use generic provider vocabulary and avoid Hermes-specific provider surfaces. |
| NFR-17 | This local handoff package and all downstream Hermes stories | Local `prd.md`, `architecture.md`, and `epics.md` remain complete enough for isolated Hermes implementation agents. |

## Dependency Record Format

Use this dependency shape in downstream story files when a story depends on Archon producer output or parent shared contract work:

```text
Depends on: <subproject or parent> Story <id or title>
Contract needed: <API/event/file/interface/schema>
Blocking behavior: <what must exist before this story can be completed>
Integration validation: <how both sides will be proven compatible>
```

## Epic 2: Project-Bound Planning And Work Backlog

Kevin can bind a project, mount its BMAD skills, run planning from the correct cwd, and see BMAD stories materialized into Hermes project work with one Phase Task per story.
This epic owns the Hermes project-work substrate.
It does not implement provider producer contracts.

### Story 2.1a: Create And Persist Project Bindings

As a workflow operator,
I want Hermes to create and retrieve explicit Project Bindings,
so that every later action has a stable persisted project identity.

Requirements Covered: FR-1.
Implementation Scope: Hermes Project Binding schema, migration, stable identity, create and read operations, restart persistence, and persistence-level uniqueness.
Depends on: parent Story 1.3a.
Contract needed: Workflow Provider Binding schema, provider metadata shape, Project Binding identity, and uniqueness constraints for profile, cwd, GitHub context, BMAD mount, and provider metadata.
Blocking behavior: This story cannot be completed until migration and uniqueness tests prove ambiguous bindings cannot be persisted.
Integration validation: Hermes migration and uniqueness tests validate stored Project Binding fields against the shared Workflow Provider Binding schema and required binding examples once those examples exist locally.

Acceptance Criteria:

- Given no Project Binding exists for a selected profile, when an authorized command creates one with profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, and display name, then Hermes persists it and returns it after restart with the same stable identity.
- Given a Project Binding would violate uniqueness, when creation runs, then Hermes rejects the record without partial write and returns a machine-readable conflict result.

### Story 2.1b: Validate Project Binding Safety And Conflicts

As a workflow operator,
I want Hermes to validate Project Binding safety and cross-binding conflicts,
so that automation fails closed for unsafe or ambiguous project context.

Requirements Covered: FR-1.
Implementation Scope: Hermes allowed-root cwd validation, GitHub and BMAD reference validation, provider-metadata validation, conflict detection, validation-state transitions, and actionable diagnostics.
Depends on: hermes-agent Story 2.1a and parent Story 1.3a.
Contract needed: Project Binding validation result, allowed-root policy, conflict categories, Workflow Provider Binding metadata, and diagnostic vocabulary.
Blocking behavior: A binding cannot become enabled until all required validation checks pass.
Integration validation: Invalid cwd, missing-reference, and conflicting-metadata tests fail closed without enabling automation.

Acceptance Criteria:

- Given a Bound Project Cwd does not exist or is outside allowed workspace roots, when validation runs, then Hermes rejects activation and returns a structured actionable diagnostic without starting workflow actions.
- Given another Project Binding conflicts on profile, cwd, GitHub, BMAD mount, or workflow provider metadata, when validation runs, then Hermes blocks ambiguous automation and returns affected binding references plus conflict category.

### Story 2.1c: Manage Project Binding Lifecycle And Status

As a workflow operator,
I want Hermes to update, disable, repair, re-enable, and query Project Bindings,
so that binding lifecycle changes remain safe, durable, and auditable.

Requirements Covered: FR-1.
Implementation Scope: Hermes binding update, disable, repair, re-enable, audit preservation, and structured status serialization.
Depends on: hermes-agent Story 2.1b and parent Story 1.3a.
Contract needed: Project Binding lifecycle command result, versioned status result, disabled-action rule, provider binding status shape, and validation-state vocabulary.
Blocking behavior: Disabled or invalid bindings must block BMAD and provider workflow actions.
Integration validation: Lifecycle tests survive restart, preserve audit history, and return display name, profile identity, Bound Project Cwd, GitHub reference, BMAD mount status, provider binding status, enabled state, and validation state.

Acceptance Criteria:

- Given an existing Project Binding has valid updated metadata, when an authorized command updates it, then Hermes preserves binding id, audit history, and validation-state transition.
- Given a Project Binding is disabled, when BMAD or provider action is requested through that binding, then Hermes blocks the action and explains that the binding must be enabled and valid.
- Given at least one Project Binding exists, when status is queried, then Hermes returns all required binding and validation fields in a versioned structured result.

### Story 2.2: Mount Project-Local BMAD Skills For A Binding

As a workflow operator,
I want Hermes to mount project-local BMAD skills for a selected Project Binding,
so that Hermes can discover and run the correct BMAD workflows for that project without relying on global skills.

Requirements Covered: FR-2.
Implementation Scope: Hermes BMAD skill mount configuration, profile skill-index reload, mount diagnostics, and disabled-binding safety.
Depends on: hermes-agent Story 2.1c.
Contract needed: Project Binding BMAD skill directory field, profile `skills.external_dirs` update rule, mount validation state, and wrong-project mount diagnostic.
Blocking behavior: BMAD skill mounting cannot be completed until Hermes associates a mounted skill directory with a valid enabled Project Binding.
Integration validation: Hermes reloads the selected profile skill index, distinguishes project-local BMAD skills from global skills, and blocks workflow execution when the mount is missing or points at the wrong project.

Acceptance Criteria:

- Given a valid enabled Project Binding with a BMAD skill directory reference, when the user mounts BMAD skills for that binding, then Hermes adds the project BMAD skill directory to the selected profile's `skills.external_dirs` and reloads the profile skill index so the mounted skills appear in skill discovery, skill view, and slash command discovery.
- Given the selected profile already contains the Project Binding BMAD skill directory in `skills.external_dirs`, when the user mounts BMAD skills again, then Hermes leaves a single normalized entry for that directory and does not duplicate `skills.external_dirs` entries.
- Given profile skill-index reload fails after `skills.external_dirs` is changed, when Hermes reports the mount result, then Hermes records the reload failure as a mount diagnostic and does not mark the mount valid until skill discovery confirms the project-local BMAD skills are visible.
- Given a mounted BMAD skill is visible in Hermes, when the user inspects the skill source, then Hermes shows the source skill directory and associated Project Binding and distinguishes project-local BMAD skills from global `~/.hermes/skills`.
- Given the configured BMAD skill directory is missing, when Hermes validates or reloads the Project Binding, then Hermes marks the BMAD mount as invalid and prevents BMAD workflow execution for that binding until the mount is repaired.
- Given the configured BMAD skill directory points to a different project than the Bound Project Cwd, when Hermes validates the mount, then Hermes reports a wrong-project mount diagnostic and does not use skill visibility alone to infer artifact placement.
- Given the user disables a Project Binding, when Hermes refreshes profile skill state, then Hermes prevents disabled binding skills from being used as active project workflow actions and preserves enough mount metadata to re-enable the binding safely later.

### Story 2.3: Enforce Bound Project Cwd For Workflow Actions

As a workflow operator,
I want Hermes to run BMAD and provider actions from the Project Binding's explicit cwd,
so that planning artifacts and workflow execution always belong to the intended local project.

Requirements Covered: FR-3.
Implementation Scope: Hermes workflow-action cwd guard, audit capture, BMAD invocation cwd, and a minimal provider-action cwd port/test double used only to prove cwd propagation through the generic provider-action boundary.
Depends on: hermes-agent Story 2.1c.
Contract needed: Bound Project Cwd authority rule, BMAD execution cwd, workflow action audit record, and minimal provider-action cwd propagation interface for tests.
Blocking behavior: Cwd enforcement can be completed when Hermes proves BMAD actions and the minimal provider-action test double receive the selected Project Binding cwd. Provider-specific adapter evidence is out of scope for Story 2.3.
Integration validation: Cwd guard tests prove BMAD actions and test-double provider actions use the Bound Project Cwd and reject missing or invalid cwd before any external workflow action runs.

Acceptance Criteria:

- Given an enabled Project Binding with a valid Bound Project Cwd, when the user starts a BMAD planning workflow from Hermes, then Hermes runs the workflow with the Bound Project Cwd as the working directory and any BMAD artifacts produced through Hermes land under that project's configured `_bmad-output` location.
- Given an enabled Project Binding with a valid Bound Project Cwd, when Hermes exercises the minimal provider-action cwd port, then Hermes passes the Bound Project Cwd to that port and records the cwd used for the action without requiring provider-specific adapter evidence in this story.
- Given the selected project has no Bound Project Cwd, when the user attempts to start any BMAD or workflow provider action, then Hermes blocks the action and shows a diagnostic that identifies the missing cwd requirement.
- Given a mounted BMAD skill is visible from a profile, when Hermes determines where a workflow action should run, then Hermes uses the active Project Binding cwd, not skill visibility, as the artifact placement and execution authority.
- Given Hermes executes any project-bound workflow action, when the action completes, fails, or is cancelled, then Hermes persists an audit record containing the Project Binding id, profile identity, cwd, command or workflow name, started time, completed time if available, result state, and correlation id if available.

### Story 2.4: Invoke BMAD Planning Workflows From Hermes

As a workflow operator,
I want Hermes to invoke BMAD planning workflows for a bound project,
so that I can create or update planning artifacts from Hermes and continue orchestration from those artifacts.

Requirements Covered: FR-4.
Implementation Scope: Hermes generic BMAD workflow discovery, invocation, Bound Project Cwd execution, artifact path recording, and failure diagnostics.
Depends on: hermes-agent Story 2.1c, hermes-agent Story 2.2, and hermes-agent Story 2.3.
Contract needed: Supported BMAD workflow list, mounted skill discovery result, produced artifact path record, and Bound Project Cwd execution rule.
Blocking behavior: BMAD workflow invocation cannot be completed until Hermes proves the selected workflow exists in mounted project-local skills and runs from the bound cwd.
Integration validation: Hermes invokes supported BMAD workflows from the Bound Project Cwd, records produced artifact paths, and blocks unsupported, unavailable, or unmounted workflows.

Acceptance Criteria:

- Given an enabled Project Binding with a valid Bound Project Cwd and valid BMAD mount, when the user selects a supported BMAD planning workflow from Hermes, then Hermes invokes that workflow from the Bound Project Cwd and records the workflow name, cwd, Project Binding id, profile identity, started time, and result state.
- Given Hermes exposes supported BMAD planning workflows for a Project Binding, when the supported-workflow list is built, then it includes brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, and dev-story preparation when those workflows are present in the mounted project-local BMAD skill directory.
- Given Hermes exposes supported BMAD planning workflows for a Project Binding, when a workflow is missing from the supported-workflow list, then Hermes reports each missing workflow by workflow name, mount source, and Project Binding.
- Given two supported BMAD planning workflows use the same invocation contract, when Hermes invokes either workflow from a Project Binding, then Hermes uses the same generic invocation adapter, cwd enforcement, result capture, artifact recording, and diagnostic path.
- Given two supported BMAD planning workflows use the same invocation contract, when the workflow does not declare a distinct contract, then Hermes does not add workflow-specific branching.
- Given a BMAD workflow produces planning artifacts, when the workflow completes successfully, then Hermes records the produced artifact paths on the Project Binding or related project-work context and can show those artifact paths to the user.
- Given a BMAD workflow fails, when Hermes receives the failure result or process output, then Hermes preserves the Project Binding context and surfaces the failure output as a user-actionable diagnostic without losing the cwd used.
- Given BMAD planning artifacts already exist for a bound project, when the user asks Hermes to continue orchestration from generated artifacts, then Hermes can locate the artifacts from the bound project `_bmad-output` location and does not search unrelated project roots.
- Given the selected BMAD workflow is unsupported, unavailable, or missing from the mounted skills, when the user attempts to run it through Hermes, then Hermes blocks the invocation and shows whether the issue is missing skill mount, unsupported workflow, or invalid Project Binding state.

### Story 2.5: Materialize Sprint Status Into Project Work Items

As a workflow operator,
I want Hermes to materialize BMAD `sprint-status.yaml` into Hermes Project Work Items,
so that BMAD planning output becomes an operational backlog without making `sprint-status.yaml` the runtime queue.

Requirements Covered: FR-5.
Implementation Scope: Hermes `sprint-status.yaml` reader, Project Work Item identity, and idempotent Project Work Item upsert behavior.
Depends on: parent Story 1.3c, hermes-agent Story 2.1c, and hermes-agent Story 2.4.
Contract needed: Project Work Item identity schema, supported sprint-status examples, persistence shape, and idempotent upsert rule.
Blocking behavior: Materialization cannot be completed until unchanged, changed, malformed, missing, and duplicate-prevention examples exist locally and pass against Hermes implementation.
Integration validation: Re-running materialization updates existing Project Work Items rather than creating duplicates and records provenance for each source artifact.

Acceptance Criteria:

- Given an enabled Project Binding with a valid Bound Project Cwd, when materialization runs, then Hermes reads `sprint-status.yaml` only from that cwd and rejects missing, malformed, or unsupported data before mutation.
- Given a BMAD story changes in `sprint-status.yaml`, when materialization runs again, then Hermes derives the same Project Work Item identity and updates the existing item without including phase kind in the identity.
- Given materialization partially fails after validation begins, when Hermes reports the failure, then provenance identifies affected binding, source file, epic, story, or Project Work Item.

### Story 2.6: Create Phase Task And Operational Work State

As a workflow operator,
I want Hermes to represent each BMAD story as one Phase Task,
so that I can choose and track story work from Hermes without confusing BMAD story status with Hermes runtime state.

Requirements Covered: FR-6 and FR-11.
Implementation Scope: Hermes Phase Task persistence, operational project-work metadata, canonical Kanban status mapping, and structured state results.
Depends on: parent Story 1.3c, hermes-agent Story 2.1c, and hermes-agent Story 2.5.
Contract needed: Project Work Item identity, Phase Task identity, phase task link, reserved gate metadata, and canonical Kanban status vocabulary.
Blocking behavior: Phase Task creation cannot be completed until repeated materialization proves one stable, non-duplicating Phase Task identity per Project Work Item.
Integration validation: Hermes idempotency tests prove duplicate materialization does not duplicate Phase Tasks or reserved gate metadata.

Acceptance Criteria:

- Given a BMAD story has been materialized into Hermes project work, when Hermes creates the phase task, then exactly one Phase Task is linked to the Project Work Item.
- Given operational work state is returned, when a later gate is pending, then the result uses canonical `blocked` status plus `gate_kind` metadata without changing canonical Kanban lifecycle values.
- Given a Project Work Item has a Phase Task, when inspected, then Hermes returns selected story, phase metadata, workflow references, gate metadata when available, next action, observed BMAD status, target BMAD status, and artifact references.

## Epic 3: Workflow Provider Control And Event Delivery

Kevin can connect Hermes to workflow providers generically, control workflow runs through provider command JSON, and see event delivery or outbox health without opening the provider dashboard.
Hermes owns the consumer side.
Archon owns the producer side for provider `archon`.

### Story 3.2: Register And Diagnose Workflow Provider Bindings From Hermes

As a workflow operator,
I want Hermes to register and inspect workflow provider bindings using generic provider identity,
so that the selected provider can route workflow events to the correct Hermes profile without Hermes-specific vocabulary.

Requirements Covered: FR-7.
Implementation Scope: Hermes workflow provider binding registration flow, status display, conflict detection, and lifecycle command adapter.
Depends on: parent Story 1.3a, hermes-agent Story 2.1c, and Archon Story 3.1.
Contract needed: Workflow Provider Binding schema, generic `provider` and `name` vocabulary, binding status result shape, and malformed JSON failure shape.
Blocking behavior: Hermes registration and diagnostics cannot be completed until Hermes has a valid Project Binding and provider `archon` exposes provider binding lifecycle output.
Integration validation: Hermes parses provider binding examples once present, associates results with Project Bindings, and blocks automation when Hermes and provider metadata conflict.

Acceptance Criteria:

- Given a valid Hermes Project Binding with workflow provider metadata, when the user registers the workflow provider binding for provider `archon`, then Hermes uses the provider adapter command that identifies the controller by generic `provider` and `name` and does not require provider commands or models named specifically for Hermes.
- Given the provider adapter returns binding status, when Hermes inspects the binding, then Hermes can show whether the binding is missing, valid, stale, disabled, rotated, or conflicting and links the diagnostic to the affected Project Binding.
- Given the provider's stored binding disagrees with Hermes Project Binding metadata, when Hermes validates the relationship, then Hermes marks the binding as conflicting and blocks state-changing automation that depends on that binding until the conflict is resolved.
- Given a workflow provider binding needs rotation, removal, or disabling, when the user performs the lifecycle action from Hermes, then Hermes invokes the generic provider adapter command and records the CLI result, correlation id, actor, timestamp, and resulting binding state.
- Given the provider adapter returns malformed JSON or an unexpected exit code for a binding action, when Hermes receives the result, then Hermes fails closed and surfaces an actionable diagnostic without mutating Project Work state.

### Story 3.4a: Start And Inspect Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to start and inspect provider workflow runs through parseable adapter results,
so that Hermes can create and refresh workflow references without relying on a provider dashboard.

Requirements Covered: FR-8.
Implementation Scope: Hermes strict provider adapter for workflow start and status commands.
Depends on: parent Story 1.3a, hermes-agent Story 2.1c, hermes-agent Story 2.3, hermes-agent Story 3.2, Archon Story 3.3a, and Archon Story 3.3b.
Contract needed: Workflow command envelope, start result schema, status result schema, timeout shape, success shape, and error shape.
Blocking behavior: Hermes start and status control cannot be completed until provider command examples are locally available and Hermes can fail closed on malformed or incompatible results.
Integration validation: Hermes adapter tests parse provider `archon` start and status examples once present, invoke the real provider adapter from the Bound Project Cwd, prove it uses the Story 2.3 cwd guard, record stdout, stderr, exit code, timeout, correlation id, and update only allowed workflow reference or diagnostic state.

Acceptance Criteria:

- Given a valid Project Binding and valid workflow provider binding, when the user starts a provider workflow run from Hermes, then Hermes invokes the selected provider adapter from the Bound Project Cwd and records stdout, stderr, exit code, timeout, correlation id, workflow name, workflow run reference, and parsed JSON result.
- Given the selected provider adapter requires cwd, when Hermes invokes provider start or status commands, then Hermes reuses the Story 2.3 cwd guard and proves the real adapter runs from the Bound Project Cwd.
- Given Hermes needs workflow status, when the user or reconciliation process requests status for a provider workflow run, then Hermes invokes the provider adapter status command and updates only the workflow reference or diagnostic state allowed by the parsed schema.
- Given a provider start or status command returns malformed JSON, a schema-version mismatch, timeout, or unexpected exit code, when Hermes processes the result, then Hermes fails closed and does not update Project Work state as if the command succeeded.

### Story 3.4b: Send Provider Decision Commands From Hermes

As a workflow operator,
I want Hermes to send approve and reject commands to the selected workflow provider through parseable adapter results,
so that human gate decisions can drive provider workflow progress without conflating user approval with command execution.

Requirements Covered: FR-8 and FR-14.
Implementation Scope: Hermes strict provider adapter for approve and reject commands, command-result persistence, and fail-closed diagnostics only.
Depends on: parent Story 1.3a, hermes-agent Story 2.1c, hermes-agent Story 3.2, hermes-agent Story 3.4a, Archon Story 3.3a, and Archon Story 3.3c.
Contract needed: Workflow command envelope, approve result schema, reject result schema, timeout shape, success shape, and error shape.
Blocking behavior: Hermes decision commands cannot be completed until Hermes records human decision state separately from provider command results.
Integration validation: Hermes adapter tests parse approve and reject examples once present, fail closed on malformed JSON, and keep command results separate from HILT Gate decision records.

Acceptance Criteria:

- Given a workflow run waits for a decision, when a user approves or rejects from Hermes, then Hermes sends the matching provider command and records command result separately from the human gate decision.
- Given an approve or reject command succeeds, when Hermes records the result, then it stores correlation id, workflow run reference, actor when available, timestamp, and parsed result state.
- Given the command fails or returns incompatible output, when Hermes processes the result, then Hermes surfaces diagnostics without mutating Project Work as if the decision succeeded.

### Story 3.4c: Resume, Retry, And Cancel Provider Workflow Runs From Hermes

As a workflow operator,
I want Hermes to resume, retry, and cancel provider workflow runs through parseable adapter results,
so that recovery actions are recorded consistently and do not depend on human-readable provider output.

Requirements Covered: FR-8.
Implementation Scope: Hermes strict provider adapter for resume, retry, cancel, timeout, and unexpected-state handling.
Depends on: parent Story 1.3a, hermes-agent Story 2.1c, hermes-agent Story 3.2, hermes-agent Story 3.4a, Archon Story 3.3a, and Archon Story 3.3d.
Contract needed: Workflow command envelope, resume result schema, retry result schema, cancel result schema, timeout shape, success shape, and error shape.
Blocking behavior: Hermes recovery commands cannot be completed until resulting run state or diagnostic state is recorded without relying on human-readable output.
Integration validation: Hermes adapter tests parse resume, retry, cancel, timeout, and unexpected-state examples once present and update only allowed workflow reference or diagnostic state.

Acceptance Criteria:

- Given a workflow run can be resumed, retried, or cancelled, when the user selects that action, then Hermes invokes the matching adapter command and records resulting run state or diagnostic.
- Given a recovery command succeeds, when Hermes records the result, then it stores stdout, stderr, exit code, timeout, correlation id, workflow run reference, action name, and parsed result state.
- Given malformed JSON, schema mismatch, timeout, unexpected state, or unexpected exit code occurs, when Hermes processes the result, then Hermes fails closed.

### Story 3.6a: Validate Profile-Routed Workflow Event Ingress

As a workflow operator,
I want Hermes to validate workflow provider events against the profile route and profile-scoped secret before mutation,
so that events cannot cross project, binding, provider, or profile boundaries.

Requirements Covered: FR-9 and NFR-6.
Implementation Scope: Hermes workflow event ingress validation before idempotency receipts or project-work mutation.
Depends on: parent Story 1.3b, hermes-agent Story 2.1c, hermes-agent Story 3.2, and Archon Story 3.5.
Contract needed: Workflow event envelope schema, rejection examples, Project Binding identity, profile route, profile-scoped secret, provider identity, signature metadata, replay metadata, and rejection diagnostic shape.
Blocking behavior: Hermes workflow event ingress cannot be completed until invalid events are rejected before mutation and without exposing secrets.
Integration validation: Hermes accepts valid provider `archon` event examples once present and rejects bad signature, stale timestamp, wrong binding, unknown project, schema mismatch, unsupported provider, and valid signature under the wrong profile secret.

Acceptance Criteria:

- Given an event is delivered to Hermes, when ingress receives it on the profile-routed workflow event path, then Hermes validates schema version, signature, replay window, provider, event id, event type, binding reference, project or codebase reference, workflow run reference, idempotency key, profile route, and profile-scoped secret before mutation.
- Given a valid signature for the wrong profile secret, when validation runs, then Hermes rejects the event and records a redacted wrong-profile-secret diagnostic.
- Given any ingress validation fails, when Hermes evaluates mutation, then it rejects before project-work state changes.

### Story 3.6b: Persist Workflow Event Idempotency Receipts And Duplicate Diagnostics

As a workflow operator,
I want Hermes to persist workflow event idempotency receipts and duplicate-safe diagnostics,
so that redelivery does not create duplicate workflow references, gates, comments, or project-work transitions.

Requirements Covered: FR-9.
Implementation Scope: Hermes workflow event receipt persistence, duplicate classification, and duplicate-safe diagnostics.
Depends on: parent Story 1.3b, hermes-agent Story 2.1c, hermes-agent Story 3.6a, and Archon Story 3.5.
Contract needed: Workflow event idempotency receipt shape, duplicate event id rule, duplicate idempotency key rule, duplicate-safe marker, and diagnostic shape.
Blocking behavior: Workflow event receipt handling cannot be completed until duplicate events are classified without duplicate mutations.
Integration validation: Duplicate workflow event examples once present are accepted as duplicate-safe receipts without duplicating workflow references, gates, comments, or transitions.

Acceptance Criteria:

- Given Hermes receives a workflow event with a new event id and idempotency key after validation passes, when processing begins, then Hermes persists a receipt with Project Binding, provider, workflow run, event type, event id, idempotency key, received time, and processing state.
- Given Hermes receives duplicate event id or idempotency key, when processing runs, then Hermes treats delivery as duplicate-safe and avoids duplicate mutation.
- Given duplicate delivery is detected, when diagnostics are recorded, then Hermes links the duplicate-safe marker to the affected receipt.

### Story 3.6c: Map Workflow Outcome Events To Project Work

As a workflow operator,
I want Hermes to map accepted workflow outcome events to existing project work,
so that completion and failure update only the intended Phase Task.

Requirements Covered: FR-9, FR-11, and FR-12.
Implementation Scope: Hermes mapping for workflow-completed and workflow-failed events, workflow references, Phase Tasks, and result state.
Depends on: parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.1c, hermes-agent Story 2.6, hermes-agent Story 3.2, hermes-agent Story 3.6a, hermes-agent Story 3.6b, and Archon Story 3.5.
Contract needed: Workflow event envelope, Project Work Item identity, Phase Task identity, workflow reference, and outcome-event mutation map.
Blocking behavior: Outcome mapping cannot be completed until accepted completion and failure events resolve existing work without creating unintended records.
Integration validation: Completion and failure examples once present map deterministically, avoid duplicate work, and preserve state with diagnostics when identity cannot be resolved.

Acceptance Criteria:

- Given Hermes accepts a workflow-completed or workflow-failed event, when the event maps to existing project work, then Hermes updates the intended Phase Task without creating new work.
- Given an outcome event cannot be mapped deterministically, when Hermes evaluates mutation, then Hermes records an unresolved mapping diagnostic and leaves project-work state unchanged.

### Story 3.6d: Map Approval Request Events To Gate State

As a workflow operator,
I want Hermes to map accepted approval-request events to reserved gate state,
so that human decisions remain explicit and are never inferred from delivery.

Requirements Covered: FR-9 and FR-13.
Implementation Scope: Hermes approval-request mapping to existing Phase Tasks and reserved gate metadata.
Depends on: parent Story 1.3b, parent Story 1.3c, hermes-agent Story 3.6c, and Archon Story 3.5.
Contract needed: Approval-request event mapping, reserved gate identity, deliver-only notification marker, and Phase Task identity.
Blocking behavior: Approval-request delivery must not create a duplicate gate or imply approval.
Integration validation: Approval-request examples once present target the correct gate, remain replay-safe, and distinguish typed mutations from deliver-only notifications.

Acceptance Criteria:

- Given Hermes accepts an approval-request event, when it maps to an existing Phase Task, then Hermes records the pending gate target exactly once and does not record a human decision.

### Story 3.6e: Map Workflow Artifact Events And Diagnose Unresolved Mappings

As a workflow operator,
I want Hermes to map workflow-artifact events with source provenance,
so that evidence is available without unsafe or ambiguous project-work mutation.

Requirements Covered: FR-9 and FR-12.
Implementation Scope: Hermes workflow-artifact mapping to artifact or comment references and standardized unresolved event-mapping diagnostics.
Depends on: parent Story 1.3b, hermes-agent Story 3.6d, and Archon Story 3.5.
Contract needed: Workflow-artifact event mapping, artifact reference, comment reference, provenance, redaction rule, and unresolved-mapping diagnostic.
Blocking behavior: Unresolved artifact mappings must leave project-work state unchanged.
Integration validation: Artifact examples once present retain provenance, redact secrets, and create actionable diagnostics for unresolved identities.

Acceptance Criteria:

- Given Hermes accepts a workflow-artifact event with resolvable identity, when Hermes applies it, then Hermes records artifact or comment reference with source provenance and redaction.
- Given an accepted event cannot be mapped deterministically, when Hermes evaluates mutation, then Hermes records an unresolved mapping diagnostic and leaves project-work state unchanged.

### Story 3.8: Return Workflow Event Delivery And Outbox Health Status

As a workflow operator,
I want Hermes to return workflow event delivery and provider outbox health for each relevant workflow run,
so that delayed, failed, duplicated, or reconciliation-pending events are queryable and actionable.

Requirements Covered: FR-10.
Implementation Scope: Hermes structured workflow event health result, workflow diagnostic record, and reconciliation-needed marker.
Depends on: parent Story 1.3a, parent Story 1.3b, hermes-agent Story 2.1c, hermes-agent Story 3.6a, hermes-agent Story 3.6b, and Archon Story 3.7.
Contract needed: Workflow delivery status schema, retry state, terminal failure category, duplicate-safe marker, and reconciliation-needed marker.
Blocking behavior: Hermes workflow event health cannot be completed until it returns provider delivery states without treating event delivery as the sole source of truth.
Integration validation: Provider `archon` delivery-status examples once present drive Hermes health states for healthy, delayed, failed, duplicated, terminal failure, and waiting for reconciliation without unsafe project-work mutation.

Acceptance Criteria:

- Given a Project Work Item has a provider workflow reference, when an authorized caller queries workflow status, then Hermes returns workflow event delivery state as healthy, delayed, failed, duplicated, terminal failure, or waiting for reconciliation when that state is known and links the state to the affected workflow run and Project Binding.
- Given workflow event delivery is delayed or retrying, when Hermes receives or polls delivery status through CLI JSON, then Hermes returns the retry state, last attempt time if available, next action if available, and whether user action is required.
- Given workflow event delivery reaches terminal failure, when Hermes records the failure, then Hermes returns an actionable diagnostic with delivery status, last error category, affected event type, workflow run reference, and recovery option.
- Given workflow event delivery reaches terminal failure, when Hermes records the failure, then Hermes does not block workflow execution solely because event notification failed.
- Given Hermes detects duplicate workflow event delivery, when an authorized caller queries event delivery health, then Hermes identifies the event as duplicate-safe and returns evidence that no duplicate project-work mutation was applied.
- Given Hermes has incomplete workflow event evidence but other systems may have progressed, when reconciliation is needed, then Hermes marks the workflow or Project Work Item as waiting for reconciliation and exposes a user action or automated reconciliation path without silently completing work.

## Epic 4: Human-Gated Story Execution

Kevin can run the combined story workflow from Hermes, review done-verification evidence, approve or reject the gate, and route reruns or recovery.
Hermes owns the gate state and human decision record.
Provider command success remains transport evidence only.

### Story 4.1: Run Story Workflow

As a workflow operator,
I want Hermes to start the combined story workflow for a selected BMAD story,
so that story creation, implementation, and review run as one provider-controlled sequence.

Requirements Covered: FR-12 and FR-14.
Implementation Scope: Hermes combined story workflow start and workflow run reference recording on the Phase Task.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.6, hermes-agent Story 3.4a, hermes-agent Story 3.4b, hermes-agent Story 3.6a, hermes-agent Story 3.6b, hermes-agent Story 3.6c, hermes-agent Story 3.8, Archon Story 3.3b, and Archon Story 3.5.
Contract needed: Combined story workflow control result, workflow start command envelope, workflow completion event, and Phase Task identity.
Blocking behavior: This story cannot be completed until provider workflow control can start the combined workflow and Hermes event handling can record its run reference for the selected BMAD story.
Integration validation: A workflow-start example once present starts through the provider adapter and records the run reference on the Phase Task.

Acceptance Criteria:

- Given a Project Work Item has a Phase Task, when the user starts the story workflow from Hermes, then Hermes starts the configured combined story workflow and records the workflow run reference on the Phase Task.
- Given the combined workflow is running, when status is checked from Hermes, then Hermes returns structured workflow progress without requiring the provider dashboard.

### Story 4.2: Block Story Workflow For Done Verification

As a workflow operator,
I want Hermes to block the Phase Task for real done verification once the combined story workflow completes,
so that the story is only marked complete when workflow evidence, PR state, and human review support it.

Requirements Covered: FR-13 and FR-14.
Implementation Scope: Hermes Done Verification gate transition after combined workflow fix loop and PR creation complete.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.6, hermes-agent Story 3.4a, hermes-agent Story 3.4b, hermes-agent Story 3.4c, hermes-agent Story 3.6a, hermes-agent Story 3.6b, hermes-agent Story 3.6e, hermes-agent Story 3.8, hermes-agent Story 4.1, Archon Story 3.3c, Archon Story 3.3d, and Archon Story 3.5.
Contract needed: Combined workflow completion event, fix-loop result, PR reference, done verification gate record, and provider decision command result.
Blocking behavior: This story cannot be completed until Done Verification approval in Hermes is authoritative and GitHub merge alone is rejected as completion proof.
Integration validation: A workflow-completion example once present completes through provider evidence, blocks on `done_verification`, rejects GitHub merge alone, and routes rejection to rerun or recovery.

Acceptance Criteria:

- Given the combined story workflow completes, when Hermes receives completion evidence, then Hermes blocks the Phase Task with gate kind `done_verification` and does not complete story work until the gate is approved.
- Given the Phase Task is blocked, when an authorized caller queries the pending gate, then Hermes returns implementation evidence, workflow result, PR reference when available, unresolved issues when available, and next actions.
- Given the user approves or rejects Done Verification, when Hermes records the decision, then approval completes the Phase Task and rejection routes to rerun or recovery without marking the story complete.

### Story 4.3: Capture Human Gate Decisions And Evidence

As a workflow operator,
I want Hermes to return gate evidence, accept my approval or rejection, and persist the decision,
so that every HILT gate is auditable and can drive the matching provider control action when needed.

Requirements Covered: FR-14.
Implementation Scope: Hermes shared HILT Gate decision record hardening, structured evidence results, audit fields, rejection reason capture, recovery-action selection, delayed-decision behavior, and provider command-result association.
Depends on: parent Story 1.3a, parent Story 1.3c, hermes-agent Story 3.4b, hermes-agent Story 3.4c, hermes-agent Story 4.1, hermes-agent Story 4.2, Archon Story 3.3c, and Archon Story 3.3d.
Contract needed: Gate decision record, evidence reference shape, provider approve result schema, provider reject result schema, provider resume result schema, and provider retry result schema.
Blocking behavior: Gate decision capture cannot be completed until Hermes persists replay-safe human decisions and sends required provider control commands without conflating command result with human decision.
Integration validation: Approval and rejection examples once present persist actor, timestamp, gate kind, decision, evidence references, reason when provided, selected recovery action, command result when required, and resulting phase state.

Acceptance Criteria:

- Given a Phase Task is blocked on a HILT Gate, when Hermes publishes a notification or returns the pending gate, then Hermes returns gate kind, affected Project Work Item, Phase Task, BMAD story reference, workflow run reference, evidence references, and available decisions.
- Given a Phase Task is blocked on a HILT Gate, when Hermes publishes a notification or returns the pending gate, then Hermes phrases the next action in user-facing workflow language.
- Given the user approves a HILT Gate, when Hermes records the decision, then Hermes stores actor, timestamp, gate kind, decision, evidence references, reason if provided, and resulting phase state.
- Given the user approves a HILT Gate, when Hermes records the decision, then Hermes distinguishes approval from rejection through explicit decision values.
- Given the user rejects a HILT Gate, when Hermes records the decision, then Hermes stores actor, timestamp, gate kind, decision, rejection reason, evidence references, and selected recovery action.
- Given the user rejects a HILT Gate, when Hermes records the decision, then Hermes routes the phase task to rerun, resume, retry, or recovery according to the selected failure reason.
- Given a gate decision requires a provider approval, rejection, resume, or retry command, when Hermes records the human decision, then Hermes sends the matching command through the strict provider adapter and records the command result separately from the human decision record.
- Given the user ignores or delays a pending gate, when Hermes evaluates gate state, then Hermes keeps the phase task blocked and returns the pending gate and recovery options without auto-continuing unless an explicit persisted policy permits it.
- Given Hermes has a pending gate decision, when the gate is queried or mirrored through an existing notification transport, then the result identifies Project Binding, BMAD story, Phase Task, gate kind, and required decision and excludes secrets, raw workflow event signatures, and unredacted command output.

## Epic 5: Story Status History, Reconciliation, And Diagnostics

Kevin can query one Story Status History, understand why a story changed state, and resolve drift across BMAD, Hermes, workflow providers, GitHub, workflow events, and gates.
This epic turns execution records into trust, auditability, conflict handling, and recovery guidance.

### Story 5.1: Produce Unified Story Status History

As a workflow operator,
I want Hermes to return one Story Status History for each BMAD story,
so that I can understand current status, evidence, ownership, and next action without opening a provider dashboard.

Requirements Covered: FR-15.
Implementation Scope: Hermes versioned structured Story Status History projection and optional human-readable command formatting.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.6, hermes-agent Story 3.4a, hermes-agent Story 3.4b, hermes-agent Story 3.4c, hermes-agent Story 3.6a, hermes-agent Story 3.6b, hermes-agent Story 3.6e, hermes-agent Story 3.8, hermes-agent Story 4.1, hermes-agent Story 4.2, and hermes-agent Story 4.3.
Contract needed: Project Work Item state, Phase Task state, provider run reference, workflow event receipt, GitHub PR reference, HILT Gate decision record, redaction rule, and next-action vocabulary.
Blocking behavior: Story Status History cannot be completed until it returns provenance from BMAD, Hermes, providers, GitHub, workflow events, and human decisions without collapsing source-specific lifecycles.
Integration validation: Status-history examples once present return BMAD milestones, Project Work Item state, Phase Task state, provider references, workflow events, GitHub PR references, gate decisions, redacted sensitive details, and user-facing next action.

Acceptance Criteria:

- Given a Project Work Item is linked to a BMAD story, when an authorized caller queries its Story Status History, then Hermes returns BMAD artifact milestones, Project Work Item state, Phase Task state, provider workflow run references, workflow events, GitHub PR references if available, HILT Gate decisions, and next action.
- Given a Story Status History contains events from multiple systems, when the history is returned, then each entry is source-labeled as BMAD, Hermes, workflow provider, GitHub, workflow event, reconciliation, or human decision.
- Given BMAD artifact status and Hermes Kanban status differ, when Hermes produces the Story Status History, then Hermes labels each status with its source system and does not collapse BMAD artifact lifecycle into Hermes runtime Kanban lifecycle.
- Given GitHub PR state and Done Verification Gate state differ, when Hermes produces completion status, then Hermes returns PR state separately from done verification state and does not treat a merged PR as human done approval.
- Given a story has pending work, when Hermes produces the next action, then Hermes returns who owns the next action as user, Hermes, workflow provider, BMAD, GitHub, or implementation agent and phrases the next action in user-facing workflow language.
- Given the Story Status History includes sensitive command, workflow event, or diagnostic data, when Hermes returns the history, then Hermes redacts secrets and preserves enough provenance for the user to understand why the story changed state.

### Story 5.2a: Reconcile BMAD Materialization Drift

As a workflow operator,
I want Hermes to reconcile BMAD artifact changes against Project Work Items and Phase Tasks,
so that changed planning artifacts update existing project work without duplicating work items or Phase Tasks.

Requirements Covered: FR-16.
Implementation Scope: Hermes reconciliation for BMAD artifacts, `sprint-status.yaml`, Project Work Item materialization, and Phase Task identity.
Depends on: parent Story 1.3c, hermes-agent Story 2.5, hermes-agent Story 2.6, and hermes-agent Story 5.1.
Contract needed: BMAD source-state adapter, reconciliation result record, deterministic materialization repair rule, unresolved conflict marker, Project Work Item identity, and Phase Task identity.
Blocking behavior: BMAD materialization reconciliation cannot be completed until missing sprint status, malformed sprint status, changed story, unchanged story, and duplicate-safe update examples exist locally.
Integration validation: BMAD drift examples once present repair deterministic gaps, preserve unresolved conflicts, avoid duplicate Project Work Items or Phase Tasks, and never auto-approve a HILT Gate.

Acceptance Criteria:

- Given BMAD story artifact state changes after materialization, when reconciliation runs, then Hermes detects whether the Project Work Item needs update without duplicating Project Work Items or Phase Tasks.
- Given reconciliation performs deterministic repair, when Story Status History or diagnostics are queried, then Hermes returns what was repaired, why it was deterministic, and which BMAD source records supported it.

### Story 5.2b: Detect Provider Workflow State Drift

As a workflow operator,
I want Hermes to compare provider workflow state with Phase Task state,
so that provider drift is detected before any repair is attempted.

Requirements Covered: FR-16.
Implementation Scope: Hermes provider workflow source-state adapter, status comparison, unreliable-status handling, and reconciliation-result persistence without repairs.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.6, hermes-agent Story 3.4a, hermes-agent Story 3.4c, hermes-agent Story 3.6a, hermes-agent Story 3.6b, hermes-agent Story 3.6c, hermes-agent Story 3.8, hermes-agent Story 4.1, hermes-agent Story 4.2, hermes-agent Story 4.3, hermes-agent Story 5.1, hermes-agent Story 5.2a, and Archon Story 3.3b.
Contract needed: Provider workflow source-state adapter, status result schema, reconciliation result record, and unresolved provider diagnostic.
Blocking behavior: Drift detection must preserve the last known Hermes state when provider evidence is unreliable.
Integration validation: Provider status examples once present cover completed, failed, unavailable, malformed, timed-out, and unexpected states without applying repair.

Acceptance Criteria:

- Given provider workflow state differs from Hermes Phase Task state, when reconciliation compares them, then Hermes persists detected drift and source references without repair in this story.
- Given provider workflow status is unavailable, malformed, timed out, or unexpected, when reconciliation runs, then Hermes records an unresolved provider diagnostic and preserves last known Phase Task state.

### Story 5.2c: Reconcile Workflow Event Delivery Evidence

As a workflow operator,
I want Hermes to compare event receipts and delivery health with provider workflow state,
so that delayed, duplicated, lost, or terminally failed delivery is classified safely.

Requirements Covered: FR-16.
Implementation Scope: Hermes reconciliation of event receipts, duplicate markers, delivery health, gateway downtime evidence, and provider workflow state.
Depends on: parent Story 1.3a, parent Story 1.3b, hermes-agent Story 5.2b, and Archon Story 3.7.
Contract needed: Workflow event receipt source-state adapter, delivery health record, duplicate marker, and event-delivery drift category.
Blocking behavior: Event-evidence reconciliation must never apply duplicate project-work mutation.
Integration validation: Delivery examples once present classify delayed, duplicated, lost, terminally failed, and reconciliation-pending evidence without duplicate mutation.

Acceptance Criteria:

- Given event delivery is duplicated, delayed, terminally failed, or absent after provider completion, when reconciliation evaluates evidence, then Hermes links evidence to affected workflow run and records delivery-drift category without duplicate mutation.

### Story 5.2d: Apply Deterministic Provider Projection Repairs

As a workflow operator,
I want Hermes to apply only deterministic provider projection repairs,
so that recoverable drift is fixed without hiding conflicts or bypassing gates.

Requirements Covered: FR-16.
Implementation Scope: Hermes deterministic repair application, unresolved-conflict preservation, repair audit records, and Story Status History projection.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 5.2c, Archon Story 3.3b, Archon Story 3.5, and Archon Story 3.7.
Contract needed: Deterministic repair rule, repair audit record, workflow reference identity, gate identity, and unresolved conflict marker.
Blocking behavior: Repairs must not duplicate workflow references or gates and must never auto-approve a HILT Gate.
Integration validation: Repair examples once present explain every change, preserve unresolved conflicts, and remain idempotent.

Acceptance Criteria:

- Given provider or event drift has one deterministic repair, when reconciliation applies it, then Hermes records repair, source evidence, and resulting state without duplicating workflow references or gates.
- Given evidence conflicts or no deterministic repair exists, when reconciliation evaluates drift, then Hermes preserves current state and records unresolved diagnostics for authorized action.

### Story 5.2e: Reconcile GitHub And Done Verification Conflicts

As a workflow operator,
I want Hermes to reconcile GitHub PR state against Done Verification Gate state,
so that merged code never counts as completed story work without human done verification.

Requirements Covered: FR-16.
Implementation Scope: Hermes reconciliation for GitHub PR state, Done Verification Gate state, completion diagnostics, and conflict projection.
Depends on: parent Story 1.3c, hermes-agent Story 4.2, hermes-agent Story 4.3, hermes-agent Story 5.1, hermes-agent Story 5.2a, and hermes-agent Story 5.2d.
Contract needed: GitHub PR source-state adapter, Done Verification gate state, reconciliation result record, unresolved completion conflict marker, and recovery option vocabulary.
Blocking behavior: GitHub and Done Verification reconciliation cannot be completed until merged PR, unresolved Done Verification Gate, rejected Done Verification Gate, and completion diagnostic examples exist locally.
Integration validation: GitHub and Done Verification examples once present preserve completion conflicts, show recovery options, and never mark a story complete without Done Verification approval.

Acceptance Criteria:

- Given GitHub PR state indicates merged but Done Verification Gate is unresolved or rejected, when reconciliation evaluates completion, then Hermes records conflict and does not mark the story complete.
- Given GitHub PR evidence and Done Verification evidence disagree, when diagnostics or Story Status History are returned, then Hermes labels both source states separately and identifies next action owner.

### Story 5.3a: Define And Persist Operational Diagnostics

As a workflow operator,
I want Hermes to persist stable, source-linked diagnostics for orchestration problems,
so that failures remain auditable and safe to query.

Requirements Covered: FR-17.
Implementation Scope: Hermes diagnostic taxonomy, severity, affected-resource references, redacted evidence storage, next-action owner, recovery-option reference, persistence rules, and diagnostic family matrix. Query formatting remains in Story 5.3b. Resolution history remains in Story 5.3c.
Depends on: parent Story 1.3a, parent Story 1.3b, parent Story 1.3c, hermes-agent Story 2.1c, hermes-agent Story 2.3, hermes-agent Story 2.5, hermes-agent Story 3.2, hermes-agent Story 3.4a, hermes-agent Story 3.4b, hermes-agent Story 3.4c, hermes-agent Story 3.6e, hermes-agent Story 3.8, hermes-agent Story 4.1, hermes-agent Story 4.2, hermes-agent Story 4.3, hermes-agent Story 5.1, hermes-agent Story 5.2a, hermes-agent Story 5.2b, hermes-agent Story 5.2c, hermes-agent Story 5.2d, hermes-agent Story 5.2e, Archon Story 3.1, Archon Story 3.3a, Archon Story 3.3b, Archon Story 3.3c, Archon Story 3.3d, Archon Story 3.5, and Archon Story 3.7.
Contract needed: Diagnostic category vocabulary, diagnostic family matrix, affected-resource reference, recovery option vocabulary, redaction rule, persistence schema, and provider diagnostic payload shape.
Blocking behavior: Every required diagnostic family in the matrix must have a stable, source-linked, secret-safe record.
Integration validation: Diagnostic examples once present cover configuration, decision, external-delay, implementation-defect, duplicate-event, outbox, stale-PR, and unresolved-gate categories.

Diagnostic Family Matrix:

| Family | Required Sources | Next-Action Owner | Minimum Recovery Reference |
| --- | --- | --- | --- |
| configuration | Project Binding validation, Bound Project Cwd checks, BMAD mount checks, provider binding diagnostics | configuration action | repair binding, cwd, BMAD mount, or provider metadata |
| decision | Pending, approved, rejected, or rerouted HILT Gate decisions | user action | approve, reject, rerun, resume, retry, or recovery route |
| external-delay | Provider status unavailable, delayed delivery, gateway downtime, or reconciliation-pending evidence | external delay or provider action | retry status, inspect delivery state, or reconcile |
| implementation-defect | Malformed provider JSON, schema mismatch, unexpected state, or contract incompatibility | implementation-agent action or provider action | inspect contract compatibility and command evidence |
| duplicate-event | Duplicate event id or idempotency key evidence | Hermes automation | keep duplicate-safe receipt and avoid duplicate mutation |
| outbox | Retrying, failed, duplicated, terminal-failure, or reconciliation-pending delivery status | provider action | inspect outbox status or reconcile without blocking workflow execution solely on delivery failure |
| stale-PR | Stale, missing, or conflicting GitHub PR evidence | GitHub action or implementation-agent action | refresh PR reference, reconcile, or preserve completion conflict |
| unresolved-gate | Missing, unresolved, rejected, or conflicting Done Verification Gate evidence | user action | review evidence and approve, reject, or route recovery |

Acceptance Criteria:

- Given Hermes detects a supported orchestration problem in any diagnostic family listed in the matrix, when diagnostics are generated, then Hermes persists category, family, severity, affected references, redacted evidence, next-action owner, recovery option, source provenance, timestamp, and state.
- Given diagnostic evidence contains command output, workflow event payload fields, provider status, GitHub references, or gate evidence, when Hermes persists the diagnostic, then secrets and raw signatures are redacted before storage while preserving enough source-linked evidence for audit.
- Given duplicate workflow event or outbox evidence is diagnostic-only, when Hermes persists the diagnostic, then it records idempotency or delivery evidence without duplicating Project Work Item, Phase Task, gate, comment, or Story Status History mutation.
- Given evidence conflicts about completion, when Hermes generates diagnostics, then Hermes classifies unresolved completion evidence and does not silently mark work complete.

### Story 5.3b: Query Operational Diagnostics And Recovery Guidance

As a workflow operator,
I want structured diagnostic and recovery results,
so that I can distinguish configuration issues, user decisions, external delays, and implementation defects without a graphical interface.

Requirements Covered: FR-17.
Implementation Scope: Hermes versioned diagnostic query results, human-readable command formatting, redaction, next-action language, and recovery references.
Depends on: parent Story 1.3c, hermes-agent Story 5.1, and hermes-agent Story 5.3a.
Contract needed: Versioned diagnostic result, optional human-readable formatter contract, redaction rule, next-action owner vocabulary, and recovery reference.
Blocking behavior: Query output must be actionable without raw stack traces or secret disclosure.
Integration validation: Diagnostic query examples once present return category, severity, affected reference, redacted evidence, owner, recovery path, timestamp, and state.

Acceptance Criteria:

- Given a diagnostic is actionable, when an authorized caller queries it, then Hermes returns next-action owner and an appropriate recovery option or inspection path.
- Given diagnostic evidence contains command output, event payload details, or external references, when Hermes returns it, then secrets are redacted while enough context remains for debugging.

### Story 5.3c: Resolve Diagnostics And Preserve Audit History

As a workflow operator,
I want diagnostic resolution to preserve its evidence and outcome,
so that recovery is auditable after the active issue clears.

Requirements Covered: FR-17.
Implementation Scope: Hermes recovery-action association, resolution source, resolution timestamp, resulting state, and immutable diagnostic history.
Depends on: parent Story 1.3a, hermes-agent Story 3.4c, hermes-agent Story 5.2e, hermes-agent Story 5.3b, and Archon Story 3.3d.
Contract needed: Diagnostic resolution record, immutable history rule, recovery command result schema, recovery action vocabulary, and resulting state shape.
Blocking behavior: Clearing a diagnostic must record why and how it resolved without deleting prior evidence.
Integration validation: Resolution examples once present preserve the original diagnostic and append resolution source, timestamp, recovery action, command evidence when relevant, and resulting state.

Acceptance Criteria:

- Given a diagnostic has been resolved, when reconciliation or authorized action clears the issue, then Hermes records resolution source, timestamp, recovery action, and resulting state while keeping prior diagnostic history available for audit.
