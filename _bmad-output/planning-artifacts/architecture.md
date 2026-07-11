---
title: hermes-agent Architecture Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-11'
updated: '2026-07-11'
source: workflow-engine parent architecture materialized as local hermes-agent input
local_contract_package: contracts/workflow-commander/
---

# Architecture: hermes-agent Slice For Hermes Agent Workflow Commander

## Scope

This file is the local architecture handoff for the `hermes-agent` slice of Hermes Agent Workflow Commander.
It is complete enough for downstream Hermes implementation agents to work from the `hermes-agent` subproject without parent planning files.
It describes Hermes-owned architecture only.
Archon producer internals are represented only as provider contracts that Hermes consumes.

Implementation root:

```text
hermes-agent/
```

## Design Paradigm

The architecture uses Bounded Context plus Ports and Adapters plus Outbox/Reconciliation.
Hermes is the human-facing command center and reconciliation owner.
BMAD owns planning artifacts and story workflow artifacts.
Workflow providers own workflow execution primitives, workflow run state, retry behavior, approval pauses, event production, and delivery.
Archon is the first workflow provider.
GitHub owns pull request and merge state.
Hermes Kanban owns runtime project-work records, while BMAD `sprint-status.yaml` remains planning and audit input only.

## Architecture Decisions For Hermes

### AD-1: Bounded Contexts With Ports, Adapters, Outbox, And Reconciliation

Hermes, BMAD, workflow providers, GitHub, and Kanban remain separately owned contexts.
Hermes connects to those contexts only through explicit ports, machine contracts, and reconciliation records.

### AD-2: Split Project Binding And Workflow Provider Binding Ownership

Hermes owns the forward Project Binding for profile, cwd, GitHub context, BMAD mount, operational status, and headless user interaction.
Each workflow provider owns the reverse binding from provider execution context to generic controller `provider`, `name`, and workflow event route.
Hermes stores provider binding metadata and diagnoses mismatch, but it does not own provider-side binding persistence.

### AD-3: Control Providers Through Adapters And Receive Signed Typed Events

Hermes state-changing workflow control uses a provider adapter that captures cwd when applicable, stdout, stderr, exit code, timeout, correlation id, and parsed JSON result.
Provider-to-Hermes mutation uses signed typed workflow events accepted only after schema, signature, replay, idempotency, profile, provider, and binding checks.
For provider `archon`, Hermes consumes CLI JSON and signed workflow events rather than provider dashboard state.

### AD-4: Make Hermes Kanban The Runtime Project-Work Owner

`sprint-status.yaml` is a BMAD planning and audit artifact only.
Hermes materializes each BMAD story idempotently into one Project Work Item and one Phase Task over canonical Kanban status plus gate metadata.

### AD-5: Put Cross-System Reconciliation Under Hermes Authority

Hermes owns reconciliation across BMAD artifacts, Hermes project work, provider run state, GitHub PR state, workflow events, and gate records.
Hermes may auto-repair deterministic projection drift.
Hermes must not auto-approve HILT Gates or mark stories complete when evidence conflicts.

### AD-6: Split Implementation Ownership By Subproject

`hermes-agent` owns user and project orchestration, Project Binding, BMAD mount, materialization, Project Work Items, Phase Tasks, HILT Gates, Story Status History, reconciliation, workflow provider adapter registration, provider command result consumption, workflow event ingress, and provider-neutral diagnostics.
Archon owns the first provider implementation and producer-side workflow contracts.
Hermes implementation stories consume those producer contracts through explicit dependency records.

### AD-7: Version Every Cross-Subproject Machine Contract

Workflow command envelopes, workflow event envelopes, Workflow Provider Binding records, provider delivery status records, Project Work Item identity, Phase Task identity, gate decision records, and diagnostic records are JSON and schema-versioned.
Schemas alone do not make downstream integration ready.
Each story must validate the specific examples it consumes once those example files exist locally.

### AD-8: Ratify The Brownfield Stack And Avoid New Runtime Infrastructure

Hermes stays on the existing local `hermes-agent` runtime.
The current Python package metadata reports `hermes-agent` version `0.18.0` and Python `>=3.11,<3.14`.
Current core package metadata includes OpenAI `2.24.0`, Pydantic `2.13.4`, pytest `9.0.2`, Ruff `0.15.10`, FastAPI `>=0.104.0,<1`, Uvicorn `>=0.24.0,<1`, and Pillow `12.2.0`.
The root Node workspace requires Node `>=20.0.0`.
The TUI stack uses React 19, Ink 6, Vitest 4, and TypeScript 6.
The web stack uses React 19, Vite 8, and TypeScript 6.
The desktop package has separate desktop metadata and should not be used as the Hermes Agent runtime version.
Workflow Commander v1 does not add a new shared database, runtime service, cloud queue, or dedicated graphical frontend.

### AD-9: Build Contract-First, Then Split Implementation By Subproject

Parent contract stories define shared schemas and examples before producer and consumer stories can complete.
Hermes downstream stories treat parent Stories 1.3a, 1.3b, and 1.3c as contract sources.
Hermes consumer work must fail closed when provider command, event, delivery, materialization, identity, gate, or diagnostic payloads are missing or incompatible.

### AD-10: Materialize Isolated Subproject Planning Handoffs Before Implementation

The flat local `README.md`, `prd.md`, `architecture.md`, and `epics.md` files in this folder are the active Hermes implementation inputs.
They are ordinary local files and should not rely on symlinks.
Downstream implementation workflows run from `hermes-agent`.

## Hermes-Owned Structural Seed

```text
hermes-agent/
  hermes_project_work/
    bindings.py
    bmad_mount.py
    materialization.py
    phase_tasks.py
    gates.py
    workflow_providers/
      base.py
      archon.py
    provider_commands.py
    workflow_events.py
    reconciliation.py
    story_status.py
  tests/
    project_work/
```

The seed is planning guidance, not a command to create every file in one story.
Downstream stories should create only the code needed for their specific acceptance criteria.

## Core Entity Shape

```text
Project Binding owns many Project Work Items.
Project Binding references provider binding metadata.
Project Binding reads BMAD artifact references.
Project Work Item has exactly one Phase Task.
Phase Task may block on HILT Gate records.
Phase Task may reference provider workflow runs.
Provider workflow runs report workflow events.
Project Work Item observes GitHub PR references.
Project Work Item projects source-labeled Story Status History entries.
```

## Consistency Conventions

| Concern | Convention |
| --- | --- |
| Controller naming | Workflow providers use generic `provider` and `name` vocabulary. |
| Hermes naming | Hermes uses Project Binding, Project Work Item, Phase Task, HILT Gate, and Story Status History. |
| Binding direction | Hermes Project Binding points outward to cwd, GitHub, BMAD, and provider metadata. |
| Control direction | Hermes controls providers through adapters. |
| Event direction | Providers report events to Hermes through signed workflow event ingress. |
| Data format | Cross-subproject contracts use JSON with explicit schema version and examples. |
| Command envelope | Provider command results include schema version, success flag, correlation id, run or binding reference, result payload, and error shape. |
| Workflow event envelope | Workflow events include schema version, event id, event type, occurred timestamp, provider binding reference, workflow run reference, project or codebase reference, signature metadata, and idempotency key. |
| Project Work Item identity | Hermes derives Project Work Item identity from Bound Project Cwd, BMAD artifact path, and BMAD epic or story identity. |
| Phase Task identity | Hermes derives Phase Task identity from Project Work Item identity plus phase kind. |
| Kanban lifecycle | Canonical status remains `triage`, `todo`, `ready`, `running`, `blocked`, `done`, and `archived`. |
| Gate interaction | V1 uses durable pending-gate queries, authorized decision commands, and canonical `blocked` plus `gate_kind=done_verification`. |
| Completion semantics | Done requires Hermes Done Verification even when provider workflow and GitHub evidence are favorable. |
| Drift handling | Deterministic drift may be repaired automatically, while conflicting evidence routes to diagnostics. |

## Contract Package Reality

The local contract package is `contracts/workflow-commander/`.
The local package currently includes schema files under `schemas/`.
No example fixture files were observed under a local `examples/` folder during this handoff refresh.
Downstream stories must keep example fixture dependencies blocked until the specific required example files exist locally and compatibility tests load them.

## Implementation Validation Gates

| Contract Area | Owner | Gate Before Hermes Story Completion |
| --- | --- | --- |
| Workflow command envelope and provider binding | Parent Story 1.3a with Archon producers | Hermes consumer tests load command and binding schemas plus required examples once present. |
| Workflow event envelope and rejection cases | Parent Story 1.3b with Archon producers | Hermes ingress tests load signed, stale, duplicate, wrong-binding, wrong-profile, and schema-failure examples once present. |
| Materialization and phase identity | Parent Story 1.3c | Hermes materialization tests load Project Work Item and Phase Task identity examples once present. |
| Provider delivery status | Parent Story 1.3a or 1.3b with Archon delivery producer | Hermes health tests classify healthy, delayed, duplicated, failed, terminal, and reconciliation-needed states once examples exist. |
| Gate decision record | Parent contract work with Hermes gate stories | Hermes gate tests keep human decision records separate from provider command results. |
| Operational diagnostics | Parent contract work with Hermes diagnostic stories | Hermes diagnostics tests redact secrets and preserve source-linked recovery evidence. |

## Candidate Validation Commands

Run downstream Hermes implementation workflows from the implementation root.
Candidate validation commands are:

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```
