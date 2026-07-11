---
title: Cross-Project Isolated Handoff Contract
status: active
created: '2026-07-12'
updated: '2026-07-12'
source: restored from local hermes-agent PRD, architecture, epics, project context, and workflow-commander contract package
---

# Cross-Project Isolated Handoff Contract

## Purpose

This file is the persistent fact used by readiness and correction workflows to keep the `hermes-agent` Workflow Commander handoff isolated from parent planning files.
It records only facts that are already present in local planning artifacts.
Implementation agents may use the local artifacts listed here without reading parent workspace planning files.

## Canonical Local Inputs

- `_bmad-output/planning-artifacts/prd.md`
- `_bmad-output/planning-artifacts/architecture.md`
- `_bmad-output/planning-artifacts/epics.md`
- `_bmad-output/planning-artifacts/ux-headless-interaction-contract.md`
- `_bmad-output/planning-artifacts/contracts/workflow-commander/README.md`
- `_bmad-output/project-context.md`

The flat local PRD, architecture, epics, UX contract, and contract package are ordinary local files and are the active Hermes implementation inputs.
They must remain complete enough for downstream `hermes-agent` implementation agents to work from the `hermes-agent` subproject root.

## Isolation Rules

- Downstream implementation agents should not read parent planning files to complete Hermes stories.
- Story numbering is preserved from the parent workspace only so cross-project dependency records stay stable.
- Parent contract stories, Archon producer stories, parent handoff stories, and parent validation stories remain dependencies, not Hermes-owned implementation work.
- Any story that depends on parent shared contract work or Archon producer output must keep an explicit dependency record with `Depends on`, `Contract needed`, `Blocking behavior`, and `Integration validation`.
- A story may not claim fixture readiness unless the matching local fixture family exists under `contracts/workflow-commander/examples/` and validates through the local contract validator.
- Provider-dependent Hermes stories may be planned against local fixtures, but they may not be marked complete from fixtures alone when compatible Archon producer output is still unproven.

## Ownership Boundary

- Hermes owns Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD invocation, materialization, Project Work Items, Phase Tasks, HILT Gates, provider adapter consumption, workflow event ingress, Story Status History, reconciliation, diagnostics, and headless validation guidance.
- BMAD owns planning artifacts and story workflow artifacts.
- Archon owns producer-side workflow execution, run state, provider binding, provider command JSON, event outbox, delivery status, signed event production, and producer-side provider binding persistence.
- GitHub owns pull request state.
- Parent shared contract stories own shared schemas and example fixture families before downstream consumer or producer stories can complete.

## Contract Package State

The local shared contract package is `_bmad-output/planning-artifacts/contracts/workflow-commander/`.
It includes schema files under `schemas/` and 65 JSON examples under `examples/`.
The example inventory is:

| Fixture family | Count |
| --- | ---: |
| Archon command examples | 16 |
| Archon provider-binding examples | 14 |
| Delivery-status examples | 7 |
| Generic workflow-event examples | 6 |
| Archon provider-event examples | 7 |
| Callback-rejection examples | 9 |
| Materialization examples | 6 |

The canonical local validation command is:

```text
uv run python _bmad-output/planning-artifacts/contracts/workflow-commander/validate_contracts.py
```

Bare `python3` is not canonical for this handoff because a system Python may be older than the project floor of `>=3.11,<3.14`.

## Readiness Assertions

- Workflow Commander v1 remains headless and does not create a dedicated dashboard, desktop surface, web app, graphical Kanban board, gate screen, timeline screen, or marketing surface.
- NFR identifiers in PRD and epics must stay aligned as NFR-1 through NFR-17.
- Story 2.3 owns generic cwd enforcement and a minimal provider-action cwd test double; real Archon adapter cwd validation belongs to Story 3.4a.
- Story 5.3a owns the operational diagnostic taxonomy, persistence contract, and diagnostic family matrix; query formatting belongs to Story 5.3b and resolution history belongs to Story 5.3c.
- External Archon producer completion status is not proven by this local handoff. Provider-dependent stories must keep their Archon dependency records until compatible producer output is available and validated.

## External Archon Validation Blocker

This isolated handoff does not contain validated runtime output captured from the external Archon producer.
Do not infer producer compatibility from the local fixture package alone.
Before any provider-dependent Hermes story is marked `done`, the missing external evidence must be supplied and validated against the local contract package:

- Archon provider binding lifecycle output for create, update, status, rotate, disable, and remove paths.
- Archon workflow command output for start, status, approve, reject, resume, retry, cancel, timeout, schema mismatch, malformed request, unexpected exit, and unexpected state paths.
- Archon workflow event output for workflow started, completed, failed, approval requested, delivery failed, artifact recorded, and redelivery paths.
- Archon delivery and outbox status output for healthy, delayed, retrying, failed, duplicated, terminal-failure, and reconciliation-pending paths.
