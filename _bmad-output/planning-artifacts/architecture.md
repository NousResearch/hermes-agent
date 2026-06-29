---
title: hermes-agent Architecture Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-06-27'
updated: '2026-06-27'
---

# Architecture: hermes-agent Slice For Hermes Agent Workflow Commander

## Scope

This architecture handoff describes the Hermes-owned implementation slice for Hermes Agent Workflow Commander.
Hermes owns user-facing orchestration, Project Binding, BMAD mount, Bound Project Cwd enforcement, BMAD workflow invocation, project-work materialization, phase tasks, HILT gates, callback ingress, Story Timeline, reconciliation, diagnostics, and the strict Archon CLI adapter.
Archon owns generic Controller Binding, workflow run state, workflow CLI producer surfaces, callback outbox, signed callback production, and callback delivery health facts.

## Governing Decisions

- Hermes remains the human-facing command center and reconciliation owner.
- Hermes Project Binding is distinct from Archon Controller Binding.
- Hermes-to-Archon control uses a strict CLI adapter.
- Archon-to-Hermes mutation uses signed typed callback events that Hermes validates before mutation.
- `sprint-status.yaml` remains a BMAD planning and audit artifact, not the Hermes runtime queue.
- Hermes Kanban remains the runtime project-work owner.
- Done requires Hermes done verification even if Archon has completed and GitHub PR state is favorable.
- V1 required notification surface is the Hermes dashboard gate prompt plus blocked phase-task state.
- Existing Hermes notification channels may mirror gate prompts when available, but out-of-dashboard notification delivery is optional for v1.
- Workflow Commander reuses the Hermes dashboard shell, existing Kanban dashboard plugin, task drawer, task events, comments, runs, attachments, and parent-child task behavior.
- v1 uses the existing `hermes-agent` Python, FastAPI, Pydantic, Uvicorn, pytest, ruff, and local SQLite WAL substrate.

## Hermes Component Boundaries

```text
hermes-agent/
  hermes_project_work/
    bindings.py
    bmad_mount.py
    materialization.py
    phase_tasks.py
    gates.py
    archon_cli_adapter.py
    archon_callbacks.py
    reconciliation.py
    story_timeline.py
  tests/
    project_work/
```

### Project Binding

Project Binding stores profile identity, Bound Project Cwd, GitHub context, BMAD skill mount path, Archon Controller Binding metadata, display name, enabled state, and validation state.
Migration and uniqueness tests must prove profile, cwd, GitHub context, BMAD mount, and Archon metadata cannot conflict.

### BMAD Mount And Invocation

BMAD mounting updates the selected profile's `skills.external_dirs` and reloads the skill index.
BMAD workflow invocation runs from the Bound Project Cwd and records produced artifact paths.
Hermes must not infer artifact placement from skill visibility alone.
BMAD workflow invocation is a generic discovery, invocation, result capture, artifact recording, and diagnostic adapter unless a workflow declares a distinct contract.

### Materialization And Phase Tasks

Materialization reads `sprint-status.yaml` from the Bound Project Cwd and upserts Project Work Items idempotently.
Project Work Item identity is derived from bound project cwd, BMAD artifact path, and BMAD epic or story identity.
Project Work Item identity does not include phase kind.
Phase task creation splits each BMAD story into linked Prepare Story and Implement Story phase tasks.
Phase Task identity is derived from Project Work Item identity plus phase kind.
Idempotency tests must prove repeated materialization does not duplicate Project Work Items, phase tasks, or reserved gate metadata.

### Archon CLI Adapter

The strict Archon CLI adapter invokes Archon commands from the Bound Project Cwd and captures stdout, stderr, exit code, timeout, correlation id, workflow name, workflow run reference, and parsed JSON.
The adapter fails closed on malformed JSON, schema mismatch, timeout, or unexpected exit code.

### Callback Ingress

The profile-routed `archon-event` callback path validates schema version, signature, replay window, event id, event type, controller binding reference, project or codebase reference, workflow run reference, idempotency key, profile route, profile-scoped secret, and authorization.
Hermes stores callback receipts and does not duplicate workflow references, gates, comments, or project-work transitions on duplicate delivery.
Hermes rejects callbacks signed with a valid secret for the wrong profile before mutation.

### Gates

Prepare Story blocks on `test_case_adequacy`.
Implement Story blocks on `done_verification`.
Gate decisions store actor, timestamp, gate kind, decision, evidence references, reason when provided, selected recovery action when rejected, command result when required, and resulting phase state.

### Timeline, Reconciliation, And Diagnostics

Story Timeline renders source-specific state from BMAD, Hermes, Archon, GitHub, callbacks, and human gates.
Reconciliation repairs deterministic projection gaps and preserves unresolved conflicts.
Diagnostics classify configuration issues, user decisions, external delays, implementation defects, duplicate callbacks, outbox backlog, stale PR references, and unresolved gates with actionable recovery paths.

## Contract Surface

Shared contract fixtures are planned under `_bmad-output/planning-artifacts/contracts/workflow-commander/`.
The relevant subset must be regenerated into this local handoff before Hermes consumer code depends on it.

Hermes consumes these shared examples:

- Controller Binding payload and status fixtures.
- Workflow control CLI success and error envelopes.
- Callback event envelope fixtures.
- Callback rejection fixtures, including valid signature under the wrong profile secret.
- Callback delivery status fixtures.
- Project Work Item identity fixtures for new, unchanged, changed, missing, malformed, and duplicate-prevention cases.
- Phase Task identity fixtures for Prepare Story and Implement Story stability across reruns.

## Dependency Notes

Depends on: local shared contract examples plus Archon producer stories for Controller Binding, workflow control CLI JSON, callback events, and delivery health.
Contract needed: Controller Binding payload schema, workflow CLI result envelope, callback event envelope, callback delivery status schema, Project Work Item identity schema, Phase Task identity schema, gate record shape, and reconciliation result shape.
Blocking behavior: Hermes consumer stories cannot complete until relevant Archon fixtures exist and Hermes tests parse them.
Integration validation: Hermes pytest fixtures consume Archon producer examples, reject invalid callback examples, and prove duplicate-safe project-work mutation.

## Validation

Run from inside `hermes-agent`.

```text
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Validation must include Project Binding uniqueness, Bound Project Cwd enforcement, BMAD mount diagnostics, materialization idempotency, strict CLI adapter failure modes, callback validation, gate decision auditability, Story Timeline redaction, reconciliation drift cases, and diagnostic recovery paths.
