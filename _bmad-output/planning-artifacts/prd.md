---
title: hermes-agent Planning Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-02'
updated: '2026-07-10'
source: workflow-engine parent workspace, materialized per cross-project-isolated-handoff-contract.md
---

# PRD: hermes-agent Slice For Hermes Agent Workflow Commander

## Purpose

Hermes Agent Workflow Commander makes Hermes Agent the human-facing, headless command surface for BMAD planning, Archon workflow execution, GitHub PR state, and local project work. `hermes-agent` owns nearly all requirements: Project Binding, BMAD mount, materialization, Phase Tasks, HILT Gates, provider adapter consumption, workflow event ingress, Story Status History, reconciliation, and diagnostics. Archon is the first workflow provider `hermes-agent` controls.
The full parent PRD lives at `_bmad-output/planning-artifacts/prds/prd-workflow-engine-2026-06-26/prd.md` in the parent workspace; this file exists so no `hermes-agent` implementation agent needs to read it.

## Vision (unabridged — this is hermes-agent's core product)

Hermes Agent Workflow Commander turns Hermes Agent into a human-facing, headless command surface for agentic implementation work. V1 provides command, agent, and API results; it does not provide a dedicated graphical dashboard or frontend. For v1, Archon is the first workflow provider.
The first release prioritizes reliability, explicit project binding, human trust, auditable decisions, and clean handoff to architecture and epics. [ASSUMPTION] This is an internal workflow-engine product for trusted local projects rather than a public SaaS launch.

## Target User / Key User Journeys (unabridged)

- **UJ-1. Kevin creates a project-bound commander workspace.** Kevin opens Hermes, chooses or creates a project binding, points it at the local repo cwd, connects GitHub context, registers the Archon controller binding. Edge case: conflicting bindings block automation and show the conflict instead of guessing.
- **UJ-2. Kevin materializes BMAD planning into Hermes project work.** Hermes reads `sprint-status.yaml` from the bound cwd, derives idempotency keys, creates/updates project-work tasks without duplicate cards. Edge case: unsupported shape surfaces a blocking validation issue, no partial duplicates.
- **UJ-3. Kevin starts the combined story workflow from Hermes.** Hermes controls the provider through its adapter; provider runs story creation, test design, implementation, and review as one continuous workflow. Edge case: delayed/duplicated completion events don't cause duplicate state changes.
- **UJ-4. Kevin verifies real done after implementation and PR evidence.** Hermes blocks the Phase Task for done verification; approval completes it, rejection reruns the fix loop. Edge case: if GitHub says merged but Hermes verification is rejected, the conflict surfaces rather than silently completing.

## Functional Requirements Owned By hermes-agent

FR-1 through FR-6 and FR-11 through FR-17 are fully owned. FR-7 through FR-10 are shared with Archon — this section states hermes-agent's consumer-side obligation for each; Archon's producer-side obligation lives in its own local PRD.

**FR-1: Create and view Project Bindings** — create, view, update, disable, and validate a Project Binding (profile identity, Bound Project Cwd, GitHub reference, BMAD skill directory reference, workflow provider binding metadata, display name). Rejects invalid cwd; shows active binding before any workflow; persists enough to reconstruct status after restart; shows whether provider binding is missing/valid/stale/conflicting.

**FR-2: Mount project-local BMAD skills** — add the bound project's BMAD skill directory to `skills.external_dirs`, reload skill index. Skills appear in discovery; records source directory; does not use global `~/.hermes/skills` as primary mount; detects missing or wrong-project mount.

**FR-3: Enforce Bound Project Cwd for workflow actions** — run BMAD and provider actions from the Bound Project Cwd unless stated otherwise. Artifacts land under the bound project's `_bmad-output`; blocks actions when cwd missing; audit records include cwd used; never infers cwd from skill visibility alone.

**FR-4: Invoke BMAD planning workflows from Hermes** — invoke brainstorming, product brief, PRD, architecture, epics, stories, sprint status, create-story, dev-story preparation from the Bound Project Cwd. Presents BMAD as behind-the-scenes; records artifact paths; surfaces failure output without losing binding context.

**FR-5: Materialize BMAD sprint status into Project Work Items** — read `sprint-status.yaml`, idempotently create/update Project Work Items. Re-running updates instead of duplicating; stores artifact references; reports unsupported/missing/malformed data before mutating; never treats the file as the runtime queue.

**FR-6: Maintain Hermes-owned operational backlog** — persist operational backlog, selected story, phase metadata, workflow references, gate metadata, and next action; expose them through structured command, agent, or API results. Canonical lifecycle stays `triage`/`todo`/`ready`/`running`/`blocked`/`done`/`archived`.

**FR-7 (consumer side): Register generic workflow provider bindings** — create/update the provider-side binding using generic `provider`/`name` vocabulary, detect disagreement with Archon's stored binding, surface rotation/removal/stale/missing states.

**FR-8 (consumer side): Control provider workflows through provider adapters** — start, check status, approve, reject, resume, retry, cancel through the adapter. Uses parseable JSON; captures stdout/stderr/exit code/cwd/timeout/correlation id; fails closed on malformed JSON/unexpected exit code; no HTTP for the `archon` state-changing path.

**FR-9 (consumer side): Receive typed workflow provider events** — receive signed events at `/p/{profile}/webhooks/workflow-events/{provider}`, validate schema/binding/replay/idempotency/authorization before mutating. Rejects unknown binding/wrong profile/wrong codebase/stale timestamp/duplicate id/invalid signature/unsupported provider/schema failure; stores accepted event ids for duplicate-safety; maps events to the correct Project Work Item.

**FR-10 (consumer side): Return provider event delivery and outbox health** — return structured healthy/delayed/failed/duplicated/reconciliation-pending status through Story Status History and diagnostic queries; never block workflow execution solely on notification failure.

**FR-11: Create one phase task per BMAD story** — each BMAD story has exactly one Phase Task sharing a Story Status History; repeated materialization never duplicates it.

**FR-12: Run the combined story workflow** — start the configured combined workflow for a selected story; record the run reference on the Phase Task; runs story creation through review without a Hermes-side pause in between.

**FR-13: Gate done verification** — block the phase task with gate kind `done_verification` after workflow provider reports completion; human approval completes the task; rejection reruns the fix loop or routes recovery.

**FR-14: Collect human decisions from Hermes** — publish or return structured gate evidence, accept approval/rejection through authorized commands or agent interaction, persist the decision, and send the matching provider command when required. Never auto-continue past a HILT Gate without explicit persisted policy.

**FR-15: Return a unified Story Status History** — return BMAD milestones, Project Work Item and Phase Task state, provider runs, workflow events, GitHub PR references, HILT decisions, provenance, and next action in one versioned structured result.

**FR-16: Reconcile cross-system state** — compare BMAD artifact state, provider workflow state, GitHub PR state, Project Work Item state, HILT Gate state to detect drift. Detects completed-but-unapplied provider runs, GitHub-merge-vs-unresolved-gate conflicts, unmaterialized BMAD changes; reports automatic repair vs. needs-human-action.

**FR-17: Provide operational diagnostics** — surface binding conflicts, cwd problems, missing artifacts, unsupported sprint status, provider command gaps, delivery failures, duplicate events, outbox backlog, stale PR references, unresolved gates. Diagnostics distinguish user/configuration/implementation-defect/external-delay; never silently mark work complete on conflicting evidence.

## Non-Functional Requirements Owned By hermes-agent (nearly all apply)

**Reliability:** NFR-1 (workflow events are delivery acceleration, not sole source of truth) · NFR-2 (reconcile after event loss/duplicate/gateway downtime/command failure/manual PR merge) · NFR-3 (materialization must be idempotent) · NFR-4 (gate decisions replay-safe and auditable).

**Security and Safety:** NFR-5 (reject events failing signature/schema/replay/binding/provider/authorization checks) · NFR-6 (scope event secrets to correct profile) · NFR-7 (prevent workflow actions outside Bound Project Cwd) · NFR-8 (redact secrets in command logs, event logs, diagnostics, and status-history results).

**Auditability:** NFR-9 (persist workflow commands, events, reconciliation actions, gate decisions, state transitions) · NFR-10 (Story Status History explains why a story changed state) · NFR-11 (state changes carry source provenance).

**Usability:** NFR-12 (phrase next actions in user-facing language, not backend state names) · NFR-13 (distinguish Done Verification Gate approval from GitHub PR merge state) · NFR-14 (surface blocking issues with recovery options, not raw stack traces).

**Maintainability:** NFR-15 (preserve bounded ownership between Hermes/BMAD/providers/GitHub) · NFR-16 (new provider integration surfaces stay generic) · NFR-17 (cross-project handoffs complete enough for isolated agents).

## Headless Product Surface

Workflow Commander v1 does not ship a dedicated web dashboard, graphical Kanban board, gate screen, timeline screen, or other graphical frontend. Commands, agent interactions, APIs, durable records, and optional existing notification transports are authoritative.

## Non-Goals (hermes-agent-relevant subset)

- Will not replace BMAD, Archon, GitHub, or Hermes Kanban with a single monolithic workflow database.
- Will not require the user to operate Archon's dashboard for normal workflow control.
- Will not use Archon HTTP APIs for the Hermes-to-Archon control path.
- Will not treat `sprint-status.yaml`, GitHub Issues, or Archon UI as Hermes runtime queue state.
- Will not rely on global `~/.hermes/skills` as the primary BMAD mount mechanism.
- Will not auto-approve HILT Gates without explicit persisted policy and evidence requirements.
- Will not ship a dedicated graphical Workflow Commander frontend in v1.

## Glossary (hermes-agent-relevant terms)

Bound Project Cwd, Controller Identity, Done Verification Gate, HILT Gate, Materialization, Phase Task, Project Binding, Project Work Item, Reconciliation, Story Status History — full definitions are in parent PRD §3.

## Cross-Project Dependencies

`hermes-agent`'s consumer/adapter stories depend on Archon's producer stories (see `Archon/_bmad-output/planning-artifacts/epics.md`). Story-level dependency records use:

```text
Depends on: <subproject> Story <id or title>
Contract needed: <API/event/file/interface/schema>
Blocking behavior: <what must exist before this story can be completed>
Integration validation: <how both sides will be proven compatible>
```

## Source

Derived from the parent workspace's canonical PRD and epics, current as of 2026-07-10 after the approved headless correction.
