---
title: hermes-agent Planning Handoff - Hermes Agent Workflow Commander
status: handoff
created: '2026-07-02'
updated: '2026-07-02'
source: workflow-engine parent workspace, materialized per cross-project-isolated-handoff-contract.md
---

# PRD: hermes-agent Slice For Hermes Agent Workflow Commander

## Purpose

Hermes Agent Workflow Commander makes Hermes Agent the human-facing command center for BMAD planning, Archon workflow execution, GitHub PR state, and local project work. `hermes-agent` owns nearly all of this product's requirements: Project Binding, BMAD mount, materialization, phase tasks, HILT gates, provider adapter consumption, workflow event ingress, Story Timeline, reconciliation, and diagnostics. Archon is the first workflow provider `hermes-agent` controls — see its own separate local handoff for the producer side.
The full parent PRD lives at `_bmad-output/planning-artifacts/prds/prd-workflow-engine-2026-06-26/prd.md` in the parent workspace; this file exists so no `hermes-agent` implementation agent needs to read it.

## Vision (unabridged — this is hermes-agent's core product)

Hermes Agent Workflow Commander turns Hermes Agent into the human-facing command center for agentic implementation work. The user should be able to brainstorm, plan, choose work, start implementation workflows, receive status, approve review gates, reject incomplete work, and understand story progress from Hermes without treating a provider dashboard as the primary control surface. For v1, Archon is the first workflow provider.
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

**FR-6: Maintain Hermes-owned operational backlog** — store operational backlog, selected story, phase metadata, workflow references, gate metadata, next action. User chooses next story without BMAD auto-picking; facade lanes (Ideas, Backlog, Active Runs, Review Test Cases, Verify Done) over canonical Kanban status; canonical lifecycle stays `triage`/`todo`/`ready`/`running`/`blocked`/`done`/`archived`.

**FR-7 (consumer side): Register generic workflow provider bindings** — create/update the provider-side binding using generic `provider`/`name` vocabulary, detect disagreement with Archon's stored binding, surface rotation/removal/stale/missing states.

**FR-8 (consumer side): Control provider workflows through provider adapters** — start, check status, approve, reject, resume, retry, cancel through the adapter. Uses parseable JSON; captures stdout/stderr/exit code/cwd/timeout/correlation id; fails closed on malformed JSON/unexpected exit code; no HTTP for the `archon` state-changing path.

**FR-9 (consumer side): Receive typed workflow provider events** — receive signed events at `/p/{profile}/webhooks/workflow-events/{provider}`, validate schema/binding/replay/idempotency/authorization before mutating. Rejects unknown binding/wrong profile/wrong codebase/stale timestamp/duplicate id/invalid signature/unsupported provider/schema failure; stores accepted event ids for duplicate-safety; maps events to the correct Project Work Item.

**FR-10 (consumer side): Surface provider event delivery and outbox health** — show whether delivery is healthy/delayed/failed/duplicated/waiting-for-reconciliation. Exposes status on Story Timeline; shows terminal failures as actionable diagnostics; never blocks workflow execution solely on notification failure.

**FR-11: Create one phase task per BMAD story** — each BMAD story has exactly one Phase Task sharing a Story Timeline; repeated materialization never duplicates it.

**FR-12: Run the combined story workflow** — start the configured combined workflow for a selected story; record the run reference on the Phase Task; runs story creation through review without a Hermes-side pause in between.

**FR-13: Gate done verification** — block the phase task with gate kind `done_verification` after workflow provider reports completion; human approval completes the task; rejection reruns the fix loop or routes recovery.

**FR-14: Collect human decisions from Hermes** — notify user, present gate evidence, capture approval/rejection, store the decision, send the matching command when required. Each decision records actor/timestamp/gate kind/decision/reason/evidence; approval and rejection are visibly distinct; never auto-continues past a HILT Gate without explicit persisted policy.

**FR-15: Show a unified Story Timeline** — show BMAD milestones, Project Work Item state, phase task state, provider run references, workflow events, GitHub PR references, HILT Gate decisions, next action in one place. Distinguishes BMAD artifact status from Hermes Kanban status; distinguishes GitHub PR merge state from Done Verification Gate state.

**FR-16: Reconcile cross-system state** — compare BMAD artifact state, provider workflow state, GitHub PR state, Project Work Item state, HILT Gate state to detect drift. Detects completed-but-unapplied provider runs, GitHub-merge-vs-unresolved-gate conflicts, unmaterialized BMAD changes; reports automatic repair vs. needs-human-action.

**FR-17: Provide operational diagnostics** — surface binding conflicts, cwd problems, missing artifacts, unsupported sprint status, provider command gaps, delivery failures, duplicate events, outbox backlog, stale PR references, unresolved gates. Diagnostics distinguish user/configuration/implementation-defect/external-delay; never silently mark work complete on conflicting evidence.

## Non-Functional Requirements Owned By hermes-agent (nearly all apply)

**Reliability:** NFR-1 (workflow events are delivery acceleration, not sole source of truth) · NFR-2 (reconcile after event loss/duplicate/gateway downtime/command failure/manual PR merge) · NFR-3 (materialization must be idempotent) · NFR-4 (gate decisions replay-safe and auditable).

**Security and Safety:** NFR-5 (reject events failing signature/schema/replay/binding/provider/authorization checks) · NFR-6 (scope event secrets to correct profile) · NFR-7 (prevent workflow actions outside Bound Project Cwd) · NFR-8 (redact secrets in command logs, event logs, diagnostics, timeline views).

**Auditability:** NFR-9 (persist workflow commands, events, reconciliation actions, gate decisions, state transitions) · NFR-10 (Story Timeline sufficient to understand why a story changed state) · NFR-11 (state changes carry enough provenance to distinguish BMAD/provider/GitHub/Hermes/workflow event/human decision sources).

**Usability:** NFR-12 (phrase next actions in user-facing language, not backend state names) · NFR-13 (distinguish Done Verification Gate approval from GitHub PR merge state) · NFR-14 (surface blocking issues with recovery options, not raw stack traces).

**Maintainability:** NFR-15 (preserve bounded ownership between Hermes/BMAD/providers/GitHub) · NFR-16 (new provider integration surfaces stay generic) · NFR-17 (cross-project handoffs complete enough for isolated agents).

## UX (existing surfaces reused — no new UI stories)

Per explicit product-scope decision (2026-07-02): UI/frontend work for this feature is **out of scope**. The existing Hermes dashboard shell, Kanban plugin, task drawer, and comment/attachment surfaces are reused as-is; new data (Project Binding state, gate evidence, Story Timeline entries, diagnostics) surfaces through those existing generic components via the "display state" fields each backend story already populates. No dedicated UI/component story is needed. If this decision changes, the parent workspace's UX delta contract (`ux-workflow-commander-delta-2026-06-27.md`) documents the intended surfaces (Project Binding selector, gate evidence panel, Story Timeline, diagnostics panel) and uses generic "workflow provider"/"workflow event" vocabulary consistent with this PRD.

## Non-Goals (hermes-agent-relevant subset)

- Will not replace BMAD, Archon, GitHub, or Hermes Kanban with a single monolithic workflow database.
- Will not require the user to operate Archon's dashboard for normal workflow control.
- Will not use Archon HTTP APIs for the Hermes-to-Archon control path.
- Will not treat `sprint-status.yaml`, GitHub Issues, or Archon UI as Hermes runtime queue state.
- Will not rely on global `~/.hermes/skills` as the primary BMAD mount mechanism.
- Will not auto-approve HILT Gates without explicit persisted policy and evidence requirements.

## Glossary (hermes-agent-relevant terms)

Bound Project Cwd, Controller Identity, Done Verification Gate, HILT Gate, Materialization, Phase Task, Product Work Dashboard, Project Binding, Project Work Item, Reconciliation, Story Timeline — full definitions in parent PRD §3; reproduced in local `architecture.md`'s Consistency Conventions table.

## Cross-Project Dependencies

`hermes-agent`'s consumer/adapter stories depend on Archon's producer stories (see `Archon/_bmad-output/planning-artifacts/epics.md`). Story-level dependency records use:

```text
Depends on: <subproject> Story <id or title>
Contract needed: <API/event/file/interface/schema>
Blocking behavior: <what must exist before this story can be completed>
Integration validation: <how both sides will be proven compatible>
```

## Source

Derived from the parent workspace's `prds/prd-workflow-engine-2026-06-26/prd.md` and `epics.md`, both current as of 2026-07-02 (post `bmad-correct-course` pass — see parent `sprint-change-proposal-2026-07-02.md`).
