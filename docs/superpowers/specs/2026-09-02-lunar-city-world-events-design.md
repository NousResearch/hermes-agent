# Lunar City World Events and Dispatcher Companion Design

**Status:** Approved for written-spec review

**Date:** 2026-09-02

## Goal

Make Lunar City a 1:1, in-world interface for Hermes Desktop and Kanban. Real
Hermes profiles, bot groups, agents, tasks, pull requests, reviews, worker
health, notifications, approvals, and releases remain the source of truth;
the city translates those facts into understandable world events, dialogue,
activities, and animations without creating a second operational system.

This spec covers the event translation layer and the dispatcher companion cube.
It is intentionally one bounded sub-project of the larger game effort. The
full 3D asset rebuild, complete animation library, voice stack, and every
Hermes data adapter remain follow-on implementation slices that consume the
interfaces defined here.

## Product principles

1. **Hermes is authoritative.** A world event may visualize a Hermes state or
   transition, but it may not invent a task, worker result, approval, or
   operational status.
2. **Actions are real.** Extinguishing a fire, repairing a station, assigning a
   worker, approving a review, or opening a mission invokes the existing Hermes
   or Kanban action and its existing permission/approval boundary.
3. **The city stays open.** Context, progress, alerts, task details, and agent
   conversation are presented as in-world trays, speech bubbles, overlays, and
   menus. The feature must not open a second OS window or require leaving the
   world to understand an alert.
4. **No compute while closed.** Lunar City has no background simulation loop or
   renderer. On reopen it reconciles the current Hermes snapshot and event
   cursor, then optionally presents a bounded transition recap.
5. **Unknown future data is safe.** New Hermes event kinds, missing optional
   fields, and unavailable animation assets fall back to a visible generic
   state rather than crashing or silently disappearing.
6. **Visual drama never obscures truth.** Fire, earthquakes, panic, celebration,
   debris, and comic reactions are metaphors with a visible source label and
   severity. They never replace the real task title, status, error, owner, or
   required action.
7. **NPC richness is a presentation layer.** Workers and leaders should feel
   closer to a high-quality open-world or cinematic RPG NPC than a marker
   sliding between coordinates: they need readable routines, body language,
   contextual reactions, social moments, and distinct personalities. Those
   behaviors may make the city feel alive, but they cannot fabricate Hermes
   work, results, decisions, or conversations.

## Architecture

```text
Hermes/Kanban REST + existing event streams + agent notices
                         |
                         v
              Source adapters and normalizer
                         |
                         v
                 Semantic world events
                         |
          +--------------+---------------+
          |                              |
          v                              v
   World projection/state          Action intents
          |                              |
          v                              v
    Scene + animation resolver    Existing Hermes/Kanban writes
          |                              |
          +--------------+---------------+
                         v
              In-world dialogue and HUD
```

The renderer consumes a projection, not raw backend payloads. The normalizer
owns compatibility with changing Hermes payloads. The action layer owns
translation from a clicked world affordance into an existing write such as
`patchTask`, `reassignTask`, `reclaimTask`, `addComment`, or the appropriate
desktop gateway action. The projection and event history are renderer-local
presentation state only; they are never an alternate task database.

The first implementation should reuse the existing Kanban plugin seams:

- the board and task detail REST queries for current truth;
- the existing board `/events` socket and its event cursor for transitions;
- the existing terminal event notification mapping for completion, blocker,
  crash, give-up, timeout, and block-loop signals;
- the existing agent notice path for Hermes-wide informational, warning, error,
  and success notices;
- the existing plugin SDK host/navigation/action doors rather than direct
  Electron or OS APIs from the world renderer.

If a future source has no current desktop seam, the adapter must be added at
the narrowest existing plugin or gateway boundary. The world must not pollute
the core agent tool schema to obtain a game-specific signal.

## Semantic event contract

The event compiler works with a stable internal shape. The exact TypeScript
names may change during implementation, but the following fields and meanings
are required:

```ts
interface WorldEvent {
  id: string
  source: 'agent_notice' | 'gateway' | 'kanban' | 'pull_request' | 'system'
  kind: string
  occurredAt: number
  receivedAt: number
  severity: 'info' | 'success' | 'warning' | 'error' | 'critical'
  scope: 'worker' | 'task' | 'building' | 'district' | 'city'
  sourceRef?: { board?: string; taskId?: string; agentId?: string; prId?: string }
  title: string
  detail?: string
  facts: Record<string, unknown>
  actionKinds: string[]
}
```

`kind` is a stable semantic identifier, not a presentation name. Examples are
`task.blocked`, `worker.crashed`, `pr.review_findings`, `pr.merged_stable`,
`gateway.disconnected`, and `credits.depleted`. A source adapter may map
multiple backend spellings to one kind. A source kind that is not recognized
becomes `system.unclassified_alert` and preserves its original label and
payload summary as facts for inspection.

Every event is classified as either:

- **Transition:** a new event ID or notice key that can trigger a one-shot
  animation, conversation, or sound.
- **Condition:** current state derived from a board/task/agent snapshot that
  keeps an ambient effect active, such as smoke on a still-blocked task or an
  idle worker for a crashed process.

The compiler must deduplicate transitions by source identity and event ID. A
socket reconnect, board switch, or reopen must not repeatedly trigger the same
celebration or catastrophe.

## Event coverage and world translation

The following table defines the initial semantic vocabulary. It is a baseline,
not a closed list; new event kinds must use the same contract and fallback.

| Real Hermes/Kanban condition | Semantic event | World expression | Required context/action |
| --- | --- | --- | --- |
| Card becomes blocked | `task.blocked` | Local fire, smoke, warning lights, workers responding | Reason, task, owner, inspect blocker, comment, reassign/reclaim if supported |
| Block loop detected | `task.block_loop` | Fire spreads through connected routes; emergency siren | Human handoff, dependency chain, exact recovery actions |
| Worker crashes or is dead beyond the liveness threshold | `worker.crashed` | Damaged station, stopped worker, repair crew | Worker/task, last heartbeat, reclaim/retry action |
| Worker gives up | `worker.gave_up` | Abandoned worksite and red beacon | Error/result, retry/reassign action |
| Worker times out | `worker.timed_out` | Frozen clock, stalled transport, queued responders | Run timing and task log, retry/reclaim action |
| Task is running | `task.running` | Assigned worker travels and performs task-specific work | Profile, elapsed time, live summary/log when available |
| Task is waiting/scheduled/ready | `task.waiting` or `task.ready` | Queue, staging area, or worker preparing tools | Why it is waiting and expected next transition |
| Task enters review | `task.in_review` | Inspector arrives and marks the building for review | Review context, open task, comment/approval action |
| PR has review findings or regressions | `pr.review_findings` | Structural damage, falling debris, frustrated leader, repair crew | Findings count/severity, PR links, review comments |
| Merge conflict | `pr.merge_conflict` | Two construction crews collide at one structure | Conflicting branches/files and the real resolution route |
| PR approved | `pr.approved` | Green inspection lights and deployment preparation | Reviewers, PR, merge action if authorized |
| PR merges to draft | `pr.merged_draft` | District-level construction milestone | Commit/PR identity and resulting change |
| PR merges to stable | `pr.merged_stable` | City-wide celebration, lights, music, fireworks | Exact PR/commit and stable branch; celebration is one-shot |
| Release/deployment succeeds | `release.succeeded` | New infrastructure comes online | Release identity and destination |
| Release/deployment fails | `release.failed` | Launch malfunction or district blackout | Failure details and recovery action |
| Gateway or provider disconnects | `gateway.disconnected` | Communications blackout and signal interference | Connection/profile, reconnect action, last known state |
| Authentication/credential failure | `auth.failed` | Locked checkpoint and access denied indicator | Safe remediation route; never display secrets |
| Approval or security gate required | `approval.required` | Locked checkpoint awaiting the operator | Exact approval request and approve/deny action |
| Agent notice is warning/error | `agent.warning` or `agent.error` | Scope determined by notice severity; visible alert marker | Original notice text and source |
| Credits depleted or service paused | `credits.depleted` | Power rationing and paused work queues | Account-level notice and existing settings/action route |
| Task completes | `task.completed` | Construction finishes; resources move; workers celebrate | Summary, artifacts, links, follow-up action |
| Task is unblocked/resolved | `task.recovered` | Workers extinguish the fire and restore the route | Original blocker, new status, source event |
| Task is archived/deleted | `task.removed` | Worksite is safely decommissioned | Identity and reason when available |

The event compiler chooses the visual metaphor from real facts. For example,
an old critical blocker can escalate from a localized fire to a district
emergency, while a single warning remains a contained alert. This is a visual
severity ramp, not a new Kanban severity field and not permission to mutate
tasks automatically.

## Agents, leaders, and conversations

Every visible worker and leader is projected from a real Hermes profile, group,
or agent identity. The projection includes a stable identity key so a refresh
does not make a worker appear to teleport to a different task.

Workers and leaders have semantic activity states such as `idle`, `walking`,
`working`, `carrying`, `inspecting`, `repairing`, `talking`, `waiting`,
`panicking`, `celebrating`, `resting`, and `returning`. The resolver chooses an
animation by state plus task/event kind. If a specialized clip is missing, it
uses the generic state animation and keeps the semantic label visible.

The target feel is a living open-world/cinematic-RPG cast:

- **Readable daily staging:** workers visibly leave a group, travel along the
  real task route, gather at the relevant station, perform the task activity,
  and return when the source state says they are finished or returning. Idle
  loops, tool handling, waiting, and looking toward active incidents prevent
  the city from feeling like pieces placed on a board.
- **Contextual reactions:** nearby agents can look toward a fire, make space
  for a repair crew, join a celebration, queue at a blocked route, or seek the
  dispatcher when the normalized event scope warrants it. These reactions are
  deterministic scene choreography derived from event scope and proximity,
  not hidden task execution.
- **Distinct identity:** leaders and workers use role, group, profile, and
  configured identity metadata to choose a consistent silhouette, color
  language, idle style, and reaction set. No invented personality trait may be
  presented as a fact about the real Hermes profile.
- **Worker-class personalities:** each worker class gets a stable presentation
  personality profile—such as methodical, curious, protective, social,
  cautious, or bold—derived from its role metadata and deterministic identity
  seed. This profile selects idle loops, gestures, reactions, and delivery
  style; it does not change Hermes behavior or assert private psychological
  facts about an agent.
- **Cinematic beats:** important transitions can use short camera-friendly
  performances—inspection, argument over a merge conflict, emergency repair,
  relief after recovery, or a stable-merge celebration—while keeping the
  source badge and dialogue available. Camera composition must never remove
  operator access to the real details or controls.
- **Social staging:** agents may face each other, gesture, exchange grounded
  dialogue bubbles, form small groups, and disperse. The event compiler may
  choose the participants from the real assignee, reviewer, leader, dependent
  tasks, and nearby workers. If no source text exists, it uses nonverbal
  performance or an honest generic line instead of pretending the agents said
  something specific.

This is a bounded NPC presentation system, not a second simulation. It should
not run an autonomous decision-making loop, invent relationships, or consume
compute while the world is closed. Ambient behavior is scheduled from the
current projection and event transitions; every operationally meaningful
change remains a Hermes event or a user-approved Hermes action.

Agent-to-agent speech is grounded in available Hermes context: task title,
latest summary, event payload, comments, worker log excerpts, profile role,
and current status. The world may stage these facts as dialogue bubbles, but it
must not fabricate a successful result, a conversation, or an agent decision
that Hermes did not provide. When no message content exists, the scene uses
nonverbal activity and a truthful label such as “working on task.”

Clicking a worker, leader, building, route, crisis, or task opens an in-world
dialogue tray containing:

- identity and role;
- current Hermes/Kanban status;
- task, PR, or event title;
- latest available summary/error/comment;
- elapsed time and liveness where available;
- source link/identifier;
- only the actions that the source and permission model allow.

Voice input, when supported by the existing desktop voice path, is converted
to the same intent/action contract as typed input. It does not create a second
agent loop.

## Dispatcher companion cube

The dispatcher leader is a small companion cube located at the city command
center. It is the operator’s primary conversational entry point and the visual
equivalent of a “new task / new session” surface.

The cube has four responsibilities:

1. **Dispatch:** start a new Hermes chat/session using the existing desktop
   session creation path, with the selected profile/group context when the
   operator chooses one.
2. **Task creation:** collect title, body, board, assignee, priority, workspace,
   and other supported fields, then call the existing Kanban create-task flow.
3. **Situation report:** summarize active conditions, recent transitions,
   blocked work, broken workers, review queues, and required approvals inside a
   dialogue panel.
4. **Navigation and explanation:** bring the operator to a selected agent,
   task, group, or event in the world and explain what Hermes is doing without
   opening another window.

The cube is not a synthetic Hermes profile and does not execute work itself.
Its identity is presentation-only. Every command is represented as a pending
intent while input is collected, then as success, rejection, approval-needed,
or failure based on the real Hermes response. The cube should visibly say when
the backend is disconnected, when an action is unavailable, or when the
operator must approve something.

The cube’s default interaction is a dialogue box with quick actions such as
“new task,” “new session,” “show blocked work,” “show broken workers,” “show
recent merges,” and “what needs my attention?” These are shortcuts into real
queries/actions, not a parallel menu model. It must remain usable with the
world disabled; the standard Hermes routes remain the fallback surface.

## Event lifecycle and zero-compute behavior

While the world route is open, it subscribes to the existing event sources and
updates its projection through normal renderer state. It must unsubscribe when
the route/plugin is disabled or unmounted.

The persisted world metadata is limited to device-scoped presentation state:

- last opened timestamp;
- last acknowledged semantic event cursor per source/board;
- chosen camera/focus and onboarding state;
- dismissed cosmetic recap IDs, never task truth.

When reopening:

1. Read the current Hermes/Kanban snapshot.
2. Read the latest available event cursor from the existing source.
3. Reconcile current conditions against the saved cursor.
4. Present at most a bounded recent-event recap, ordered by source timestamp.
5. Mark only successfully projected source IDs as seen.

No timer, worker, simulation process, polling loop, or animation loop runs while
the world route and its event subscriptions are closed. An animation may play
after reopen as a deterministic visual recap, but it must not claim that work
progressed while the app was closed unless the Hermes source reports that
progress.

## Extensibility and upgrade compatibility

The semantic event registry is versioned independently of presentation assets.
Each entry declares:

- source kinds and required facts;
- condition/transition classification;
- severity and scope resolver;
- default scene/animation tags;
- supported action kinds;
- dialogue fields;
- fallback behavior.

Presentation packs use tags such as `task.blocked`, `repair`, `pr.merge.success`,
and `celebration.citywide`, rather than importing backend-specific event names.
Animation packs can therefore improve independently. A missing clip falls back
to a generic state clip; a missing scene falls back to an alert marker and
dialogue; a changed payload field falls back to the raw source label plus the
facts that remain available.

The registry must not rely on exhaustive string unions from the backend. New
backend status columns and event kinds should continue to render through
unknown-status/unknown-event fallbacks. Tests must prove that adding an
unrecognized status or payload field does not make the route fail.

## Error handling and truth boundaries

- **Source unavailable:** show a communications/interference state and retain
  the last known snapshot with an explicit stale indicator.
- **Write rejected:** stop the corresponding repair/celebration action, show the
  real rejection reason, and leave the world condition unchanged.
- **Approval required:** show the approval checkpoint and use the existing
  approval surface/door; never bypass it from a game click.
- **Malformed event:** render `system.unclassified_alert`, preserve a safe
  human-readable summary, and log diagnostic details through existing desktop
  logging only.
- **Duplicate/replayed event:** suppress duplicate one-shot effects by source
  and event ID while still allowing current conditions to render.
- **Sensitive payload:** redact credentials, tokens, and secret values before
  dialogue, scene facts, or logs are rendered.
- **Cosmetic failure:** keep the task/event dialogue and status visible even if
  a mesh, texture, animation, sound, or shader fails to load.

## Testing strategy

The implementation should be split so the most important truth rules are
testable without Blender, a running renderer, or a live model:

1. **Normalizer tests:** map known Kanban terminal events, task statuses,
   liveness fields, agent notices, and PR/release inputs to semantic events;
   preserve IDs, timestamps, facts, severity, and action kinds.
2. **Compatibility tests:** unknown kinds, unknown statuses, missing optional
   fields, malformed payloads, and future fields produce safe fallbacks.
3. **Deduplication/replay tests:** reconnects, reopen reconciliation, board
   switching, and repeated frames never replay a transition twice.
4. **Severity/scope tests:** verify deterministic local/building/district/city
   escalation from real facts and stable event IDs.
5. **Action contract tests:** every exposed world action calls the same existing
   Hermes/Kanban action seam; rejected, approval-required, and disconnected
   responses leave the projection truthful.
6. **Dispatcher cube tests:** new task, new session, situation report, and
   selected-entity explanation remain inside the route and use the real action
   interfaces.
7. **Projection tests:** stable identities keep workers attached to the same
   profile/group/task, and missing animation tags select generic fallbacks.
8. **Lifecycle tests:** event subscriptions are disposed when the route is
   disabled/unmounted; no background timer or process is created for the world.
9. **Renderer smoke test:** a small fixture containing a blocked task, crashed
   worker, successful stable merge, review findings, and an unknown event
   produces visible scenes and truthful dialogue without a second window.

Acceptance requires both the existing standard Hermes/Kanban surface and Lunar
City to show the same source identifiers and current statuses after each test
action. A visually impressive scene with mismatched source state is a failure.

## Initial delivery boundaries

The first implementation plan should deliver the event compiler, projection
types/store, initial Kanban and agent-notice adapters, dispatcher cube action
contracts, in-world dialogue tray, deterministic fallback scenes, and focused
tests. It should not attempt to finish every high-fidelity building, character,
or animation asset in the same change. Asset packs can then be added against
the stable semantic tags and verified with the same fixture events.
