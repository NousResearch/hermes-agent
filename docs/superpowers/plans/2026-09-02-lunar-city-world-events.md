# Lunar City World Events and Dispatcher Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truthful Lunar City event layer and dispatcher companion that turns live Hermes/Kanban state into inspectable world scenes, NPC activity, dialogue, and real operator actions.

**Architecture:** Add pure normalizers and a versioned presentation registry between existing Hermes/Kanban sources and the Lunar City renderer. Keep backend data authoritative, store only device-scoped presentation cursors, and route every mutation through existing Kanban or gateway action seams. The dispatcher cube and dialogue tray are in-world React surfaces; they do not create a second agent loop or OS window.

**Tech Stack:** TypeScript, React, nanostores, React Query through `@hermes/plugin-sdk`, existing Kanban REST/event socket, existing gateway event handling, Vitest, Testing Library, Tailwind utility classes, and the current Vite-relative asset build.

**Spec:** `docs/superpowers/specs/2026-09-02-lunar-city-world-events-design.md`

## Global Constraints

- Hermes/Kanban remains the sole source of truth for profiles, groups, agents, tasks, PRs, approvals, workers, and releases.
- World actions call existing Hermes/Kanban writes and preserve permission, approval, rejection, and rollback behavior.
- No timer, worker, simulation process, polling loop, or animation loop runs while the world route is closed.
- Unknown event kinds, status values, missing fields, and missing animation assets render safe fallbacks and never crash the route.
- Background events must not navigate, steal focus, or open a second OS window.
- Persisted values are device-scoped presentation metadata only; task truth and operational history are never stored as a local duplicate.
- Existing uncommitted Lunar City onboarding/settings changes remain user-owned WIP and must be preserved while these modules are added.
- No new core model tool, backend simulation loop, or game-specific `HERMES_*` environment variable is introduced.
- Every task ends with focused tests and a small commit containing only that task’s files.

---

## File map

Create the following focused modules:

- `apps/desktop/src/app/lunar-city/world-events.ts` — stable semantic event, condition, source reference, and normalized payload contracts plus pure source normalizers.
- `apps/desktop/src/app/lunar-city/world-events.test.ts` — table-driven normalizer, fallback, severity, and deduplication tests.
- `apps/desktop/src/app/lunar-city/world-presentation.ts` — event-to-scene, NPC activity, worker-class personality, participant, animation-tag, and deterministic severity resolution.
- `apps/desktop/src/app/lunar-city/world-presentation.test.ts` — deterministic presentation and missing-asset fallback tests.
- `apps/desktop/src/app/lunar-city/world-sync.ts` — source fan-in, snapshot reconciliation, cursor handling, lifecycle binding, and bounded reopen recap.
- `apps/desktop/src/app/lunar-city/world-sync.test.ts` — source subscription disposal, replay, reconnect, board scope, and stale-source tests.
- `apps/desktop/src/app/lunar-city/world-actions.ts` — typed action-intent facade over existing Kanban and gateway actions.
- `apps/desktop/src/app/lunar-city/world-actions.test.ts` — success, rejection, approval-required, and disconnected action tests.
- `apps/desktop/src/app/lunar-city/dispatcher-cube.tsx` — command-center companion UI for new task, new session, situation reports, and selected context.
- `apps/desktop/src/app/lunar-city/dispatcher-cube.test.tsx` — in-world dispatcher interaction tests.
- `apps/desktop/src/app/lunar-city/dialogue-tray.tsx` — shared in-world detail, dialogue, source, and action panel.
- `apps/desktop/src/app/lunar-city/dialogue-tray.test.tsx` — truthful context and available-action rendering tests.
- `apps/desktop/src/app/lunar-city/world-scene.tsx` — semantic world projection renderer, NPC staging, event markers, and animation-tag hooks.
- `apps/desktop/src/app/lunar-city/world-scene.test.tsx` — fixture scene rendering and generic asset fallback tests.

Modify the following existing seams:

- `apps/desktop/src/store/lunar-city.ts` — retain existing device preferences and add sanitized device-scoped world cursor/focus/recap state plus runtime projection atoms.
- `apps/desktop/src/store/agent-notices.ts` — expose a renderer-local notice observer without changing the existing toast/native-notification contract.
- `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/status.ts` — publish accepted `notification.show`/`notification.clear` inputs to the world observer after normal notice handling.
- `apps/desktop/src/plugins/kanban/completion-notify.ts` — expose the existing deduplicated Kanban event stream to the world source adapter without adding a second socket or process.
- `apps/desktop/src/app/lunar-city/index.tsx` — replace the onboarding-only body with the source-bound world shell while preserving current first-open asset setup and disabled-world behavior.
- `apps/desktop/src/app/lunar-city/index.test.tsx` — extend existing onboarding tests with world lifecycle, dispatcher, event, and action fixtures.
- `apps/desktop/src/app/contrib/surfaces.tsx` — pass the existing desktop gateway/action context into Lunar City only if the current route contribution cannot obtain it through the established renderer seam; do not create a second gateway client.

The current route/sidebar/settings changes from the earlier Lunar City slice are not rewritten. The executor should integrate with them and verify their existing tests still pass.

## Task 1: Define semantic events and pure normalizers

**Files:**
- Create: `apps/desktop/src/app/lunar-city/world-events.ts`
- Create: `apps/desktop/src/app/lunar-city/world-events.test.ts`
- Read: `apps/desktop/src/plugins/kanban/types.ts`
- Read: `apps/desktop/src/plugins/kanban/completion-notify.ts`
- Read: `apps/desktop/src/store/agent-notices.ts`

**Interfaces:**
- Consumes: `CompletionEvent`, `KanbanTask`, `KanbanTaskFull`, `AgentNoticePayload`.
- Produces: `WorldEvent`, `WorldCondition`, `WorldSourceRef`, `WorldSeverity`, `WorldScope`, `WorldActionKind`, `normalizeKanbanEvent`, `normalizeAgentNotice`, `normalizeExternalEvent`, `classifyTaskCondition`, `dedupeWorldEvents`, and `worldEventId`.

- [ ] **Step 1: Write failing table-driven tests for known event normalization.**

```ts
it.each([
  ['blocked', 'task.blocked', 'warning', 'task'],
  ['crashed', 'worker.crashed', 'error', 'worker'],
  ['gave_up', 'worker.gave_up', 'error', 'task'],
  ['timed_out', 'worker.timed_out', 'warning', 'task'],
  ['block_loop_detected', 'task.block_loop', 'critical', 'district'],
  ['completed', 'task.completed', 'success', 'task']
])('normalizes %s to %s', (kind, expectedKind, severity, scope) => {
  const event = normalizeKanbanEvent('main', {
    id: 41,
    task_id: 'task-7',
    kind,
    payload: { reason: 'dependency failed', summary: 'done' }
  })

  expect(event).toMatchObject({
    kind: expectedKind,
    severity,
    scope,
    sourceRef: { board: 'main', taskId: 'task-7' }
  })
})
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `npm run test:ui -- src/app/lunar-city/world-events.test.ts`

Expected: FAIL because `world-events.ts` and its normalizers do not exist.

- [ ] **Step 3: Implement the stable contract and normalizers.**

Use string-based `kind` and `status` handling so backend additions do not break TypeScript exhaustiveness. Preserve safe facts, source IDs, timestamps, original labels, and redacted summaries. Map unknown input to `system.unclassified_alert` with `warning` severity and `city` scope. Treat a Kanban event as a transition and a task snapshot as a condition.

```ts
export type WorldActionKind =
  | 'inspect'
  | 'inspect_blocker'
  | 'comment'
  | 'recover_task'
  | 'reassign_task'
  | 'reclaim_task'
  | 'create_task'
  | 'create_session'
  | 'request_approval'
  | 'show_source'

export interface ExternalWorldEventInput {
  source: 'gateway' | 'pull_request' | 'system'
  id: string
  kind: string
  occurredAt?: number
  title: string
  detail?: string
  severity?: WorldSeverity
  scope?: WorldScope
  sourceRef?: WorldSourceRef
  facts?: Record<string, unknown>
}

export interface WorldEvent {
  id: string
  source: WorldSource
  kind: string
  occurredAt: number
  receivedAt: number
  severity: WorldSeverity
  scope: WorldScope
  sourceRef?: WorldSourceRef
  title: string
  detail?: string
  facts: Record<string, unknown>
  actionKinds: WorldActionKind[]
  transition: true
}

export function normalizeKanbanEvent(board: string, event: CompletionEvent, now = Date.now()): WorldEvent
export function normalizeAgentNotice(payload: AgentNoticePayload, now = Date.now()): WorldEvent | null
export function normalizeExternalEvent(input: ExternalWorldEventInput, now = Date.now()): WorldEvent
export function classifyTaskCondition(task: KanbanTask, now = Date.now()): WorldCondition | null
export function dedupeWorldEvents(previous: readonly WorldEvent[], incoming: readonly WorldEvent[]): WorldEvent[]
```

- [ ] **Step 4: Add tests for unknowns, notice levels, liveness, and deduplication.**

Assert that empty notices return `null`, warning/error/success notice levels map correctly, a stale running task becomes a worker condition only when its heartbeat exceeds the existing liveness threshold, unknown event kinds preserve a safe source label, and duplicate `(source, id)` pairs occur once while distinct boards remain distinct.

- [ ] **Step 5: Run the focused test and commit the contract.**

Run: `npm run test:ui -- src/app/lunar-city/world-events.test.ts`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/world-events.ts apps/desktop/src/app/lunar-city/world-events.test.ts
git commit -m "feat(desktop): add lunar city semantic events"
```

## Task 2: Add deterministic scenes, NPC activities, and animation tags

**Files:**
- Create: `apps/desktop/src/app/lunar-city/world-presentation.ts`
- Create: `apps/desktop/src/app/lunar-city/world-presentation.test.ts`
- Modify: `apps/desktop/src/app/lunar-city/world-events.ts` only if a shared type is required by the resolver

**Interfaces:**
- Consumes: `WorldEvent`, `WorldCondition`, `WorldSourceRef`, `WorldActionKind`.
- Produces: `WorldPresentation`, `NpcActivity`, `resolveWorldPresentation`, `resolveNpcActivity`, `stableEventSeed`.

- [ ] **Step 1: Write failing tests for event scenes and deterministic NPC activity.**

```ts
it('turns a blocked card into a repairable local crisis', () => {
  const presentation = resolveWorldPresentation(blockedEvent('task-7'), [])

  expect(presentation).toMatchObject({
    sceneTag: 'crisis.fire.local',
    scope: 'task',
    animationTags: expect.arrayContaining(['task.blocked', 'repair', 'extinguish'])
  })
  expect(presentation.actionKinds).toContain('inspect_blocker')
})

it('keeps the same participants and choreography for the same source event', () => {
  const event = mergeEvent('pr-9')

  expect(resolveWorldPresentation(event, [])).toEqual(resolveWorldPresentation(event, []))
})
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `npm run test:ui -- src/app/lunar-city/world-presentation.test.ts`

Expected: FAIL because the presentation resolver does not exist.

- [ ] **Step 3: Implement the registry and resolver.**

Create a data-driven registry keyed by semantic event names. Include the approved initial mappings for blocked, block loop, crashed, gave up, timed out, running, waiting, review, review findings, merge conflict, approved, draft merge, stable merge, release success/failure, gateway disconnect, auth failure, approval required, agent notices, credits depletion, completion, recovery, and removal. Resolve visual severity/scope from facts, using a stable hash of `source + id` for participant selection and cosmetic variation.

```ts
export interface WorldPresentation {
  sceneTag: string
  scope: WorldScope
  animationTags: string[]
  npcActivities: NpcActivity[]
  participants: WorldSourceRef[]
  actionKinds: WorldActionKind[]
  cosmetic: { soundTag?: string; cameraBeat?: string; intensity: 0 | 1 | 2 | 3 }
}

export function resolveWorldPresentation(
  event: WorldEvent,
  conditions: readonly WorldCondition[],
  assetTags: ReadonlySet<string> = new Set()
): WorldPresentation
```

Unknown events must resolve to `scene.alert.unclassified`, `alert`, and a readable dialogue marker. Missing specialized tags must retain the semantic tag and append `fallback.generic` rather than failing.

- [ ] **Step 4: Add tests for severity escalation, NPC-style behavior, and worker-class personality.**

Cover local fire versus district catastrophe, city-wide stable-merge celebration, contextual nearby reactions, task-specific states such as `inspecting`, `repairing`, `carrying`, `talking`, `panicking`, and `celebrating`, and deterministic “comic celebration” as a cosmetic flourish that never changes source facts or status. Verify role metadata maps research, review, operations/security, release, and social/support classes to stable presentation personalities, while unknown classes use an identity-seeded fallback.

- [ ] **Step 5: Run the focused test and commit the presentation registry.**

Run: `npm run test:ui -- src/app/lunar-city/world-presentation.test.ts`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/world-presentation.ts apps/desktop/src/app/lunar-city/world-presentation.test.ts apps/desktop/src/app/lunar-city/world-events.ts
git commit -m "feat(desktop): map lunar city events to scenes"
```

## Task 3: Build source fan-in, reconciliation, and zero-compute lifecycle

**Files:**
- Create: `apps/desktop/src/app/lunar-city/world-sync.ts`
- Create: `apps/desktop/src/app/lunar-city/world-sync.test.ts`
- Modify: `apps/desktop/src/store/lunar-city.ts`
- Modify: `apps/desktop/src/plugins/kanban/completion-notify.ts`
- Modify: `apps/desktop/src/store/agent-notices.ts`
- Modify: `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/status.ts`

**Interfaces:**
- Consumes: `fetchBoard`, `fetchTask`, the existing Kanban event stream, `AgentNoticePayload`, `ExternalWorldEventInput`, `normalizeKanbanEvent`, `normalizeAgentNotice`, `normalizeExternalEvent`, and `classifyTaskCondition`.
- Produces: `WorldSourceDoors`, `WorldProjection`, `bindWorldSources`, `reconcileWorldSnapshot`, `recordWorldCursor`, `worldSourceScopeKey`.

- [ ] **Step 1: Write failing lifecycle and cursor tests.**

```ts
it('disposes every source when the world closes', () => {
  const closeKanban = vi.fn()
  const closeGateway = vi.fn()
  const dispose = bindWorldSources({ kanban: on => (kanbanListener = on, closeKanban), gateway: on => (gatewayListener = on, closeGateway) }, sink)

  dispose()

  expect(closeKanban).toHaveBeenCalledOnce()
  expect(closeGateway).toHaveBeenCalledOnce()
})

it('does not replay an event already acknowledged for the same board', () => {
  const result = reconcileWorldSnapshot(snapshot, [eventWithId('main', '41')], { 'kanban:main': '41' })

  expect(result.transitions).toEqual([])
})
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `npm run test:ui -- src/app/lunar-city/world-sync.test.ts`

Expected: FAIL because the lifecycle and reconciliation functions do not exist.

- [ ] **Step 3: Extend the store with sanitized presentation state.**

Keep `$worldEnabled` and `$worldOnboardingDismissed` unchanged. Add a JSON codec that accepts only finite timestamps, bounded cursor strings, valid source scopes, and bounded recap IDs. Persist keys with explicit device scope, for example `hermes.desktop.world.cursors.v1`; keep the live projection and transition queue in non-persisted atoms.

```ts
export interface WorldCursorState {
  bySource: Record<string, string>
  lastOpenedAt: number | null
  dismissedRecapIds: string[]
}

export interface WorldProjection {
  conditions: WorldCondition[]
  recentEvents: WorldEvent[]
  transitions: WorldEvent[]
  stale: boolean
  sourceError: string | null
}

export const $worldCursors: WritableAtom<WorldCursorState>
export const $worldProjection: WritableAtom<WorldProjection>
```

- [ ] **Step 4: Add source fan-in without creating new sockets.**

Expose a narrowly scoped listener from `completion-notify.ts` that receives only new event IDs already accepted by its existing per-board cursor. Add a renderer-local notice subscription in `agent-notices.ts`; invoke it from the existing gateway status handler after toast/native notification handling. `bindWorldSources` must accept these doors, register them only while Lunar City is mounted, and return one disposer.

Add an external-event door for PR/release/gateway/system adapters using the shared `ExternalWorldEventInput` shape. Connect it only to event sources already available to the desktop renderer; do not invent PR polling or a second WebSocket. When a source is unavailable, the world exposes a stale/unavailable condition and does not synthesize a PR or release result.

- [ ] **Step 5: Implement reopen reconciliation.**

Read current board/task snapshots through existing React Query fetchers, derive persistent conditions, compare source event IDs against the persisted cursor, sort a bounded recent recap by `occurredAt`, and write cursors only after events have been normalized successfully. Reconcile external PR/release/system inputs through the same path when their existing desktop source is available. Use board/source scope keys so switching boards never mixes history. A source error keeps the last projection with `stale: true` and an explicit reason.

- [ ] **Step 6: Test zero-compute behavior and commit.**

Add assertions that no `setInterval`, `setTimeout`, worker, process, second socket, navigation, or OS window call is made by `bindWorldSources`, and that disposal removes all listeners. Test reconnect replay, current conditions after a long closed interval, malformed events, and board switching.

Run: `npm run test:ui -- src/app/lunar-city/world-sync.test.ts src/store/agent-notices.test.ts`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/world-sync.ts apps/desktop/src/app/lunar-city/world-sync.test.ts apps/desktop/src/store/lunar-city.ts apps/desktop/src/plugins/kanban/completion-notify.ts apps/desktop/src/store/agent-notices.ts apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/status.ts
git commit -m "feat(desktop): sync lunar city with Hermes events"
```

## Task 4: Add the truthful world action facade

**Files:**
- Create: `apps/desktop/src/app/lunar-city/world-actions.ts`
- Create: `apps/desktop/src/app/lunar-city/world-actions.test.ts`
- Modify: `apps/desktop/src/plugins/kanban/api.ts` only when an existing exported write is insufficient for an already-supported Kanban action

**Interfaces:**
- Consumes: existing Kanban `patchTask`, `createTask`, `addComment`, `reassignTask`, `reclaimTask`, and the existing desktop `requestGateway`/session action seam.
- Produces: `WorldActionContext`, `createWorldActionRunner`, `runWorldAction`.

- [ ] **Step 1: Write failing tests for action routing and failure honesty.**

```ts
it('maps extinguish to the existing recovery action without inventing a game mutation', async () => {
  const patchTask = vi.fn().mockResolvedValue({})
  const runner = createWorldActionRunner({ kanban: { patchTask } })

  await runner.run({ kind: 'recover_task', taskId: 'task-7', patch: { status: 'todo' } })

  expect(patchTask).toHaveBeenCalledWith('task-7', { status: 'todo' })
})

it('does not clear a crisis when Hermes rejects the write', async () => {
  const runner = createWorldActionRunner({ kanban: { patchTask: vi.fn().mockRejectedValue(new Error('409 blocked')) } })

  await expect(runner.run({ kind: 'recover_task', taskId: 'task-7', patch: { status: 'todo' } })).rejects.toThrow('409 blocked')
})
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `npm run test:ui -- src/app/lunar-city/world-actions.test.ts`

Expected: FAIL because the action facade does not exist.

- [ ] **Step 3: Implement typed action routing.**

Support only actions exposed by the source context: inspect/open, comment, recover/unblock through the exact existing status write, reassign, reclaim/retry, create task, create session, request approval, and show standard Hermes context. Return a discriminated result `{ ok: true } | { ok: false; kind: 'rejected' | 'approval_required' | 'disconnected'; message: string }`; do not optimistically remove an event or condition.

- [ ] **Step 4: Test dispatcher/session routing against the established owner path.**

Use the same `session.create` payload builder and request gateway ownership rules exercised by `use-session-actions`; the world action must not create a second gateway client or route a profile-scoped request through the ambient wrong profile. Cover disconnected and approval-required responses.

- [ ] **Step 5: Run tests and commit the action facade.**

Run: `npm run test:ui -- src/app/lunar-city/world-actions.test.ts`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/world-actions.ts apps/desktop/src/app/lunar-city/world-actions.test.ts apps/desktop/src/plugins/kanban/api.ts
git commit -m "feat(desktop): route lunar city actions through Hermes"
```

## Task 5: Build the dispatcher cube and dialogue tray

**Files:**
- Create: `apps/desktop/src/app/lunar-city/dispatcher-cube.tsx`
- Create: `apps/desktop/src/app/lunar-city/dispatcher-cube.test.tsx`
- Create: `apps/desktop/src/app/lunar-city/dialogue-tray.tsx`
- Create: `apps/desktop/src/app/lunar-city/dialogue-tray.test.tsx`

**Interfaces:**
- Consumes: `WorldProjection`, `WorldEvent`, `WorldPresentation`, `WorldActionRunner`, existing profile/session/task context.
- Produces: `DispatcherCube`, `DialogueTray`, `DialogueEntry`, and accessible action callbacks that invoke `runWorldAction`.

- [ ] **Step 1: Write failing tests for the command center and truth panel.**

```tsx
it('offers new task, new session, and situation report inside the world', () => {
  render(<DispatcherCube context={fixtureContext()} />)

  expect(screen.getByRole('button', { name: 'New task' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'New session' })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'What needs my attention?' })).toBeTruthy()
})

it('shows source truth and only allowed actions for a blocked task', () => {
  render(<DialogueTray subject={blockedSubject()} />)

  expect(screen.getByText('task.blocked')).toBeTruthy()
  expect(screen.getByText('dependency failed')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Inspect blocker' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Approve review' })).toBeNull()
})
```

- [ ] **Step 2: Run the focused tests to verify they fail.**

Run: `npm run test:ui -- src/app/lunar-city/dispatcher-cube.test.tsx src/app/lunar-city/dialogue-tray.test.tsx`

Expected: FAIL because the components do not exist.

- [ ] **Step 3: Implement the dispatcher cube as an in-world conversation entry point.**

Render the cube with a focused conversation tray and quick actions for new task, new session, blocked work, broken workers, recent merges, and situation report. New task collects the same fields as the current Kanban create dialog. New session uses the existing session action path. Show pending, success, rejected, approval-required, disconnected, and failed states without navigating or opening another window.

- [ ] **Step 4: Implement the dialogue tray with grounded speech bubbles.**

Render identity, role, task/PR/event title, current status, summary/error/comment, liveness, source identifier, and allowed actions. Agent-to-agent dialogue uses only available source text; with no source text it shows nonverbal activity and an honest generic label. Redact token-like values before rendering. Keep keyboard focus inside the tray and provide an explicit close/back action.

- [ ] **Step 5: Test action callbacks and error states, then commit.**

Assert that clicking New task calls `createTask` with the submitted fields, New session calls `session.create` through the provided runner, Inspect blocker opens the source context, rejected actions keep the crisis visible, and no callback invokes navigation/window APIs.

Run: `npm run test:ui -- src/app/lunar-city/dispatcher-cube.test.tsx src/app/lunar-city/dialogue-tray.test.tsx`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/dispatcher-cube.tsx apps/desktop/src/app/lunar-city/dispatcher-cube.test.tsx apps/desktop/src/app/lunar-city/dialogue-tray.tsx apps/desktop/src/app/lunar-city/dialogue-tray.test.tsx
git commit -m "feat(desktop): add lunar city dispatcher companion"
```

## Task 6: Render the semantic world and NPC activity layer

**Files:**
- Create: `apps/desktop/src/app/lunar-city/world-scene.tsx`
- Create: `apps/desktop/src/app/lunar-city/world-scene.test.tsx`
- Modify: `apps/desktop/src/app/lunar-city/index.tsx`
- Modify: `apps/desktop/src/app/lunar-city/index.test.tsx`

**Interfaces:**
- Consumes: `$worldProjection`, `resolveWorldPresentation`, `DispatcherCube`, `DialogueTray`, current world onboarding/settings state.
- Produces: a mounted Lunar City scene with semantic data attributes and accessible in-world controls.

- [ ] **Step 1: Write failing scene fixture tests.**

```tsx
it('renders a blocked task, crashed worker, stable merge, and unknown alert together', () => {
  render(<WorldScene projection={fixtureProjection()} />)

  expect(screen.getByTestId('world-scene-crisis.fire.local')).toBeTruthy()
  expect(screen.getByTestId('world-npc-worker.crashed')).toBeTruthy()
  expect(screen.getByTestId('world-scene-celebration.citywide')).toBeTruthy()
  expect(screen.getByTestId('world-scene-alert.unclassified')).toBeTruthy()
})
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `npm run test:ui -- src/app/lunar-city/world-scene.test.ts src/app/lunar-city/index.test.tsx`

Expected: FAIL because the semantic scene renderer is not present.

- [ ] **Step 3: Implement semantic scene rendering and NPC staging.**

Render scene containers with `data-scene`, `data-source-id`, and `data-animation-tags` attributes. Use CSS class hooks for animation packs and generic fallback hooks for missing tags. Stable identity keys keep leaders/workers attached to the same profile/group/task. Render routes, crisis markers, event badges, and the dispatcher cube as selectable entities. Do not add an autonomous movement timer; while open, use source-driven transition choreography and short CSS/asset animations only.

- [ ] **Step 4: Integrate the source lifecycle into `LunarCity`.**

Keep the current disabled-world screen and first-open onboarding. After onboarding, bind `world-sync` with the existing source doors, render the current projection, show a stale-source banner when necessary, and dispose all subscriptions on unmount or disabled state. Do not remove the existing relative asset path helper or turn review-only uploads into an importer.

- [ ] **Step 5: Test operator-visible states and commit the world shell.**

Cover onboarding, disabled world, loading, connected, stale, disconnected, empty, and populated states. Assert the dispatcher cube remains reachable, selecting a worker opens the dialogue tray, a rejected repair leaves the fire visible, and closing the route disposes source listeners.

Run: `npm run test:ui -- src/app/lunar-city/world-scene.test.ts src/app/lunar-city/index.test.tsx`

Expected: PASS.

```bash
git add apps/desktop/src/app/lunar-city/world-scene.tsx apps/desktop/src/app/lunar-city/world-scene.test.tsx apps/desktop/src/app/lunar-city/index.tsx apps/desktop/src/app/lunar-city/index.test.tsx
git commit -m "feat(desktop): render lunar city world events"
```

## Task 7: Integrate desktop seams and perform acceptance verification

**Files:**
- Modify: `apps/desktop/src/app/contrib/surfaces.tsx` only if required by the established action context
- Modify: `apps/desktop/src/app/routes.ts` only if route registration tests identify a missing route contract
- Modify: `apps/desktop/src/app/chat/sidebar/index.tsx` only if world visibility gating regresses
- Modify: `apps/desktop/src/app/settings/plugins-settings.tsx` only if the existing Enable World setting must expose the lifecycle behavior
- Modify: `apps/desktop/src/app/lunar-city/index.test.tsx` for final integration coverage

**Interfaces:**
- Consumes: all previous tasks and the current desktop route/settings/sidebar WIP.
- Produces: a world surface that can be enabled, opened, used, disabled, and rebuilt with the standard desktop commands.

- [ ] **Step 1: Run focused Lunar City tests and inspect failures.**

Run: `npm run test:ui -- src/app/lunar-city src/store/agent-notices.test.ts`

Expected: all Lunar City and notice tests pass; any failure must identify an actual seam mismatch rather than be suppressed.

- [ ] **Step 2: Run TypeScript and lint checks for the renderer.**

Run: `npm run typecheck`

Expected: PASS with no missing SDK types, implicit `any`, or route/action type drift.

Run: `npm run lint -- --max-warnings=0`

Expected: PASS with no new lint warnings.

- [ ] **Step 3: Run the desktop UI suite.**

Run: `npm run test:ui`

Expected: PASS, including existing Kanban, agent notice, session, route, sidebar, and settings tests.

- [ ] **Step 4: Perform the renderer smoke check against the mock backend.**

Run: `npm run dev:mock` in one terminal, then `npm run dev:renderer` in a second terminal. Open `/lunar-city`, enable the world, exercise the dispatcher cube with a fixture task, produce blocked/completed/error notification frames in the mock source, and confirm scenes/dialogue update without a second window. Close the route and confirm the mock source shows no remaining world listener.

Expected: the same source ID/status appears in standard Kanban and Lunar City; blocked repair rejection remains visible; stable merge celebration is one-shot; unknown events show an alert fallback; closing Lunar City stops world subscriptions.

- [ ] **Step 5: Verify scope and commit the integration.**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only the intended Lunar City implementation changes are present. Preserve unrelated user WIP and do not stage it.

```bash
git add apps/desktop/src/app/contrib/surfaces.tsx apps/desktop/src/app/routes.ts apps/desktop/src/app/chat/sidebar/index.tsx apps/desktop/src/app/settings/plugins-settings.tsx apps/desktop/src/app/lunar-city/index.test.tsx
git commit -m "test(desktop): verify lunar city integration"
```

## Follow-on asset track

After this plan is green, high-fidelity buildings, leaders, workers, children,
NPC facial/body animation, task-specific clips, PBR materials, and Blender
scene packaging should be delivered as separate asset-focused plans. They must
consume the semantic tags from `world-presentation.ts` and never encode Hermes
backend event names directly. This keeps visual upgrades independent from
Hermes protocol changes and lets missing assets use the tested generic fallbacks.
