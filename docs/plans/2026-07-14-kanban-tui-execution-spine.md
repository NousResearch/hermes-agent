# Kanban TUI Execution Spine Implementation Plan

> **For Hermes:** Use `subagent-driven-development` to implement this plan task-by-task, with an independent reviewer after the implementation pass.

**Goal:** Add a native, visually restrained Kanban activity surface to the Hermes TUI that shows durable worker/task progress as a vertical execution spine without injecting status messages into the conversation transcript.

**Architecture:** Extract a shared, read-only Kanban activity projection from the existing database layer, expose it through a TUI JSON-RPC method, and render it in a composer-adjacent Ink component. The resting surface is one compact summary line; focusing it reveals a vertical task/dependency spine. Reuse existing TUI tree, timing, theme, focus, and polling patterns rather than building a second rendering system.

**Tech Stack:** Python, SQLite, Hermes JSON-RPC gateway, TypeScript, React 19, Ink 6, Nanostores, Vitest, Pytest.

### 2026-07-14 execution hold and official-doc alignment

The first implementation pass intentionally stops short of mounting or polling this surface in live TUI sessions. Backend projection/RPC behavior and dormant, fixture-driven Ink components may be built and tested, but `appLayout.tsx`, live session hooks, keyboard routing, transcript state, and notification/subscription behavior remain unchanged until a separate visual approval pass.

This matches the current Nous documentation and product boundaries:

- The TUI is chrome around the same Python runtime and uses its local `tui_gateway`; it should not call the dashboard HTTP API or create a second chat surface.
- Ambient TUI activity is hidden by default, so a future Kanban dock should be explicitly scoped and visually calm rather than reviving a noisy generic activity feed.
- `/agents` is a current-session `delegate_task` audit/control surface. Kanban is a durable, multi-profile, multi-board work queue, so its visualization should remain distinct and read-only.
- Boards are hard isolation boundaries backed by separate databases. The projection must resolve and group boards explicitly, while the resting line may omit board chrome for the common single-board case.
- Kanban workers use `kanban_*` tools and the shared `kanban_db` layer; the TUI is a human observability consumer, not a worker-control or message-delivery path.

The activation work in Tasks 8–9 is therefore deferred, not silently completed. The dormant slice must prove truthfulness, bounded data, deterministic topology, narrow-terminal rendering, and zero transcript/session mutation before live wiring is reconsidered.

---

## Copy/paste kickoff prompt

```text
Implement the native Hermes Kanban TUI Execution Spine described in:

/Users/henrymiell/.hermes/hermes-agent/docs/plans/2026-07-14-kanban-tui-execution-spine.md

Repository:
/Users/henrymiell/.hermes/hermes-agent

Verified planning baseline:
- base branch: main
- baseline commit at planning time: df5700ebe317ff9f2d9ea4677513e012eb68b6f4
- main checkout currently contains unrelated dirty Supermemory edits:
  - plugins/memory/supermemory/__init__.py
  - tests/plugins/memory/test_supermemory_provider.py

Do not edit, stash, reset, restore, commit, or otherwise disturb the dirty main checkout. Create a clean worktree from current origin/main, for example:

/Users/henrymiell/.hermes/worktrees/kanban-tui-execution-spine
branch: feat/kanban-tui-execution-spine

First verify the plan against current source because line numbers and APIs may have moved. Then implement the plan with TDD and small local commits.

Product requirement:
Create a native TUI activity dock that remains outside transcript/message history. In its resting state it occupies one calm line near the composer. When focused, it expands into a vertical execution spine using actual Kanban tasks, dependencies, worker state, heartbeat freshness, blockers, and completion state. It must never fabricate percentage completion or workflow phases.

Visual direction:
- distinctive Hermes execution spine, not a Claude Agent View clone
- vertical filled/unfilled rail: ● completed, ◉ active, ○ upcoming, ! blocked/needs input, × failed
- branch child tasks sideways using restrained box-drawing characters
- no card wall, giant rounded container, status-pill soup, rainbow colors, or permanent shortcut footer
- no continuous distracting animation; a subtle active-node pulse is acceptable and must respect static/reduced-motion behavior
- one accent color; amber only for attention; red only for actual failure
- hide the dock when there is no relevant activity
- narrow-terminal behavior must remain useful and must not crush the composer

Critical constraints:
1. This is a display-only observability surface. Do not synthesize MessageEvent objects, call adapter.handle_message(), append Msg rows, wake an agent, resume a session, or inject completion content into a transcript.
2. Do not enable Kanban auto-subscription or change notification routing.
3. Do not call the dashboard HTTP API from the TUI and do not duplicate its SQL. Put shared read-only activity projection logic in hermes_cli/kanban_db.py and use it from both consumers where appropriate.
4. Do not add termination, mutation, approval, push, merge, or deployment controls.
5. Do not show raw prompts, environment variables, command lines, secrets, full logs, or private worker context.
6. Do not create a general TUI plugin framework. Kanban is bundled; implement the native surface first and generalize only if a second real activity provider exists.
7. Do not add a new database or broad schema migration unless direct inspection proves existing task/task_run/task_event/task_link data cannot satisfy the accepted V1.
8. No push, PR, merge, deployment, branch deletion, or modification of the dirty main checkout.

Required implementation sequence:
1. Establish clean worktree and baseline tests.
2. Build pure activity model and deterministic vertical-rail fixtures/tests.
3. Build the static Ink dock and visually verify wide/narrow states before backend wiring.
4. Extract a shared read-only Kanban activity projection.
5. Add a TUI `kanban.activity` JSON-RPC method.
6. Add adaptive polling/store logic with cleanup and no idle flicker.
7. Integrate the resting dock and focused execution spine near the composer.
8. Add keyboard/focus behavior without stealing normal text navigation.
9. Verify multi-board, branching, stale, blocked, failed, completed, empty, and narrow-terminal states.
10. Run targeted Python and full TUI checks, then perform an independent review.

Definition of done:
- the dock reflects real Kanban state while work progresses
- the vertical rail is visually stable and truthful
- existing subagent/todo/status UI continues to work
- no activity update enters transcript history
- polling is bounded and cleaned up
- all required tests pass
- wide and narrow Ghostty screenshots are supplied for visual review

Final handoff must include:
- absolute worktree path
- branch and base commit
- exact changed files
- git status
- test commands and real outputs
- screenshots for wide, narrow, blocked, and branching states
- unresolved risks or design compromises
- confirmation that no push/merge/deploy occurred
```

---

## 1. Verified current state

### Existing infrastructure to reuse

- `plugins/kanban/dashboard/plugin_api.py`
  - `GET /workers/active`
  - `GET /runs/{run_id}`
  - `GET /runs/{run_id}/inspect`
  - bounded task log and diagnostics endpoints elsewhere in the same module
- `hermes_cli/kanban_db.py`
  - board enumeration
  - tasks, runs, events, links, heartbeat, and diagnostics persistence
- `tui_gateway/server.py`
  - typed JSON-RPC method registry
  - existing `delegation.status` read-only status method
- `ui-tui/src/components/thinking.tsx`
  - `SubagentAccordion`
  - tree stems, expandable sections, elapsed-time treatment, restrained status tones
- `ui-tui/src/lib/subagentTree.ts`
  - tree construction and duration formatting utilities
- `ui-tui/src/components/agentsOverlay.tsx`
  - focus, cursor, expansion, scrolling, and periodic live refresh patterns
- `ui-tui/src/components/appLayout.tsx`
  - composer-adjacent rendering location
  - current coarse `N background tasks running` line at lines approximately 337–341
- `ui-tui/src/app/useMainApp.ts`
  - existing local JSON-RPC polling pattern with change-aware store updates

### What does not exist

There is no current `KanbanActivity`, `kanban.activity`, Kanban execution rail, TUI worker dock, or Kanban-to-TUI activity store.

A closed upstream issue, `NousResearch/hermes-agent#37109`, proposed an active-worker panel for the **web dashboard**, not the TUI. It was closed without an implementation branch or PR. `origin/bb/desktop-kanban` concerns the desktop contribution shell, not this feature.

### Dirty-checkout warning

The planning checkout is `main` and currently has unrelated modifications:

```text
M plugins/memory/supermemory/__init__.py
M tests/plugins/memory/test_supermemory_provider.py
```

Implementation must happen in a clean worktree. The plan file itself will initially be untracked in the main checkout, so the implementer should read it from this absolute path or copy it into the clean worktree before beginning.

---

## 2. Product contract

### Resting state

The persistent surface must use at most one line when unfocused:

```text
KANBAN  3 active   ●━●━◉─○   adaptive cycle · verifying
```

If width is constrained, progressively collapse detail:

```text
KANBAN  3 active   ◉ verifying
```

Then:

```text
KANBAN  3 active
```

Never wrap the composer because of this dock.

### Focused state

```text
KANBAN / adaptive-cycle-core                         08:14

●  architecture audit                               done
┃
◉  implementation                                   running
├── ● models                                        done
├── ◉ persistence                                   running
└── ○ independent review                            waiting
│
○  completion
```

This is illustrative; labels must come from real task titles/status, not a fabricated universal phase taxonomy.

### Visual grammar

| Glyph | Meaning |
|---|---|
| `●` | Completed task/node |
| `◉` | Running/current task/node |
| `○` | Queued, ready, scheduled, or dependency-gated future node |
| `!` | Blocked or requires human input |
| `×` | Failed run |
| `┃` | Completed connection |
| `│` | Pending/current connection |
| `├──` / `└──` | Parallel/dependent child task |

Task position communicates workflow progress. Color/animation communicates health. Do not overload one signal with both meanings.

### Truthfulness rules

- Never show a percentage unless backed by a real `current/total` value.
- Prefer actual task links and statuses over inferred stages.
- For a root task without children, show its lifecycle and worker health without inventing missing phases.
- Use `block_reason` verbatim only after safe truncation/redaction; do not expose worker context.
- Use run summary only after completion.
- If no meaningful headline exists, show `Working`, not a guessed activity.
- Stale heartbeat is health information, not proof that the task failed.

### Visibility rules

- Hidden when there is no active, blocked-attention, or recently completed work.
- Resting mode shows the highest-attention root task plus aggregate counts.
- Focused mode may show multiple boards, grouped by board.
- Completed tasks may linger briefly in local UI state, then disappear; do not persist presentation-only linger state to Kanban.
- The panel must not steal focus when a task changes state.

---

## 3. Non-goals

- Replacing the web Kanban dashboard
- Building a generic dashboard/plugin framework for TUI components
- Attaching the user to a worker conversation
- Sending messages to workers from this panel
- Mutating task status
- Terminating or reclaiming workers
- Showing full worker logs in the resting/focused dock
- Model-generated progress summaries in V1
- Continuous tool-call streaming from every Kanban worker in V1
- Cross-session completion delivery
- Enabling Kanban notification subscriptions
- Redesigning `delegate_task` subagent UI
- Adding fake `review` or other statuses not present in the actual board schema

---

## 4. Implementation plan

### Task 1: Establish isolated workspace and baseline

**Objective:** Prove the feature starts from a clean, reproducible checkout and existing tests pass.

**Files:** None.

**Steps:**

1. From the main repository, create a clean worktree from current `origin/main`:

   ```bash
   git worktree add -b feat/kanban-tui-execution-spine \
     /Users/henrymiell/.hermes/worktrees/kanban-tui-execution-spine \
     origin/main
   ```

2. Copy this plan into the worktree without modifying the dirty main checkout:

   ```bash
   mkdir -p /Users/henrymiell/.hermes/worktrees/kanban-tui-execution-spine/docs/plans
   cp /Users/henrymiell/.hermes/hermes-agent/docs/plans/2026-07-14-kanban-tui-execution-spine.md \
      /Users/henrymiell/.hermes/worktrees/kanban-tui-execution-spine/docs/plans/
   ```

3. Record:

   ```bash
   git status --short --branch
   git rev-parse HEAD
   ```

4. Run baseline TUI checks:

   ```bash
   npm --prefix ui-tui run check
   ```

5. Run baseline Kanban dashboard tests:

   ```bash
   python -m pytest tests/plugins/test_kanban_dashboard_plugin.py -q
   ```

6. If baseline failures exist, record them before changing code. Do not silently repair unrelated failures.

**Done when:** Clean branch/worktree exists and baseline results are recorded.

---

### Task 2: Define the pure activity projection contract

**Objective:** Create a serializable, read-only activity model that both gateway and TUI can reason about deterministically.

**Files:**

- Modify: `hermes_cli/kanban_db.py`
- Create: `tests/hermes_cli/test_kanban_activity_projection.py`

**Design:**

Add a focused helper, name subject to current conventions, equivalent to:

```python
def get_activity_snapshot(*, board: str) -> dict:
    """Return a bounded, read-only activity projection for one Kanban board."""
```

The snapshot should include only bounded presentation-safe fields:

```python
{
    "board": "default",
    "checked_at": 1721000000,
    "roots": [
        {
            "task_id": "t_...",
            "title": "...",
            "status": "running",
            "assignee": "localimplementer",
            "block_reason": None,
            "parents": [],
            "children": [...],
            "run": {
                "run_id": 12,
                "profile": "localimplementer",
                "started_at": 1720999000,
                "ended_at": None,
                "outcome": None,
                "last_heartbeat_at": 1720999990,
                "max_runtime_seconds": 7200,
            },
        }
    ],
}
```

Do not include:

- prompt/body text
- comments
- environment variables
- raw command line
- full logs
- credentials or configuration

**Required tests:**

1. Empty board returns a stable empty snapshot.
2. Active run appears with task/profile/timing data.
3. Root/child dependencies are nested deterministically.
4. Multiple roots preserve deterministic ordering.
5. Blocked task includes bounded reason and no private body.
6. Completed/failed outcomes serialize correctly.
7. Cycles or malformed links fail safely without recursive explosion.
8. Snapshot query does not mutate board/task/run state.

**Verification:**

```bash
python -m pytest tests/hermes_cli/test_kanban_activity_projection.py -q
```

**Done when:** The pure snapshot contract passes without dashboard or TUI dependencies.

---

### Task 3: Reuse the shared projection in the dashboard backend

**Objective:** Remove worker-list SQL duplication and preserve the existing API contract.

**Files:**

- Modify: `plugins/kanban/dashboard/plugin_api.py`
- Modify: `tests/plugins/test_kanban_dashboard_plugin.py`

**Steps:**

1. Add a failing regression test for `GET /workers/active` using an active `task_runs` fixture.
2. Refactor the endpoint to consume the shared `kanban_db` helper or a narrower shared worker helper extracted alongside it.
3. Preserve current response fields and ordering exactly unless a breaking change is explicitly justified.
4. Do not add dashboard UI work in this task.

**Verification:**

```bash
python -m pytest tests/plugins/test_kanban_dashboard_plugin.py -q
```

**Done when:** Existing API behavior is preserved and active-worker SQL has one source of truth.

---

### Task 4: Add the TUI gateway activity method

**Objective:** Expose Kanban activity through the existing local JSON-RPC transport without HTTP or chat-message delivery.

**Files:**

- Modify: `tui_gateway/server.py`
- Create or modify the focused TUI-gateway test file matching current test organization

**Method:**

```text
kanban.activity
```

**Request:**

```json
{
  "boards": ["default", "formly"]
}
```

If `boards` is omitted, return activity for all non-archived boards that currently contain relevant tasks. Board ordering must be deterministic.

**Response:**

```json
{
  "boards": [...],
  "active_count": 3,
  "attention_count": 1,
  "checked_at": 1721000000
}
```

**Requirements:**

- Read-only.
- No session lookup required.
- No `MessageEvent` creation.
- No notification queue interaction.
- No `adapter.handle_message()`.
- No auto-subscription.
- A broken/missing board should not crash the TUI; return a bounded per-board error or skip with diagnostics.
- Do not inspect every PID or load logs on each poll.

**Required tests:**

1. Empty response.
2. Multi-board aggregation.
3. Active and blocked counts.
4. Malformed board fails safely.
5. Method does not alter session history or notification queues.

**Verification:**

```bash
python -m pytest <focused-tui-gateway-test-file> -q
```

**Done when:** The method returns deterministic snapshots with no delivery side effects.

---

### Task 5: Build pure TUI rail-model helpers

**Objective:** Convert gateway snapshots into stable, truthful execution-spine rows independently of React rendering.

**Files:**

- Create: `ui-tui/src/lib/kanbanActivity.ts`
- Create: `ui-tui/src/__tests__/kanbanActivity.test.ts`
- Modify: `ui-tui/src/gatewayTypes.ts`

**Pure helpers should cover:**

- normalization from snake_case gateway payloads
- deterministic root/task ordering
- glyph choice
- tone choice
- heartbeat freshness classification
- focused/collapsed labels
- width-based truncation
- branch compression
- aggregate counts

**Required states:**

- queued/ready/scheduled
- running
- blocked/needs attention
- completed
- failed/stopped outcome
- stale heartbeat
- missing worker/run metadata
- malformed/cyclic child data

**Tests:**

- glyph and tone are deterministic
- completed path uses `┃`; upcoming/current path uses `│`
- no fake percentage appears
- branch compression engages beyond the visible limit
- narrow labels do not wrap
- attention takes precedence over ordinary active work
- task order does not jump when only heartbeat timestamps change

**Verification:**

```bash
npm --prefix ui-tui test -- --run src/__tests__/kanbanActivity.test.ts
```

**Done when:** Fixtures generate stable rail rows without rendering or network code.

---

### Task 6: Build and visually approve a static Ink execution spine

**Objective:** Lock the visual treatment before connecting live polling.

**Files:**

- Create: `ui-tui/src/components/kanbanActivity.tsx`
- Create: `ui-tui/src/__tests__/kanbanActivityRender.test.tsx`
- Reuse: `ui-tui/src/theme.ts`
- Reuse carefully: `ui-tui/src/lib/subagentTree.ts`, `ui-tui/src/components/thinking.tsx`

**Components:**

```tsx
<KanbanActivityDock />
<KanbanExecutionSpine />
<KanbanActivityRow />
```

Avoid duplicating large portions of `SubagentAccordion`. Extract a small shared primitive only when it genuinely serves both components.

**Fixture states to render:**

1. One active task, no children.
2. Parent with three parallel children.
3. Dependency pipeline with completed/current/upcoming nodes.
4. Blocked task requiring attention.
5. Stale worker.
6. Failed run.
7. Multiple boards.
8. Empty state.
9. Narrow terminal.
10. Many children compressed into a summary.

**Visual acceptance:**

- Resting dock uses one line.
- Focused spine has no mandatory surrounding box.
- One accent color dominates.
- Amber/red are semantic only.
- Labels align without turning into columns of metadata.
- Branches remain legible.
- No raw IDs in resting mode.
- No permanent shortcut footer.
- Narrow terminal retains task count/state and does not wrap composer content.
- Static mode is fully understandable without animation.

**Verification:**

```bash
npm --prefix ui-tui test -- --run src/__tests__/kanbanActivityRender.test.tsx
npm --prefix ui-tui run typecheck
```

Also run the deterministic fixture in Ghostty and capture wide/narrow screenshots before proceeding.

**Stop condition:** If the rail looks like a dense log, card stack, or arbitrary tree dump, revise the visual model before backend integration.

---

### Task 7: Add activity store and adaptive polling

**Objective:** Keep the dock current without expensive work, idle flicker, or transcript updates.

**Files:**

- Create: `ui-tui/src/app/kanbanActivityStore.ts`
- Create: `ui-tui/src/app/useKanbanActivity.ts`
- Create: `ui-tui/src/__tests__/kanbanActivityStore.test.ts`
- Modify: `ui-tui/src/app/useMainApp.ts` or the narrowest appropriate app composition hook

**Requirements:**

- Fetch `kanban.activity` immediately on mount.
- Poll more frequently only while active/attention work exists.
- Back off while idle.
- Stop timers on unmount/session teardown.
- Do not patch global state when the semantic snapshot has not changed.
- Keep presentation-only completion linger local to the store.
- Network/gateway errors should degrade to last-known state plus a quiet stale indicator, not crash the TUI.
- Polling is local RPC and must not trigger model calls.

**Tests:**

- initial fetch
- active and idle cadence selection
- timer cleanup
- unchanged snapshot does not notify subscribers
- completion linger expires locally
- RPC failure preserves last-known state
- no transcript store access/import

**Verification:**

```bash
npm --prefix ui-tui test -- --run src/__tests__/kanbanActivityStore.test.ts
```

**Done when:** Live data updates without unnecessary whole-TUI rerenders.

---

### Task 8: Integrate dock and keyboard focus near the composer

**Objective:** Place the activity surface inside the TUI chat chrome without placing it in transcript history.

**Files:**

- Modify: `ui-tui/src/components/appLayout.tsx`
- Modify: `ui-tui/src/app/interfaces.ts`
- Modify: `ui-tui/src/app/uiStore.ts` or add a dedicated activity-focus store
- Modify the existing keyboard-routing module selected after inspection
- Add focused integration tests under `ui-tui/src/__tests__/`

**Placement:**

Render the dock in `ComposerPane`, near the existing background-task indicator and queued-message area, before the input itself. It must remain outside `historyItems`, `Msg`, `appendMessage`, and transcript virtualization.

Replace or subsume the coarse:

```text
N background tasks running
```

only if doing so does not remove useful non-Kanban background-process visibility. Kanban and terminal background processes are different domains; do not mislabel one as the other.

**Focus behavior:**

- The resting dock must not intercept typing.
- When input is empty and the cursor is in its neutral position, a non-conflicting action may focus the dock.
- Inspect existing arrow-key and overlay bindings before selecting the final shortcut.
- Focused mode supports task selection and expansion.
- `Esc` always returns cleanly to the composer.
- State changes never steal focus.
- Do not append shortcut hints to transcript or permanent chrome.

**Integration tests:**

1. Empty activity renders nothing.
2. Active activity renders above composer.
3. Rendering does not add a `Msg` or history item.
4. Updating activity does not change transcript length.
5. Focus/escape round trip.
6. Normal arrow navigation remains intact when dock is not focused.
7. Narrow composer width remains stable.
8. Existing queued messages, status rule, overlays, and background-process indicator still render.

**Verification:**

```bash
npm --prefix ui-tui run check
```

**Done when:** The dock behaves like TUI chrome, not conversation content.

---

### Task 9: Add restrained activity animation and accessibility fallback

**Objective:** Communicate liveness without visual noise.

**Files:**

- Modify: `ui-tui/src/components/kanbanActivity.tsx`
- Modify: relevant tests

**Rules:**

- Default understanding must not depend on animation.
- At most the active node may pulse or alternate subtly.
- Completed, blocked, failed, and stale nodes remain static.
- Do not animate every worker.
- Reuse existing indicator timing primitives where appropriate.
- If Hermes lacks an explicit reduced-motion setting, provide a static mode tied to the closest existing accessibility/config mechanism or leave animation out of V1 rather than inventing a broad settings system.

**Verification:**

- fake-timer test proves animation timer cleanup
- static fixture remains legible
- no idle animation loop when no active node exists

**Done when:** Motion adds liveness without becoming a spinner wall.

---

### Task 10: End-to-end verification with isolated Kanban fixtures

**Objective:** Prove the feature against real local Kanban state, not only mocked React props.

**Files:**

- Add a focused integration fixture/test in the existing TUI-gateway or Kanban test location
- Do not use the user’s live board database

**Scenarios:**

1. Empty board: dock absent.
2. One running worker: active node visible.
3. Parent done, implementer running, reviewer dependency-gated: vertical branch is correct.
4. Blocked task: amber attention state with bounded reason.
5. Heartbeat becomes stale: health changes without moving task position.
6. Run completes: node fills, result lingers locally, then collapses.
7. Failed run: red failure node.
8. Multiple boards: deterministic grouping.
9. Gateway unavailable: last-known state degrades quietly.
10. No conversation/session history mutation in any scenario.

**Required commands:**

```bash
python -m pytest tests/hermes_cli/test_kanban_activity_projection.py -q
python -m pytest tests/plugins/test_kanban_dashboard_plugin.py -q
python -m pytest <focused-tui-gateway-test-file> -q
npm --prefix ui-tui run check
```

Run any broader repository tests required by files touched.

**Manual visual verification:**

Capture Ghostty screenshots for:

- wide terminal, branching task
- narrow terminal
- blocked/attention state
- multiple active boards

Check for:

- no flicker while idle
- no composer rewrap
- no focus theft
- no transcript lines produced by updates
- stable node placement
- readable hierarchy without a border/card wall

---

### Task 11: Independent review and final handoff

**Objective:** Verify specification compliance, code quality, and contamination safety independently.

**Reviewer checklist:**

- [ ] No existing feature was unnecessarily reimplemented.
- [ ] Existing subagent/tree/theme/focus primitives are reused appropriately.
- [ ] Shared Kanban SQL/projection has one source of truth.
- [ ] TUI does not call dashboard HTTP endpoints.
- [ ] No synthetic message, agent wake, auto-subscription, or transcript mutation path exists.
- [ ] No private prompts, body text, raw env, cmdline, or unbounded logs are exposed.
- [ ] No fake percentage or invented workflow phase appears.
- [ ] Multi-board and malformed graph handling are bounded.
- [ ] Polling cleans up and avoids unchanged global-state writes.
- [ ] Narrow-terminal and empty-state behavior are acceptable.
- [ ] Visual result is an execution spine, not a Claude Agent View clone.
- [ ] Tests contain explicit contamination regression coverage.
- [ ] Dirty main-checkout Supermemory files were untouched.

**Final handoff format:**

```text
Worktree:
Branch:
Base commit:
Current git status:

Changed files:
- ...

Behavior delivered:
- ...

Verification:
- command: ...
  result: ...

Visual evidence:
- wide screenshot: ...
- narrow screenshot: ...
- blocked screenshot: ...
- branching screenshot: ...

Safety confirmation:
- no transcript injection
- no notification/subscription changes
- no push/merge/deploy
- dirty main checkout untouched

Open risks / follow-up candidates:
- ...
```

Do not push, merge, open a PR, deploy, delete the worktree, or modify remote state without fresh explicit approval.

---

## 5. Acceptance criteria

### Functional

- [ ] TUI shows a one-line Kanban summary only when relevant work exists.
- [ ] Focus reveals a vertical execution spine based on real tasks/links/statuses.
- [ ] Active, queued, blocked, completed, failed, and stale states are distinct.
- [ ] Multi-board activity is grouped deterministically.
- [ ] Branches compress gracefully when numerous.
- [ ] Polling updates the view as Kanban changes.
- [ ] Gateway or board failure degrades safely.

### Visual

- [ ] Resting state consumes one line.
- [ ] No permanent giant box or card wall.
- [ ] No rainbow status treatment.
- [ ] Rail remains legible without motion.
- [ ] Narrow terminal does not break composer geometry.
- [ ] Focused graph remains spatially stable.
- [ ] Attention is clear without focus stealing.

### Isolation and safety

- [ ] No activity update enters transcript/history state.
- [ ] No synthetic user or system message is generated.
- [ ] No agent is woken or resumed.
- [ ] No Kanban subscription is created or enabled.
- [ ] No raw prompt/body/env/cmdline/full log is displayed.
- [ ] No mutation controls are introduced.
- [ ] Existing dirty files remain untouched.

### Quality

- [ ] Pure projection/model logic has deterministic tests.
- [ ] TUI component has wide/narrow/state rendering tests.
- [ ] Polling cleanup and unchanged-state behavior are tested.
- [ ] Existing dashboard worker API tests remain green.
- [ ] Full `ui-tui` check passes.
- [ ] Independent reviewer approves or records bounded required fixes.

---

## 6. Risks and mitigations

### Risk: The vertical spine implies false sequential progress

**Mitigation:** Render actual dependency/task topology. For single tasks, show lifecycle/health only. Never invent phases or percentages.

### Risk: A global board view recreates cross-session contamination

**Mitigation:** The activity surface is read-only UI chrome. It does not attach results to a session, send messages, or enter context. Group by board and make the global scope explicit.

### Risk: Polling causes flicker or CPU churn

**Mitigation:** Adaptive cadence, change-aware store updates, no PID/log inspection on every poll, timer cleanup tests.

### Risk: Height consumption makes chat worse

**Mitigation:** One-line resting state, focused expansion only, branch compression, narrow-terminal collapse.

### Risk: Raw operational data leaks secrets

**Mitigation:** Projection allowlist. Exclude task bodies, prompts, comments, environment, command lines, and full logs by construction.

### Risk: Over-generalizing into a plugin framework delays value

**Mitigation:** Native bundled Kanban implementation first. Extract a generic activity-source interface only after a second real consumer exists.

### Risk: Existing subagent UI is copied wholesale and becomes a second maintenance burden

**Mitigation:** Reuse small primitives and formatting helpers, but keep Kanban domain modeling separate from delegation-domain modeling.

---

## 7. Future work explicitly deferred

Only consider after V1 is visually approved and stable:

- structured `phase`, `current`, and `total` fields on `kanban_heartbeat`
- sanitized per-tool activity headlines for Kanban workers
- optional model-written stale activity summaries
- full-screen Kanban activity overlay
- dashboard execution-spine parity
- generic TUI `activity_source` plugin contract
- explicit, session-safe worker attachment after routing identity is redesigned

These are not required for the first implementation.
