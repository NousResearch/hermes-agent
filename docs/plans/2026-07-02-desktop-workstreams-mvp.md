## SPEC

spec-stakes: high

**Goal:** Make Hermes Desktop show Telegram-topic-style task state directly in the session sidebar, so Alfredo can monitor active Hermes work from Desktop and reserve Telegram title edits for fallback/mobile use.

**Acceptance criteria (observable):**

1. Desktop has a typed Workstream vocabulary with the 11 Telegram topic states (`work`, `verify`, `done`, `close`, `blocked`, `warn`, `delegate`, `workflow`, `plan_review`, `restart`, `idle`) and complete display metadata (`icon`, `label`, `tone`).
2. Phase 1 auto-derives only the states backed by existing Desktop renderer stores: `warn`, `blocked`, `delegate`, `work`, `done`, and `idle`.
3. Sidebar rows show the derived workstream status icon/label plus active todo/subagent counts without replacing existing working/needs-input behavior.
4. Sidebar rows use a per-session workstream selector that resolves the selected stored row to the active runtime id before reading live todo/subagent stores, while unrelated todo/subagent map updates preserve row activity object identity.
5. The MVP adds no backend schema, no second session database, and no Telegram API dependency.
6. Targeted Desktop tests prove state priority, stale-finished todo behavior, selector identity, counts, and row rendering.

**Validation method (how each criterion is proven):**

1. `cd apps/desktop && npm run test:ui -- src/store/workstream.test.ts` checks the exact icon/label/tone values for all 11 metadata entries.
2. `cd apps/desktop && npm run test:ui -- src/store/workstream.test.ts` checks derivation priority: needs input > failed subagent > active subagent > active todo/working > completed todos > idle.
3. `cd apps/desktop && npm run test:ui -- src/app/chat/sidebar/session-row.test.tsx` renders row badges from todos/subagents/attention and proves the working arc/status survives.
4. `cd apps/desktop && npm run test:ui -- src/store/workstream.test.ts` proves the per-session selector returns the same object when another session's todo, subagent, attention, or working state changes and reads runtime-keyed activity for the selected stored row.
5. `git diff --name-only` matches the explicit allowlist: `apps/desktop/src/store/workstream.ts`, `apps/desktop/src/store/workstream.test.ts`, `apps/desktop/src/app/chat/sidebar/session-row.tsx`, `apps/desktop/src/app/chat/sidebar/session-row.test.tsx`, and this plan doc only. `rg -n "telegram|tg-|Bot API|createForumTopic"` over the four changed Desktop app files returns no matches.
6. `cd apps/desktop && npm run typecheck`, changed-file eslint, targeted Vitest, and `cd apps/desktop && npm run build` pass.

**Out of scope:** Mission Control cockpit, GitHub bridge migration, plan-review renderer rewrite, Telegram suppression config, native notifications, persistent closed/safe-delete state, and right-rail live progress panel.

**Constraints / assumptions:** Build in Alfredo's fork on a short-lived feature branch/worktree. Preserve Desktop event flow. Reuse existing stores instead of adding a canonical backend state. No new user-facing `HERMES_*` env vars.

# Desktop Workstreams MVP Implementation Plan

> **For Hermes:** Implement directly in the isolated worktree, test-first, then run dual-review + dual-verify before merging to `main`.

**Goal:** Make Hermes Desktop show Telegram-topic-style task state directly in the session sidebar, so Alfredo can monitor active Hermes work from Desktop and reserve Telegram title edits for fallback/mobile use.

**Problem:** Telegram topic titles are useful but rate-limited. Desktop already has real-time event stores, but the sidebar does not expose the task/workstream state clearly.

**Architecture:** Add a renderer-only Workstream read model. It derives conservative status from existing Nanostores (`session`, `todos`, `subagents`) and feeds small sidebar badges through a per-session selector. No backend state. No Telegram clone.

**Tech Stack:** React 19, TypeScript, Nanostores, Vitest, Testing Library.

---

created: 2026-07-02T21:41Z
modified: 2026-07-02T21:41Z
commits: -
agents: gpt-5.5/main
sessions: telegram-thread-61951
back refs: - prior Telegram customization inventory in this session
fwd refs: -

## MVP Wireframe

```wireframe
surface: desktop
url: hermes://desktop/sidebar
<div class="wf-col" style="width:360px;gap:8px">
  <div class="wf-row" style="justify-content:space-between"><b>Hermes</b><span class="wf-pill">⌘N New</span></div>
  <div class="wf-card wf-row" style="gap:8px;align-items:center">
    <span>🤖</span><b>Desktop customizations</b><span class="wf-pill accent">2 agents</span><span class="wf-pill">4 todos</span>
  </div>
  <div class="wf-card wf-row" style="gap:8px;align-items:center">
    <span>❓</span><b>Plan review pending</b><span class="wf-pill warn">needs input</span>
  </div>
  <div class="wf-card wf-row" style="gap:8px;align-items:center;opacity:.75">
    <span>💤</span><b>Old research thread</b><span class="wf-pill">idle</span>
  </div>
</div>
```

## State Model

```data-model
{"entities":[{"name":"WorkstreamActivity","fields":[{"name":"sessionId","type":"string","pk":true},{"name":"state","type":"WorkstreamState"},{"name":"label","type":"string"},{"name":"icon","type":"string"},{"name":"activeTodoCount","type":"number"},{"name":"activeSubagentCount","type":"number"},{"name":"failedSubagentCount","type":"number"},{"name":"needsInput","type":"boolean"},{"name":"isWorking","type":"boolean"}]},{"name":"WorkstreamStateMeta","fields":[{"name":"state","type":"string","pk":true},{"name":"icon","type":"string"},{"name":"label","type":"string"},{"name":"tone","type":"string"}]}],"relations":[{"from":"WorkstreamActivity","to":"WorkstreamStateMeta","kind":"many-to-one","label":"uses display metadata"}]}
```

## File Footprint

```file-tree
{"title":"Phase 1 files","entries":[{"path":"apps/desktop/src/store/workstream.ts","change":"created","note":"typed state vocabulary, conservative derivation, per-session selector"},{"path":"apps/desktop/src/store/workstream.test.ts","change":"created","note":"metadata, state priority, stale todo, selector identity tests"},{"path":"apps/desktop/src/app/chat/sidebar/session-row.tsx","change":"modified","note":"row consumes per-session activity and renders badges"},{"path":"apps/desktop/src/app/chat/sidebar/session-row.test.tsx","change":"created","note":"row rendering and existing working/needs-input regression tests"},{"path":"docs/plans/2026-07-02-desktop-workstreams-mvp.md","change":"created","note":"implementation contract"}]}
```

## Phase 1: Workstream read model

- [x] Task 1.1 — Write failing `workstream.test.ts` for metadata completeness, conservative state priority, stale finished todos, and selector identity.
- [x] Task 1.2 — Create `store/workstream.ts` with `WORKSTREAM_STATE_META`, `deriveWorkstreamActivity`, count helpers, and a keyed `$workstreamActivity(sessionId)` selector.
- [x] Task 1.3 — Run targeted store test green.

## Phase 2: Sidebar row badges

- [x] Task 2.1 — Write failing row render test for `warn`, `delegate`, stale completed todos, existing working behavior, todo count, and subagent count.
- [x] Task 2.2 — Wire `SidebarSessionRow` to keyed `$workstreamActivity(sessionId)` instead of whole-map todo/subagent subscriptions, with active runtime-id fallback for the selected stored row.
- [x] Task 2.3 — Render compact badge text without disrupting existing lead dot / handoff avatar / action menu.
- [x] Task 2.4 — Run targeted row test green.

## Phase 3: Verification

- [x] `cd apps/desktop && npm run test:ui -- src/store/workstream.test.ts src/app/chat/sidebar/session-row.test.tsx` — proves MVP behavior. Result after review fixes: 2 files, 15 tests passed.
- [x] `cd apps/desktop && npm run typecheck` — proves TS integration. Result after selector change: passed.
- [x] `cd apps/desktop && npx eslint src/store/workstream.ts src/store/workstream.test.ts src/app/chat/sidebar/session-row.tsx src/app/chat/sidebar/session-row.test.tsx` — proves changed-file lint. Result after selector change: passed.
- [x] `cd apps/desktop && npm run build` — proves Desktop bundle builds. Result after selector change: passed with pre-existing bundle/CSS warnings only.
- [x] `git diff --name-only` against explicit allowlist — proves no backend, DB, or Telegram script edits. Result: only 5 allowed files changed.
- [ ] Dual-review diff before commit.
- [ ] Merge feature branch into `main`, push `origin main`, clean worktree.

## Build Notes (deviations from plan)

- Fresh worktree had no `node_modules`; test/build used local symlinks to the main checkout's installed `node_modules`. These symlinks are excluded from git and not part of the diff.
- Plan/code critics found four real blockers: 11-state overclaim, monitor-vs-manage goal drift, row-level full-map subscription risk, and stored-vs-runtime live store mismatch. Folded by narrowing Phase 1 derivation scope, narrowing the goal to monitoring, adding keyed `$workstreamActivity(sessionId)`, adding selected-row runtime fallback, and adding selector/stale-todo/working/runtime-id tests.
- A stale async spec review still found useful validation gaps. Folded by making metadata tests exact, broadening selector-identity tests across unrelated todo/subagent/attention/working updates, and adding a direct no-Telegram-import grep over changed Desktop app files.
