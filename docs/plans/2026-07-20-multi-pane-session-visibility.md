# Refactor: pane-owned content, not one selected session

> **Status:** DRAFT — rewritten for review before implementation.

## Problem

Desktop still carries a single-session model through `$selectedStoredSessionId`.
That atom means “the one session selected by the URL router,” and many parts of
the app accidentally use it to answer unrelated questions:

- Is this session on screen?
- Is this sidebar row selected?
- Which session should clear its unread marker?
- What session should the statusbar/title describe?

That worked when the app had one chat surface. It is wrong now that the layout
tree can show several chat panes at once.

The concrete bug proves it:

1. Session A runs in the background.
2. The user drags A to an edge, creating a split beside session B.
3. A and B are both visible on screen.
4. A finishes.
5. The sidebar gives A the green “Finished — unread” dot because A is not
   `$selectedStoredSessionId`.

A is visibly open. It is not unread.

A stacked tab is the useful contrast: a session can be open in a tab but not
be its group’s active tab. That session is hidden and *should* receive the
unread dot when it finishes. The existing Playwright E2E test covers both
cases:

- tab, hidden → unread is correct (green)
- split, visible → unread is wrong (red today)

## Root cause

The app has two models of content:

1. The **layout tree** already models many panes, groups, tabs, splits, active
   tabs, and dismissed panes.
2. `$selectedStoredSessionId` models one routed session outside that tree.

The second model is a leftover single-pane authority. It duplicates part of
what the tree should own, and it cannot represent a split.

The fix is not a bigger collection of “selected” atoms. The fix is to make
**each pane own what it renders**.

## Target model

### Pane content is the source of truth

Every pane has content. A pane renders exactly one of:

```ts
type PaneContent =
  | { kind: 'chat'; storedSessionId: string | null }
  | { kind: 'page'; page: PageId }
```

`storedSessionId: null` means a fresh chat draft.

The layout tree already owns pane identity, grouping, ordering, active tabs,
splits, and dismissal. A new pane-content store owns the content binding for
those identities:

```ts
type PaneContentById = Record<string, PaneContent>

export const $paneContentById = atom<PaneContentById>({
  main: { kind: 'chat', storedSessionId: null },
})
```

There is no global “selected session.” There is no global “workspace session.”
A session belongs to the chat pane that renders it.

### Standard chat pane IDs

The current `workspace` pane becomes `main` (or another neutral permanent pane
id chosen during implementation). It is not semantically special: it is simply
the initial chat pane created by the default layout.

Tile panes retain stable generated ids such as `session-tile:<storedId>` while
that implementation remains useful. The key change is that their session
binding lives in `$paneContentById`, not in a separate `$sessionTiles` record.

A later cleanup may replace session-derived tile ids with opaque pane ids. That
is not required for this refactor; avoid combining identity migration with the
content-authority migration unless it proves necessary.

### Derived concepts, never writable duplicates

All UI concepts derive from pane content plus the layout tree:

| Question | Derived from |
|---|---|
| Which sessions are visible? | active chat pane in every non-dismissed visible group |
| Which session is focused? | active pane of `$activeTreeGroup`, then its chat content |
| Which session does a sidebar row represent? | its stored id |
| Is a session highlighted in the sidebar? | membership in `$visibleSessionIds` |
| Is a session unread? | it finished while absent from `$visibleSessionIds` |
| What should titlebar/statusbar show? | `$focusedPaneContent` |
| What does a pane’s composer submit to? | that pane’s own chat content/runtime binding |

The main invariants:

```ts
// Visibility is derived, not set by callers.
$visibleSessionIds = visible panes
  → active chat pane contents
  → non-null stored session ids

// Focus is derived, not set independently.
$focusedPaneId = active pane in active tree group
$focusedPaneContent = $paneContentById[$focusedPaneId]
$focusedStoredSessionId =
  $focusedPaneContent.kind === 'chat'
    ? $focusedPaneContent.storedSessionId
    : null
```

A stacked hidden tab is excluded because it is not the active pane in its
group. A split is included because each split group has one active pane and
both groups are rendered.

## Router removal

The desktop uses `HashRouter`, but the user never sees an address bar and the
hash is not a meaningful shareable/deep-linkable public interface. It currently
acts as a second, competing content registry for the main pane:

```txt
#/abc-session-id  → selected session
#/skills          → skills page
#/settings         → settings overlay/page
```

Remove it.

This does **not** replace it with a global `$workspaceRoute` or
`$workspaceSessionId`. Those would recreate the same single-pane ownership
problem under a different name.

Instead:

- Page navigation changes the content of the target pane:
  ```ts
  setPaneContent('main', { kind: 'page', page: 'skills' })
  ```
- Opening/resuming a chat changes the content of the pane the user targeted:
  ```ts
  setPaneContent(targetPaneId, { kind: 'chat', storedSessionId })
  ```
- Opening a fresh chat changes that pane to:
  ```ts
  setPaneContent(targetPaneId, { kind: 'chat', storedSessionId: null })
  ```
- Overlays remain overlay state. They are not pane routes.

`react-router-dom`, `HashRouter`, `Routes`, `Route`, `Navigate`, `useNavigate`,
`useLocation`, `useParams`, and URL-derived resume hooks are removed from the
desktop renderer after their remaining route responsibilities move to
pane-content actions.

Browser-style back/forward history is deliberately dropped. Desktop navigation
should follow the pane tree and explicit session history, not an invisible URL
history stack.

## Runtime binding

A stored session id is durable. A runtime id is ephemeral and belongs to the
live backend process. Each chat pane needs its own runtime binding:

```ts
type ChatPaneState = {
  storedSessionId: string | null
  runtimeId: string | null
  resumeError?: string
}
```

The current tile system already has this shape (`SessionTile.runtimeId`,
`SessionTile.error`). The main chat’s runtime state currently lives in global
atoms such as `$activeSessionId`, `$messages`, `$resumeFailedSessionId`, and
`$resumeExhaustedSessionId`.

Move that state behind a per-pane session controller/cache:

```ts
type ChatPaneRuntimeById = Record<string, ChatPaneRuntime>
```

The existing session cache remains authoritative for gateway events per runtime
id. Pane runtime bindings select a slice of that cache. A pane never owns a
copy of gateway truth; it owns only its choice of which runtime/session it
shows.

This is the key migration that makes composing, loading, retrying, and
interrupting correct for every visible chat pane instead of only the main one.

## Action API

Replace global “select/resume the session” actions with pane-targeted actions:

```ts
openSessionInPane({ paneId, storedSessionId }): Promise<void>
openFreshChatInPane({ paneId }): Promise<void>
closePane(paneId): void
splitPane({ sourcePaneId, direction, content }): string
focusPane(paneId): void
```

Callers choose their target deliberately:

- Sidebar click: focus an existing pane rendering the session; otherwise load
  it in the focused chat pane (or create a new pane only when the user chose a
  split/tab action).
- Sidebar drag to edge: create/move a pane with the dropped session content.
- Sidebar Ctrl-click: stack a new pane in the target group, but do not claim it
  is visible until it is the group’s active pane.
- Statusbar/title actions: operate on `$focusedPaneContent`.
- Session picker: opens content in the focused pane.

No action implicitly assumes one globally selected session.

## Unread behavior

Unread is still transient writable state because it survives a busy → idle
transition. Its rule becomes simple:

```ts
if (!$visibleSessionIds.get().includes(storedSessionId)) {
  addUnread(storedSessionId)
}
```

Whenever a session enters `$visibleSessionIds`, remove it from unread:

```ts
$visibleSessionIds.listen(ids => {
  removeUnread(ids)
})
```

This yields correct behavior for all cases:

| State at finish | In `$visibleSessionIds`? | Unread marker |
|---|---:|---:|
| focused main chat | yes | no |
| visible edge split | yes | no |
| active tab in another visible group | yes | no |
| inactive stacked tab | no | yes |
| closed tile | no | yes |
| background session | no | yes |

## Migration plan

This is a large refactor. Do it in vertical slices; each slice must preserve a
working desktop and pass its focused tests before moving on.

### Phase 0: Freeze the current contract with E2E tests

**Goal:** Establish real user-facing behavior before changing authority.

- Keep `e2e/tile-unread-bug.spec.ts`:
  - hidden tab → green unread dot (currently green)
  - visible split → no unread dot (currently red)
- Keep `e2e/sidebar-states.spec.ts` screenshots and assertions.
- Add an E2E assertion that the split test visibly has two active chat panes
  before asserting its dot state. Do not rely only on the drag completing.

### Phase 1: Introduce pane-content types and projection helpers

**Goal:** Create the new authority without moving behavior yet.

Create `src/store/panes.ts` (or extend the existing pane-tree store if it is
already the natural owner):

- `PaneContent`, `ChatPaneContent`, `PagePaneContent`, `PaneContentById`
- `$paneContentById`
- `paneContent(paneId)`
- `setPaneContent(paneId, content)`
- `clearPaneContent(paneId)`
- `$visiblePaneIds` derived from `$layoutTree` + `$dismissedPanes`
- `$visibleSessionIds` derived from `$visiblePaneIds` + `$paneContentById`
- `$focusedPaneId`, `$focusedPaneContent`, `$focusedStoredSessionId`

Seed the existing main pane’s current content into `$paneContentById` in a
one-way compatibility bridge. Do not remove `$selectedStoredSessionId` yet.

Tests:

- pure unit tests for split vs inactive-tab vs dismissed-pane visibility
- no DOM snapshot tests; assert the set of visible pane/session ids

### Phase 2: Fix unread from the new visibility projection

**Goal:** Make the known bug green without changing navigation yet.

- Change `handleTransition` in `session-states.ts` to consult
  `$visibleSessionIds`.
- Move unread clearing out of `setSelectedStoredSessionId`; add it as a
  listener in the pane-content owner so *any* newly visible session clears its
  marker.
- Keep `$selectedStoredSessionId` only as a temporary compatibility writer for
  main-pane content.

Verification:

```bash
cd apps/desktop
npm run build
npm run test:e2e:visual -- e2e/tile-unread-bug.spec.ts
```

Expected: both the hidden-tab and visible-split scenarios pass.

### Phase 3: Move chat view/runtime state behind pane ids

**Goal:** Let any chat pane load, resume, retry, compose, and interrupt itself.

- Extract the main chat’s global runtime state into per-pane runtime bindings.
- Port `$messages`, `$activeSessionId`, resume failure/exhaustion, and
  composer operations behind a `ChatPaneRuntime` keyed by pane id.
- Make `session-view.tsx` and `session-tile.tsx` render the same chat-pane
  component with a `paneId` prop. Remove their separate main-vs-tile data
  paths.
- Retain one gateway cache keyed by runtime id; panes subscribe to their
  selected runtime slice.

Tests:

- two chat panes can stream independently
- interrupting one pane does not change the other
- switching a pane’s session does not steal focus or overwrite the other pane

### Phase 4: Replace navigation with pane-targeted actions

**Goal:** Stop routing/resume code from writing a global selected session.

- Replace `setSelectedStoredSessionId` call sites with pane-targeted actions.
- Update sidebar, command palette, session picker, session switcher, project
  actions, preview, and profile-switch behavior.
- Sidebar rows use `$visibleSessionIds` for multi-row highlighting.
- Sidebar click focuses an already-visible session rather than loading a second
  copy.

Tests:

- sidebar has two highlighted rows for two visible split sessions
- clicking either highlighted row focuses its pane
- opening a hidden tab focuses/fronts it and clears unread
- creating a new chat only changes the target pane

### Phase 5: Remove `HashRouter` and route-derived session state

**Goal:** Remove the obsolete second content registry.

- Replace `HashRouter` in `src/main.tsx`.
- Remove `Routes`, `Route`, `Navigate`, `useNavigate`, `useLocation`,
  `useParams`, and `useSearchParams` from the desktop’s own navigation.
- Convert page navigation to `setPaneContent(paneId, { kind: 'page', page })`.
- Convert session navigation to `openSessionInPane`.
- Decide explicitly whether settings overlays remain overlay atoms or become
  page content; do not route them through a global page atom.
- Remove legacy session-route redirects and route-resume hooks.

Tests:

- app starts in a fresh `main` chat pane without a hash
- skills/settings/messaging open in the targeted pane
- session selection works with no URL mutation
- Electron secondary windows receive their pane content through window launch
  parameters, not hash routes

### Phase 6: Delete compatibility state and audit every former consumer

**Goal:** Finish the authority migration; leave no single-session shadow state.

- Delete `$selectedStoredSessionId`, `setSelectedStoredSessionId`, and all
  compatibility bridges.
- Delete route-specific tests and replace them with pane-content tests.
- Audit every former import site:
  - needs visible sessions → `$visibleSessionIds`
  - needs focused content → `$focusedPaneContent`
  - needs a specific pane’s session → `paneContent(paneId)`
  - needs to change content → pane-targeted action
- Search must return zero matches:
  ```bash
  rg '\$selectedStoredSessionId|setSelectedStoredSessionId|react-router-dom' apps/desktop/src
  ```

## Non-goals

- Making pane layouts URL-shareable. The desktop has no address bar; this adds
  complexity without a user-facing use case.
- Changing stored session identity or compression lineage semantics.
- Replacing the layout tree. It is the correct structural authority and should
  gain content ownership, not be replaced.
- Turning every overlay into a pane. Overlays remain transient UI state unless
  a concrete product need says otherwise.

## Risks and constraints

### Circular imports

`session-states.ts` imports from `session.ts`; `session.ts` must not import
back from `session-states.ts` because module-level listeners can run during a
partial cycle. Put visibility and unread-clearing ownership together in the
new pane-content store to keep imports one-directional.

### Runtime identity

Stored ids are durable; runtime ids are backend-process-scoped. Pane content
must store durable ids only. Runtime bindings must reset/re-resume after
reconnect/profile switch, exactly as current tiles do.

### Persisted layout

The layout tree currently persists tile placement. Pane content must persist
only the durable content needed to reconstruct a pane after restart. Never
persist runtime ids.

### Profile isolation

Content bindings, like current tiles, must scope to the active gateway profile.
A profile switch must not leave panes showing sessions from the previous
backend.

## Acceptance criteria

1. Two chat sessions in a side-by-side split are simultaneously highlighted in
   the sidebar.
2. A visible split session never receives an unread marker when it finishes.
3. A hidden inactive tab still receives an unread marker when it finishes.
4. Focusing a visible/hidden session clears its unread marker exactly once.
5. Sending, interrupting, retrying, and streaming are isolated per chat pane.
6. No `$selectedStoredSessionId`, `setSelectedStoredSessionId`, or
   `react-router-dom` imports remain in the desktop renderer after Phase 6.
7. Existing background-process/subagent sidebar E2E tests still pass.
