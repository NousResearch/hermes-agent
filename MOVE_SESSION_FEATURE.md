# Move session to… (desktop)

Feature contract + integration map. Re-implemented against the post-refactor
architecture after the original `74fbac717` was dropped when `main` was reset
to `origin/main` (the upstream refactor rewrote every file this feature
touched). This document exists so the feature's boundaries survive future
refactors — keep it in sync with any change to the files below.

## What it does

Adds a **"Move to…"** submenu to the session actions menu (sidebar row kebab
`⋯`, right-click context menu). The submenu lists the user's other profiles;
selecting one relocates the session's row from the current profile's `state.db`
into the target profile's `state.db` (copy session + messages +
`session_model_usage`, delete from source, **preserve session id**). The row
disappears from the current profile's view and appears under the target
profile (visible only when that profile is the active gateway profile).

Target of the move = **another profile** (cross-profile relocate), NOT cwd.

## Backend (must exist for the feature to function)

- `hermes_state.py` — `SessionDB.move_session_to_profile(self, session_id, target_profile_name, target_db_path)`.
  Copy session row + messages + `session_model_usage` into the target db,
  delete from source, keep the session id stable.
- `hermes_cli/web_server.py` — `PATCH /api/sessions/{session_id}` must accept a
  `move_to_profile` field (currently only `title`/`archived` are handled at
  the handler near line 10159; a 400 fires when neither known field is present,
  so the new field must be added there). Route through the owning `profile`'s
  `state.db` (profile-scoped request).

If either backend piece is missing, the desktop UI is dead — verify both before
claiming the feature works.

## Desktop integration map (file → change)

| Layer | File | Change |
|---|---|---|
| REST helper | `src/hermes.ts` | `moveSessionToProfile(id, targetProfile, profile?)` → `PATCH /api/sessions/{id}` with `{ move_to_profile: targetProfile }`, profile-scoped |
| Hook | `src/app/session/hooks/use-session-actions/index.ts` | `moveSessionToProfile` — optimistic remove from view + rollback on failure (mirror `archiveSession`) |
| Controller | `src/app/contrib/wiring.tsx` (`nextActions`, ~line 681) | `onMoveSessionToProfile: (sessionId, targetProfile) => void` |
| Sidebar types | `src/app/contrib/types.ts` (`SidebarActions`) | add `onMoveSessionToProfile` |
| Sidebar | `src/app/chat/sidebar/index.tsx` → `SidebarSurface` → `ChatSidebar` | thread the prop through `SidebarActions` |
| Row | `src/app/chat/sidebar/session-row.tsx` (both menu wrappers, ~lines 101-128) | pass `onMoveToProfile` into `SessionActionsMenu` + `SessionContextMenu` |
| Menu | `src/app/chat/sidebar/session-actions-menu.tsx` | extend `SessionActions` with `onMoveToProfile`; add a self-contained `MoveSubmenu` component (do NOT inline into the menu body) |
| MenuKit | `src/app/chat/sidebar/session-actions-menu.tsx` (`MenuKit`, ~line 120) | add `Sub`/`SubTrigger`/`SubContent` (mirror `split-submenu.tsx` `SplitMenuKit`) |
| Tile (optional) | `src/store/session-states.ts` + `src/app/contrib/hooks/use-session-tile-delegate.ts` | add `moveSessionToProfile` to `SessionTileDelegate` + wire, if the tile-tab `⋯` menu should also move |
| i18n | `src/i18n/{en,types,zh,ja,zh-hant}.ts` (`sidebar.row` + `desktop`) | add `moveTo`, `movedToProfile`, `moveFailed` |

## Data flow

Target profiles come from the **`$profiles`** atom (`src/store/profile.ts:35`,
populated by `refreshProfiles()` → `GET /api/profiles`). In `useSessionActions`,
read `useStore($profiles)`, filter out the current `session.profile` (already
passed as the `SessionActions.profile` prop), map the rest into `SubContent`.

## Hard rules (violating these is how the feature breaks)

1. **Profile routing is mandatory.** Every session mutation in `hermes.ts` takes an
   owning `profile` and routes via `request.profile`. A move of a session owned by
   a non-active profile MUST pass that `profile`, or the backend opens the wrong
   `state.db`. The menu already has `session.profile` — wire it through.
2. **Optimistic update + rollback.** Mirror `archiveSession`: optimistically remove
   the row from the current view → call REST → on failure restore via
   `setSessions(prev => [removed, ...prev])` + `untombstoneSessions` + `notifyError`.
   A failed move must not strand the row.
3. **Three session identities** (runtime / stored / lineage-root). Move is a
   persisted-row operation — pass the **stored** `sessionId` (`SessionActions.sessionId`).
4. **Keep the UI submenu isolated.** `MoveSubmenu` is its own component imported
   by the menu, not inlined. A refactor of the actions menu should then only touch
   one import line + one call site, not the feature body.
5. **Backend method stays isolated.** `move_session_to_profile` is a standalone
   method on `SessionDB`; do not fold it into adjacent mutation paths.

## Regression guards

- Tests for `moveSessionToProfile` assert the **contract** (optimistic remove +
  rollback on REST failure + correct `move_to_profile` payload), not the internals
  of `use-session-actions/index.ts`. Contract tests survive a hook refactor.
- Backend: a test that `PATCH /api/sessions/{id}` with `move_to_profile`
  relocates the row and preserves the session id.

## Status

- Last re-implementation: against `origin/main` post-refactor (see git log for
  the implementing commit). Original `74fbac717` is an orphaned, pre-refactor
  commit — DO NOT cherry-pick it; it will not apply to the rewritten files.
- Not sent upstream. Lives on a local branch / working tree only.
