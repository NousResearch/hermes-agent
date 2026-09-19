# Decision Record — Files Pane cwd Sync on Focused Session Switch

**Date:** 2026-08-14
**Scope:** `apps/desktop/src/app/open-session.ts`
**Related:** #71254, #77496, #80213, #76696 (historical cwd-ownership bugs)
**Type:** UI bug fix (renderer-side state synchronization)

## Problem

Clicking a sidebar session row whose conversation is **already open** in
main or as a tile did not update the right-side **Files** pane to that
session's directory. The pane kept showing whichever folder the previously
loaded chat owned, and only caught up on the next backend
`session.info` heartbeat (i.e. the next user message).

## Why this matters

The Files pane is the only direct lens onto a session's working tree.
When a user paged between projects the pane silently lied, which made
"the workspace you can see" and "the conversation you clicked"
disagree. Long-running work in the wrong tree produced file paths the
backend couldn't resolve, then surfaced later as confusing
"read-error" placeholders.

## What we considered

### Option A — Patch `focusOpenSession` to also write cwd
**Rejected.** `focusOpenSession` is a pure UI primitive
(`revealTreePane` / `noteActiveTreeGroup`); it has no knowledge of
`$sessions` or `$currentCwd`. Adding it would couple a low-level pane
focus helper to the session/workspace atoms, which is a layering
violation. Also affects every caller of `focusOpenSession` (route-resume
self-heal, command palette, session refs) — too broad a blast radius
for a UI focus helper.

### Option B — Patch `resumeSession` to fast-path focused hits
**Rejected.** `resumeSession` is the cold/warm resume pipeline; it
rightly assumes "no live state exists for this session yet" and runs
through REST prefetch, transcript reconciliation, and the
`session.activate` RPC. Reusing it for a tab that's already on screen
would trigger redundant network calls and overwrite in-memory caches.

### Option C (chosen) — Mirror the stored-row cwd at the `openSession`
focus hit, matching `applyStoredSessionPreviewRuntimeInfo`'s semantics

`openSession` already has the right surface:

- It already runs `focusOpenSession(storedSessionId)` and inspects the
  result. A focused hit is the one place where `openSession` *skips*
  `resumeSession`, so the cwd gap is local to this branch.
- The sidebar row's compact projection (the same `SessionInfo.cwd` that
  `applyStoredSessionPreviewRuntimeInfo` reads in
  `use-session-actions/index.ts:881`) is already in `$sessions`. A
  targeted `find(s => sessionMatchesStoredId(s, storedSessionId))` is
  the canonical lookup.
- Reusing the same `storedCwd ? setCurrentCwdTransient + setWorkspaceCwdOwner : releaseWorkspaceCwdOwner`
  decision tree means the focus path and the resume path can never
  diverge in ownership semantics — the marker can only end up in one
  of three stable states (this session, no session, or the unowned
  sentinel from `releaseWorkspaceCwdOwner`), and the right-sidebar's
  `useProjectTree(cwd)` consumer is unaffected.

We chose **not** to import `applyStoredSessionPreviewRuntimeInfo` from
`session-states/utils.ts` directly. That helper lives inside the resume
hook and pulls in a chain of dependencies (`useStore` callbacks,
reconciliation helpers) that would risk a circular import when the
focus path imports it. The four-line inline copy is intentional: it
keeps the two paths in lock-step semantically while keeping the import
graph clean. If the two diverge later, a refactor that lifts the inline
block into a shared store-level helper is the right move.

## What we did NOT change

- `right-sidebar/index.tsx` — the consumer was correct, only the
  producer (`openSession`) was missing the write.
- `use-session-actions/index.ts:884` — the cold-resume path already
  called `applyStoredSessionPreviewRuntimeInfo`; the focus path now
  mirrors the same effect.
- `$workspaceCwdOwner` semantics — the sentinel
  (`WORKSPACE_CWD_UNOWNED`) is still owned by the resume/follow
  pipeline. The focus path only ever writes either a real id or a
  `release`; it never fabricates the unowned marker.
- Test count of the existing `openSession` test file. The 23
  pre-existing tests still pass; 4 new tests pin the new behaviour.
- IPC, native bindings, `package.json`, `app.asar` layout.

## Invariants preserved

- **#71254 (workspace-derived surfaces hide stale cwd during the
  resume gap).** A focused hit does not erase `$currentCwd` — it only
  updates it when the new session's stored `cwd` is present. Detached
  sessions still release ownership instead of writing `''`.
- **#77496 / #80213 (don't claim the previous chat's folder as the
  user's chosen workspace).** `openSession` uses `setCurrentCwdTransient`
  — never `setCurrentCwd` — so the persisted "last chosen cwd" key
  isn't bumped by a sidebar click. That key is reserved for explicit
  user actions (folder pick, project entry, worktree move).
- **Cache-, alternation-, invariant-safety.** No prompt cache impact
  (renderer-only mutation). No message role alternation impact (no
  message stream involvement).
- **Prompt-caching.** The system prompt is not touched.

## Risk

**Low.** A 10-line patch in a single function, mirrored semantics with
the existing cold-resume path, fully covered by new unit tests, with
type-safe store accessors and the same call shape that
`applyStoredSessionPreviewRuntimeInfo` already uses 200+ lines away in
the same module tree. The 223-test regression sweep across
`right-sidebar`, `chat/sidebar`, and `session-ref-open` confirms no
sibling path was perturbed.

## Rollback

`git revert` the commit. No data migration, no schema change, no
persisted-state impact. Reverting returns the pane to the pre-fix
"stale until next session.info" behaviour, which is the documented
status quo prior to this PR.
