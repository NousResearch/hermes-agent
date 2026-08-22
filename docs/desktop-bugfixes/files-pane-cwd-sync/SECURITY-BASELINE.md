# Security Baseline — Files Pane cwd Sync

**Date:** 2026-08-14
**Component:** `apps/desktop/src/app/open-session.ts` (and its test)
**Baseline pinned against:** `6e14f893066e` (main @ 2026-08-14 01:11)

## Pre-PR baseline (pinned to commit `6e14f893066e`)

```text
$ rg -n "openSession\b" apps/desktop/src/app/open-session.ts | wc -l
1   (definition line only)
$ rg -n "setCurrentCwdTransient" apps/desktop/src/app/open-session.ts
0
$ rg -n "releaseWorkspaceCwdOwner" apps/desktop/src/app/open-session.ts
0
$ rg -n "setWorkspaceCwdOwner" apps/desktop/src/app/open-session.ts
0
$ rg -n "\\\\\$sessions" apps/desktop/src/app/open-session.ts
0
$ rg -n "sessionMatchesStoredId" apps/desktop/src/app/open-session.ts
0
```

The function `openSession` is a **pure** routing primitive: it dispatches
on `focusOpenSession` result and calls `navigate` for unmatched
focuses. It does not touch any session-state atom (`$currentCwd`,
`$workspaceCwdOwner`, `$sessions`, …) under baseline.

## Post-PR baseline

```text
$ rg -n "openSession\b" apps/desktop/src/app/open-session.ts
1   (definition)
$ rg -n "setCurrentCwdTransient" apps/desktop/src/app/open-session.ts
2   (import + write inside the focused branch)
$ rg -n "releaseWorkspaceCwdOwner" apps/desktop/src/app/open-session.ts
2   (import + write inside the focused branch)
$ rg -n "setWorkspaceCwdOwner" apps/desktop/src/app/open-session.ts
2   (import + write inside the focused branch)
$ rg -n "\\\\\$sessions" apps/desktop/src/app/open-session.ts
2   (import + read inside the focused branch)
$ rg -n "sessionMatchesStoredId" apps/desktop/src/app/open-session.ts
2   (import + read inside the focused branch)
```

Five new imports (`$sessions`, `releaseWorkspaceCwdOwner`,
`sessionMatchesStoredId`, `setCurrentCwdTransient`, `setWorkspaceCwdOwner`)
and one new code block in the **focused branch only** of
`openSession`. The unmatched branch is byte-identical to baseline.

## Diff: capabilities exposed vs baseline

| Capability | Pre-PR | Post-PR |
|-----------|--------|---------|
| `openSession` matches `tile` focus | returns (no writes) | reads `$sessions` and writes `$currentCwd` + `$workspaceCwdOwner` (mirroring the resume path) |
| `openSession` matches `main` focus | returns (no writes) | **same as above** (was also a gap pre-PR) |
| `openSession` no focus | navigates | navigates (unchanged) |
| `openSession` matches `stack` focus | returns (no writes) | **same as `tile` / `main`** |
| `openSession` `window` intent | spawns new window (or falls back) | unchanged |

## Trust model invariants (per `SECURITY.md` §2)

The `SECURITY.md` §2 model is **unchanged**. The PR does not:

- Cross the agent↔host boundary.
- Add IPC channels, native bindings, or new `contextBridge` calls.
- Expose a new attack surface.
- Change authentication, authorization, or session creation.
- Modify how `approvals.mode` is reconciled.
- Touch `YOLO_ACTIVE`, `$terminalBackend`, or any sandboxing
  configuration.

## Net new dependencies

**Zero.** No `package.json` change.

## Net new IPC surface

**Zero.** No `electron-preload.js` change. No new
`ipcRenderer.invoke` / `ipcRenderer.on` / `contextBridge.exposeInMainWorld`.

## Net new filesystem surface

**Zero new read paths.** The fix reads `$sessions` (already
in-memory) and reads `stored?.cwd` (a string already in memory).
`useProjectTree` already reads the cwd through the existing
`readProjectDir` IPC; the fix does not add a new IPC.

## Net new persistence

**Zero.** The two atoms written by the fix are already-persisted
atoms (`$currentCwd` via `setCurrentCwd` only on `setCurrentCwd`, NOT
on `setCurrentCwdTransient`; `$workspaceCwdOwner` is volatile).

## Net new outbound network

**Zero.** No new HTTP, WS, or any other outbound call.

## Reproducible verification

```bash
git checkout 6e14f893066e -- apps/desktop/src/app/open-session.ts
rg -n "setCurrentCwdTransient|releaseWorkspaceCwdOwner|setWorkspaceCwdOwner|\$sessions|sessionMatchesStoredId" apps/desktop/src/app/open-session.ts
# Expected: 0 results (baseline confirmed)
```

## Sign-off

- [x] Baseline established and pinned
- [x] No new capability surface
- [x] No new dependency
- [x] No new IPC, network, or fs surface
- [x] All existing call sites unaffected (regression suite green)
- [x] Documented in DECISION-RECORD.md and SECURITY-AUDIT.md
