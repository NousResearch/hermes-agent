# Security Audit — Files Pane cwd Sync on Focused Session Switch

**Date:** 2026-08-14
**Scope:** `apps/desktop/src/app/open-session.ts` + `open-session.test.ts`
**Auditor:** x7peeps (PR author)
**Reviewer:** (pending — open-reviewer)
**PR:** (pending — see PR URL once opened)

## TL;DR

**No new security boundary introduced.** This PR is a renderer-side
state synchronization fix for the desktop GUI. It does not extend the
IPC, native, gateway, or network attack surface. The trust model in
`SECURITY.md` §2 is unchanged.

## Trust boundary review (per `SECURITY.md` §2)

Hermes's load-bearing security boundary is the **agent↔host** boundary:
the agent executes commands on the user's machine. Per `SECURITY.md` §3,
anything in this PR that crosses that boundary is in-scope for the
security review process.

| Boundary                          | Affected? | Notes |
|-----------------------------------|-----------|-------|
| agent ↔ host (CLI / process spawn) | **No**    | No new process spawn, no new shell call, no new fs read/write outside what `useProjectTree` already does. |
| renderer ↔ Electron (preload bridge) | **No**    | No new `contextBridge` surface, no new `ipcRenderer.invoke`, no new event listener. The change is in the renderer React tree only. |
| renderer ↔ Hermes gateway (WS)    | **No**    | No new RPC, no new event subscription, no payload change. |
| session data on disk              | **No**    | `setCurrentCwdTransient` is the same atom the resume path already uses 200+ lines away in `use-session-actions/index.ts:721`. The atom is renderer-only. |
| profile / gateway config           | **No**    | No new config keys, no schema change. |
| cross-window message bus          | **No**    | `nanostores` is single-window; cross-window sync for `$currentCwd` already exists and is not touched. |

## Privacy review (PII)

- No PII is collected, transmitted, or persisted by this change.
- `SessionInfo.cwd` is an absolute filesystem path. The PR reads
  exactly the same path that the sidebar row already displays in the
  Projects overview (`path`). The new read site is in the renderer
  process; the value never leaves the renderer; it never crosses the
  IPC bridge. **Equivalent privacy posture to the existing
  `applyStoredSessionPreviewRuntimeInfo` call site at
  `use-session-actions/index.ts:881`.**

## Prompt-cache review (per `AGENTS.md`)

- The system prompt is not touched.
- The conversation turn history is not touched.
- No new `<system>` / `<user>` message is injected.
- The atom write happens before any user-visible render commits, so the
  "byte-stable for the life of a conversation" property is unaffected.

## Failure mode analysis

What happens if the inline `setCurrentCwdTransient + setWorkspaceCwdOwner`
pair is called with a malicious / malformed `SessionInfo.cwd` value?

- `SessionInfo.cwd` originates from the gateway's `projects.list` /
  `session.resume` response and is already used to drive the right
  sidebar's `ProjectBackRow` label and the project menu's
  "move session to project" submenu. The renderer already trusts
  this value for UI labelling.
- The value is `.trim()`-ed; an empty string falls into the
  `releaseWorkspaceCwdOwner()` branch, which is the same code path the
  resume path takes for detached sessions. The Files pane's
  `useProjectTree` then receives `''` and shows the existing
  `noProjectTitle` empty state.
- The `setWorkspaceCwdOwner` call receives the same `storedSessionId`
  string the resume path passes (line 1101 of utils.ts). It cannot
  escape the project's React-tree ownership model.

No injection, no path-traversal, no privilege escalation, no data
exfiltration vector introduced.

## Threat model delta

**None.** The change is a strict improvement to renderer-side
synchronization between two renderer-resident atoms (`$currentCwd` and
`$workspaceCwdOwner`) that were already mutated by a different
code path. We are plugging a gap, not opening a new surface.

## Automated security checks (run by upstream CI)

| Check                                 | Status |
|---------------------------------------|--------|
| `bandit` (Python)                      | N/A — no Python files changed |
| `npm audit` (Node)                    | Will run on CI; no new deps added |
| `npm run lint` (eslint)                | Will run on CI; no new lint rules needed |
| `npm run typecheck`                    | Run locally — 0 errors |
| CodeQL / `security-audit.yml` (upstream) | Will run on PR; no new query triggers expected |
| `SECURITY.md` §2 boundary diagram      | Unchanged — see table above |

## Items intentionally NOT included

- No new security advisory. This is a UX-correctness fix, not a
  vulnerability.
- No new entry in `docs/SECURITY-MODEL.md` — the model in
  `SECURITY.md` is unchanged.
- No new test category. The four new unit tests assert behaviour
  contracts (invariant-style), not snapshots, per the project
  guideline against change-detector tests.

## Sign-off checklist

- [x] No new IPC surface
- [x] No new native binding
- [x] No new gateway RPC
- [x] No new persisted state
- [x] No new system prompt content
- [x] No new telemetry
- [x] No new dependency
- [x] No new file system access pattern
- [x] PII review: no change
- [x] Failure mode: bounded
