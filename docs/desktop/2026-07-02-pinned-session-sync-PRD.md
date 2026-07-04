# PRD: Server-Side Pinned Sessions (fix client-only pin desync)

- **Status:** v0.2 — pass-1 AWC folded (6 RCs); light-review track, ready to build after B1 merges
- **Author:** Apollo, 2026-07-02 · **Requested:** Ace ("fixing the desync is important. it's
  pretty bad currently.")
- **Repo:** `Kyzcreig/hermes-agent`. **Sequencing:** build AFTER swarm lane B1
  (`t_8da3d890`) merges — both touch `tui_gateway/server.py`.

## 1. Problem

Pins are client-only localStorage: `$pinnedSessionIds` →
`persistentAtom('hermes.desktop.pinnedSessions')` (`apps/desktop/src/store/layout.ts:19,69`;
mutations at `:295-321`; pin id via `sessionPinId()`). Every device keeps its own divergent
pin set — MBP, Studio, dashboard-in-browser never agree ("pretty bad currently"). Nothing
syncs because the server has no concept of a pin.

## 2. Fix (option A — approved by Ace over local-sync hacks)

Server-side `pinned` flag, mirroring the existing `archived` column and `session.title` RPC
shape:

1. **Schema:** `pinned INTEGER NOT NULL DEFAULT 0` on `sessions` — rides the existing
   auto-migration (`hermes_state.py:1276-1321` ALTER TABLE ADD COLUMN path; `archived` at
   `:741` is the precedent). No manual migration step. NOTE: this is a SCHEMA ADDITION to
   state.db performed by the app's own migration machinery on startup — it is NOT the
   forbidden vacuum/prune/mutation class (Ace's "leave state.db alone" = no maintenance ops
   on the data; additive app-owned migration is how every column got there).
2. **hermes_state:** `set_session_pinned(key, bool)` + include `pinned` in
   `list_sessions_rich` (`:2701`) row dict.
3. **Gateway RPC (RC-3: ONE rpc, decided):** `session.pin {session_id, pinned: bool}` —
   mirrors `set_session_pinned(key, bool)` 1:1, halves the surface. Modeled on
   `session.title` (`tui_gateway/server.py:5886`): resolve via `_sess_nowait` (same
   structured error contract on miss — AC-4 asserts THAT code, not "no garble"),
   `_get_db()` guard (5007 on unavailable), write, then `_emit_session_info_for_session`.
   **Authz (RC-security):** same session-resolution/authz path as `session.title` — the RPC
   operates only on sessions the caller's authenticated gateway connection can already
   address; no new enumeration surface.
   **Echo-safety (RC-6):** the originator's derived-view update must be idempotent against
   its own optimistic write (emit races local action → same final state, no oscillation).
4. **`session.list`:** return `pinned` per session (it already projects
   `list_sessions_rich`).
5. **Desktop client:** `$pinnedSessionIds` becomes a derived view of session data; pin/unpin
   actions call the RPC. **Grep-and-kill ALL direct localStorage writers** (`layout.ts:295-321`
   are the known three; the cutover fails if any imperative mutation site survives).
   **Migration (RC-1: set-membership ADD, never set-replace):** on first run with a
   pin-capable backend, each local pin is pushed as a per-session idempotent
   `session.pin {pinned:true}` — NEVER a bulk replace of the server pin set (a read-union-
   write-back would clobber a concurrent device's pin between read and write). Then mark the
   local store migrated. **Server truth wins thereafter (RC-2):** on reconnect/`session.list`
   the derived view is rebuilt from server data, overwriting the local fallback — an unpin on
   device A cannot be resurrected by device B's stale localStorage; the fallback is READ only
   when the backend is pin-incapable, and never re-pushed. **Observability (RC-SRE):** one log
   line with migration push count, so a silent merge failure is visible.
6. **Sidebar ordering** unchanged (client already sorts pinned-first from the atom; it now
   sorts from server data).

## 3. Non-Goals
- No pin ordering/rank (a flag, not a list position) — matches current UX.
- No per-device pins (the desync IS the bug; pins are account-global by design).
- No dashboard-web UI work beyond what falls out of `session.list` carrying the flag.

## 4. Acceptance
- AC-1: pin on MBP → visible pinned on Studio desktop + dashboard within one
  session-info event (no restart).
- AC-2: migration — device with local pins {a,b}, another with {b,c} → server ends with
  {a,b,c}; no pin lost, no duplicate. **Including CONCURRENT first-run from both devices
  (RC-1/RC-5): per-session idempotent adds interleaved in any order converge to the union.**
- AC-2b (RC-2): unpin on device A → gone on device B after B reconnects (stale local
  fallback must not resurrect it).
- AC-3: `pinned` survives restart (DB roundtrip test, per persistence-roundtrip discipline:
  write → reopen DB → read).
- AC-4 (RC-4): pin RPC on a nonexistent/dead session returns the SAME structured error
  code/shape `session.title` returns on `_sess_nowait` miss — test asserts the exact code,
  not merely "didn't crash".
- AC-5: gateway tests + desktop typecheck/vitest green; migration idempotent (second startup
  no-ops).

## 5. Risks
- R1: `state.db` sensitivity → the change is one additive column via the app's own migration
  path; INV: no row deletion/rewrite; rollback = ignore the column (default 0).
- R2: server.py collision with B1 → sequenced after B1 merge (stated above).

## Review Log

### Pass 1 — Opus (claude-api-proxy), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.2
Zero blockers ("genuinely additive, migration precedent real, rollback sound"). 6 RCs folded:
RC-1 migration = per-session idempotent ADD never set-replace (race-proof); RC-2 unpin
propagation = server-truth-wins on reconnect + AC-2b; RC-3 single `session.pin {pinned}` RPC
decided; RC-4 AC-4 asserts exact error contract; RC-5 concurrent first-run in AC-2; RC-6
originator echo idempotence. Lens folds: authz scoped to session.title's resolution path;
migration-count log line; grep-and-kill all direct localStorage writers.
Light-review track: converged in 1 pass, build after swarm lane B1 merges.
