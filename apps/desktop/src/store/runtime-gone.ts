import { $activeSessionId, requestSessionResume } from './session'
import { $sessionStates, $sessionTiles, unbindTileRuntime } from './session-states'

/** Heal a session view whose bound runtime id the gateway no longer holds.
 *
 *  The desktop learns a runtime is gone through two channels, and only one of
 *  them ever repaired anything:
 *
 *  - PUSH — `session.reclaimed`. `gateway-event/lifecycle.ts` drops the cached
 *    state and calls `unbindTileRuntime`, which re-arms SessionTilePane's
 *    resume effect (gated on `!runtimeId`) so the tile rebinds a fresh runtime
 *    from the intact stored row.
 *  - PULL — a session-scoped RPC rejected `4001 "session not found"`. The
 *    gateway logs "client should resume the stored session" precisely because
 *    this is the terminal verdict; `_sess_nowait` has no other way to say it.
 *
 *  Every user action already honours the pull verdict: submit, slash, rewind,
 *  interrupt and the attaches run through `withSessionNotFoundResume`, which
 *  resumes the stored id and rebinds. The background pollers — the only callers
 *  that run while the user is NOT acting — did not. They either re-sent the dead
 *  id forever or (with the gone-latch) went silent against it, and in both cases
 *  the view stayed bound to a phantom runtime for the rest of its life.
 *
 *  That gap is only reachable through the pull channel. The push channel cannot
 *  cover a runtime that died with a previous app process (boot-restore), was
 *  reaped while this client was disconnected, or was reaped by a remote gateway
 *  this renderer had not yet dialled — the broadcast has no live listener. In
 *  those cases the 4001 is the *only* notice that ever arrives.
 *
 *  So route it to the same recovery the broadcast drives. Both surfaces that can
 *  hold a binding get their existing re-arm lever pulled:
 *
 *  - Tiles: `unbindTileRuntime` (SessionTilePane's resume effect refires).
 *  - The primary chat: `requestSessionResume`, the explicit-request lever. Its
 *    route-resume effect skips on `alreadyActive` — route === selected and the
 *    cached runtime === the active one — which stays true forever against a
 *    dead id, and only `explicitlyRequested` bypasses it without a reconnect.
 */

/** Runtime ids already healed. A reaped runtime id is dead permanently, and a
 *  successful heal binds a NEW one, so this never needs clearing: a second heal
 *  for the same id could only come from a duplicate report of the same death. */
const healedRuntimes = new Set<string>()

/** Consecutive heals per stored session id, reset by {@link noteRuntimeAlive}.
 *  A backend that reaps as fast as we resume would otherwise turn this into the
 *  very storm it exists to stop — one resume per poll tick, forever. Cap it and
 *  let the user's next action (which carries its own recovery) take over. */
const healsByStoredId = new Map<string, number>()

/** Enough to ride out a reap that races a resume, low enough that a backend
 *  reaping on sight cannot be turned into a resume loop. */
const MAX_CONSECUTIVE_HEALS = 3

/** Resolve the durable identity behind a runtime id. The cached session state is
 *  authoritative; a tile that resumed before the state landed is the fallback. */
function storedIdForRuntime(runtimeId: string): null | string {
  const cached = $sessionStates.get()[runtimeId]?.storedSessionId

  if (cached) {
    return cached
  }

  return $sessionTiles.get().find(tile => tile.runtimeId === runtimeId)?.storedSessionId ?? null
}

/** A poll against `runtimeId` succeeded — the binding is healthy, so the stored
 *  session's heal budget is spent on real deaths only, not on one bad stretch. */
export function noteRuntimeAlive(runtimeId: string): void {
  if (healsByStoredId.size === 0) {
    return
  }

  const storedId = storedIdForRuntime(runtimeId)

  if (storedId) {
    healsByStoredId.delete(storedId)
  }
}

/** Report the gateway's terminal verdict for `runtimeId` and re-arm whichever
 *  surface holds the binding. Returns true when a recovery was requested.
 *
 *  Safe to call on every 4001: it is idempotent per runtime id, and the levers
 *  it pulls are themselves bounded (the tile resume effect single-flights
 *  through `resumingRef` and latches its own terminal errors; the route resume
 *  backs off over `MAX_RESUME_RETRIES`). */
export function markRuntimeGone(runtimeId: string): boolean {
  if (!runtimeId || healedRuntimes.has(runtimeId)) {
    return false
  }

  healedRuntimes.add(runtimeId)

  const storedId = storedIdForRuntime(runtimeId)

  if (!storedId) {
    // No durable identity to resume from — a never-persisted draft, or a
    // runtime whose view is already gone. Latching alone is the whole fix.
    return false
  }

  const heals = healsByStoredId.get(storedId) ?? 0

  if (heals >= MAX_CONSECUTIVE_HEALS) {
    return false
  }

  healsByStoredId.set(storedId, heals + 1)

  // Tiles: clearing the binding re-arms the resume effect. A no-op when no tile
  // holds this runtime.
  unbindTileRuntime(runtimeId)

  // The primary chat: only an explicit request gets past its `alreadyActive`
  // skip. Scoped to the runtime the primary is actually showing, so a tile's
  // dead runtime never navigates the main view.
  if ($activeSessionId.get() === runtimeId) {
    requestSessionResume(storedId)
  }

  return true
}

/** Tests only: forget every heal so cases start from a clean slate. */
export function resetRuntimeGoneHealing(): void {
  healedRuntimes.clear()
  healsByStoredId.clear()
}
