import { $activeSessionId, requestSessionResume } from './session'
import { $sessionStates, $sessionTiles, unbindTileRuntime } from './session-states'

/** Session ids the gateway has told us are gone. A session-scoped RPC against a
 *  runtime the gateway no longer holds fails 4001 "session not found" — terminal
 *  for THIS runtime id, not a transient socket loss.
 *
 *  Shared by every background poller (process.list, approval.pending, goal
 *  status). One set, one clear path: a fresh-runtime rebind calls
 *  {@link resetBackgroundPollingGuard} and every poller resumes. */
const goneSessions = new Set<string>()

/** Gateway JSON-RPC code for "session not found" (tui_gateway `_sess_nowait`). */
const GATEWAY_SESSION_NOT_FOUND_CODE = 4001

/** A gone session is unrecoverable for THIS runtime id; a timeout or transport
 *  blip is not. Only the former may stop a poll — misclassifying a transient
 *  failure would silently freeze a healthy session.
 *
 *  Match the gateway's 4001 code when the error carries one. The message
 *  fallback survives only for errors with no numeric code at all. */
export function isSessionGoneForBackgroundPolling(error: unknown): boolean {
  const code =
    error && typeof error === 'object' && typeof (error as { code?: unknown }).code === 'number'
      ? (error as { code: number }).code
      : undefined

  if (code !== undefined) {
    return code === GATEWAY_SESSION_NOT_FOUND_CODE
  }

  // Codeless errors: the frame's structure was lost somewhere (IPC bridge,
  // wrapped rethrow). Accept only a bare "session not found" body — a tool or
  // report string that merely mentions the phrase must not latch a live runtime.
  const message = (error instanceof Error ? error.message : String(error ?? ''))
    .trim()
    .replace(/^Error invoking remote method '[^']+':\s*Error:\s*/i, '')
    .replace(/^Error:\s*/i, '')

  return /^(?:4001\s*[:,-]?\s*)?session not found[.!]?$/i.test(message)
}

export function isSessionGone(sid: string): boolean {
  return goneSessions.has(sid)
}

/** Latch `sid` off and heal the bound view. Safe to call on every 4001. */
export function markSessionGone(sid: string): void {
  if (!sid) {
    return
  }

  goneSessions.add(sid)
  markRuntimeGone(sid)
}

/** Clear the gone-latch. Called with a session id when a fresh runtime binds to
 *  it (so polling resumes), or with no argument to reset everything (tests /
 *  gateway reconnect). */
export function resetBackgroundPollingGuard(sid?: string): void {
  if (sid) {
    goneSessions.delete(sid)

    return
  }

  goneSessions.clear()
}

/** Clear the gone-latch for the ids a successful `session.resume` /
 *  `session.activate` just rebound — the stored id it was asked for and the
 *  runtime id it answered with.
 *
 *  A socket reconnect is NOT a rebind: the backend may have reaped the old
 *  runtime, and merely reopening a WebSocket does not make that id valid
 *  again. Only a successful resume/activate response is proof the runtime can
 *  be targeted, so this is the one seam that un-latches per id. */
export function resetBackgroundPollingGuardAfterRebind(
  method: string,
  params: Record<string, unknown>,
  result: unknown
): void {
  if (method !== 'session.activate' && method !== 'session.resume') {
    return
  }

  const candidates = [params.session_id, (result as { session_id?: unknown } | null)?.session_id]

  for (const value of candidates) {
    if (typeof value === 'string' && value.trim()) {
      goneSessions.delete(value.trim())
    }
  }
}

/** Heal a session view whose bound runtime id the gateway no longer holds.
 *
 *  The desktop learns a runtime is gone through two channels:
 *
 *  - PUSH — `session.reclaimed`. `gateway-event/lifecycle.ts` calls
 *    {@link markRuntimeGone} (same levers as the pull path) before dropping
 *    the cached state, so the primary chat resumes instead of sitting on the
 *    dead runtime until the user types.
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
 *  The pull channel is the only notice when the client missed the broadcast
 *  (boot-restore, disconnect, remote gateway). Both surfaces that can hold a
 *  binding get their existing re-arm lever pulled:
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
