/**
 * Single-flight guard for `session.resume`, keyed by STORED session id.
 *
 * After sleep/wake or a reconnect, many independent surfaces discover the same
 * dead runtime at once — submit recovery, slash/rewind recovery, tile resumes,
 * the route resolver — and each used to fire its own `session.resume` for the
 * same durable conversation. The gateway happily mints a runtime per call and
 * the losers become orphans for the reaper (#91276 storm).
 *
 * Module-level so EVERY call site in the window shares one in-flight promise
 * per stored id, no matter which hook instance it lives in. All participating
 * callers resolve to a `session.resume`-shaped response (an object carrying
 * `session_id`); joiners receive whatever the winning call returns.
 */

import { withTimeout } from '@/lib/with-timeout'

const _inFlightResumeByStoredSessionId = new Map<string, Promise<unknown>>()

/**
 * Ceiling on how long ONE flight may hold the shared slot.
 *
 * Sharing the promise is what makes an unsettled resume so expensive: the
 * entry is only released by `.finally()`, so while it hangs every later
 * caller for that stored id joins the same dead promise and the conversation
 * is unrecoverable for the life of the window. The deadline exists to break
 * that wedge, NOT to make a slow resume feel responsive — the RPC's own 30s
 * budget already covers responsiveness.
 *
 * Derived from the longest LEGITIMATE settlement rather than picked round, so
 * a slow-but-healthy resume is never aborted. `run()` bodies resolve the
 * owning profile before they send anything, and `resolveStoredSession()`
 * probes backends sequentially on a cache miss:
 *
 *   30s  active-profile `getSession` probe    (Electron DEFAULT_FETCH_TIMEOUT_MS)
 * + 30s  one cross-profile `getSession` probe (same budget, per profile)
 * + 30s  the `session.resume` RPC             (HermesGateway DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS)
 * = 90s
 *
 * An install with more profiles can still exceed this, and that is deliberate:
 * past this point the flight has held the slot longer than any single healthy
 * recovery ever needs, and breaking it is better than stranding the window.
 */
export const SESSION_RESUME_SETTLEMENT_TIMEOUT_MS = 90_000

/** Best-effort `session_id` off a `session.resume`-shaped response. */
function resumedRuntimeId(result: unknown): string {
  if (typeof result !== 'object' || result === null) {
    return ''
  }

  const sessionId = (result as { session_id?: unknown }).session_id

  return typeof sessionId === 'string' ? sessionId : ''
}

export function singleFlightSessionResume<T>(
  storedSessionId: string,
  run: () => Promise<T>,
  timeoutMs = SESSION_RESUME_SETTLEMENT_TIMEOUT_MS
): Promise<T> {
  const existing = _inFlightResumeByStoredSessionId.get(storedSessionId)

  if (existing) {
    return existing as Promise<T>
  }

  // Promise.resolve().then(run) tolerates run() being synchronous, returning a
  // bare value, or throwing synchronously (test doubles and legacy callers do
  // all three) — a raw run().finally() would crash on a non-promise return.
  const work = Promise.resolve().then(run)

  // withTimeout does NOT cancel the work it bounds, and here the straggler is
  // not inert: `session.resume` can still land a REAL runtime, registered on
  // the gateway, that no client is holding. Dropping it on the floor is the
  // exact orphan-per-resume shape this module exists to prevent (#91276), so
  // hand a late arrival to the same recovered-runtime cache the drift-abort
  // path uses and let the next resume-shaped action adopt it.
  const adoptStraggler = () => {
    void work.then(
      result => {
        // A newer flight already owns this stored id — its caller will adopt
        // whatever it returns, so caching an older runtime here would just
        // aim the next action at the wrong one.
        if (_inFlightResumeByStoredSessionId.has(storedSessionId)) {
          return
        }

        registerRecoveredRuntime(storedSessionId, resumedRuntimeId(result))
      },
      // A straggler that failed minted nothing — there is nothing to adopt.
      () => undefined
    )
  }

  const flight = withTimeout(
    work,
    timeoutMs,
    `Timed out resuming session ${storedSessionId}`,
    adoptStraggler
  ).finally(() => {
    if (_inFlightResumeByStoredSessionId.get(storedSessionId) === flight) {
      _inFlightResumeByStoredSessionId.delete(storedSessionId)
    }
  })

  _inFlightResumeByStoredSessionId.set(storedSessionId, flight)

  return flight
}

/**
 * Adopt-or-reuse cache for recovered runtimes a drift-abort walked away from.
 *
 * A recovery resume can succeed while the caller's drift check says the user
 * moved on (SessionRecoveryAborted). The freshly-minted runtime is REAL and
 * registered on the gateway; abandoning the id client-side strands it for the
 * orphan reaper AND makes the next action for the same stored session mint yet
 * another runtime. When adoption (rebinding the caller's runtime ref via
 * onRecovered/onRuntimeRecovered) is wrong — the user is elsewhere — record it
 * here so the next resume-shaped action reuses it instead of re-minting.
 */
const _recoveredRuntimeByStoredSessionId = new Map<string, string>()

export function registerRecoveredRuntime(storedSessionId: string, runtimeId: string): void {
  if (storedSessionId && runtimeId) {
    _recoveredRuntimeByStoredSessionId.set(storedSessionId, runtimeId)
  }
}

/**
 * Consume a previously-abandoned recovered runtime for this stored session.
 * Take-semantics: the entry is removed so a dead cached id can only cost one
 * bounded retry, never a loop. `deadRuntimeId` skips (and drops) the entry
 * when the caller already knows that exact runtime is dead.
 */
export function takeRecoveredRuntime(storedSessionId: string, deadRuntimeId?: null | string): string | undefined {
  const cached = _recoveredRuntimeByStoredSessionId.get(storedSessionId)

  if (cached === undefined) {
    return undefined
  }

  _recoveredRuntimeByStoredSessionId.delete(storedSessionId)

  return deadRuntimeId && cached === deadRuntimeId ? undefined : cached
}

/** Test seam: reset all module-level single-flight/recovery state. */
export function clearSingleFlightSessionResumeState(): void {
  _inFlightResumeByStoredSessionId.clear()
  _recoveredRuntimeByStoredSessionId.clear()
}
