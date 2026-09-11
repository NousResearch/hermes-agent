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

const _inFlightResumeByStoredSessionId = new Map<string, Promise<unknown>>()

export function singleFlightSessionResume<T>(storedSessionId: string, run: () => Promise<T>): Promise<T> {
  const existing = _inFlightResumeByStoredSessionId.get(storedSessionId)

  if (existing) {
    return existing as Promise<T>
  }

  // Promise.resolve().then(run) tolerates run() being synchronous, returning a
  // bare value, or throwing synchronously (test doubles and legacy callers do
  // all three) — a raw run().finally() would crash on a non-promise return.
  const flight = Promise.resolve()
    .then(run)
    .finally(() => {
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
