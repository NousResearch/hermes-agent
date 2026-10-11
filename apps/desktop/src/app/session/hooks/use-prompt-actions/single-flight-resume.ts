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

// Every session.resume entry point converges on this module-level flight, and
// the `run()` body is NOT just the RPC: it awaits resolveSessionProfile first
// (use-prompt-actions/utils.ts, resolve-target-session.ts, submit.ts), whose
// resolveStoredSession ladder probes the active profile and then every other
// configured profile sequentially — one bounded 30s getSession each (Electron
// DEFAULT_FETCH_TIMEOUT_MS, hardening.ts) — before the 30s session.resume RPC
// (DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS, api/client.ts). A deadline sized to a
// single-profile resume (PR 95926's original 35s) therefore aborts a
// slow-but-legitimate multi-profile recovery. Derive the ceiling from that
// ladder instead: one 30s window per configured profile, plus one for the RPC.
// The profile count is read per-flight (not at module load) through a lazy
// import: @/store/profile is statically entangled with @/store/session and
// this module's callers, so a static import would close a cycle.
const RESUME_PROBE_WINDOW_MS = 30_000

let _profileCountOverride: (() => number) | null = null

/** Test seam: pin the profile count the ceiling is derived from. */
export function setSessionResumeProfileCountOverride(count: (() => number) | null): void {
  _profileCountOverride = count
}

async function sessionResumeSettlementTimeoutMs(): Promise<number> {
  let count: number

  if (_profileCountOverride) {
    count = _profileCountOverride()
  } else {
    try {
      const { $profiles } = await import('@/store/profile')

      count = $profiles.get().length
    } catch {
      // Profile store unavailable (isolated unit-test setups): assume the
      // single-profile shape — one probe window plus the RPC.
      count = 1
    }
  }

  return RESUME_PROBE_WINDOW_MS * (Math.max(1, count) + 1)
}

/** The live id a settled `session.resume`-shaped outcome carries, if any. */
function sessionResumeOutcomeSessionId(outcome: unknown): string | undefined {
  const id = (outcome as { session_id?: unknown } | null | undefined)?.session_id

  return typeof id === 'string' && id ? id : undefined
}

export function singleFlightSessionResume<T>(storedSessionId: string, run: () => Promise<T>): Promise<T> {
  const existing = _inFlightResumeByStoredSessionId.get(storedSessionId)

  if (existing) {
    return existing as Promise<T>
  }

  // Promise.resolve().then(run) tolerates run() being synchronous, returning a
  // bare value, or throwing synchronously (test doubles and legacy callers do
  // all three) — a raw run().finally() would crash on a non-promise return.
  const work = Promise.resolve().then(run)

  // Outcome of the (uncancellable) work, inspected only by the straggler
  // handler after a timeout: a straggler that landed a session_id minted a
  // REAL runtime on the gateway (#96522 — _claim_or_reuse_live registers the
  // record before returning); a straggler that rejected minted nothing.
  let workOutcome: { ok: true; value: unknown } | { ok: false } | null = null

  work.then(
    value => {
      workOutcome = { ok: true, value }
    },
    () => {
      workOutcome = { ok: false }
    }
  )

  // The ceiling is derived per-flight (profile count), so the deadline wraps
  // the work only once the budget is known. Until then the flight is already
  // discoverable below, so a concurrent caller joins THIS attempt rather than
  // starting a second one while the import resolves.
  const flight = sessionResumeSettlementTimeoutMs()
    .then(timeoutMs =>
      withTimeout(work, timeoutMs, `Timed out resuming session ${storedSessionId}`, () => {
        work.then(() => {
          if (workOutcome?.ok !== true) {
            return
          }

          const lateId = sessionResumeOutcomeSessionId(workOutcome.value)

          if (!lateId) {
            return
          }

          // A NEWER flight owns the stored id: its caller adopts its own
          // result, so caching the old straggler would be a stale adoption.
          // An evicted slot (this flight settled and no successor started)
          // still adopts — the runtime is real and otherwise orphaned.
          const currentFlight = _inFlightResumeByStoredSessionId.get(storedSessionId)

          if (currentFlight !== undefined && currentFlight !== flight) {
            return
          }

          // Same contract as a drift-abort: record it so the next resume-shaped
          // action reuses the runtime instead of minting another orphan.
          registerRecoveredRuntime(storedSessionId, lateId)
        })
      })
    )
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
  _profileCountOverride = null
}
