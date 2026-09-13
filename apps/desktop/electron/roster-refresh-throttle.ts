/**
 * roster-refresh-throttle.ts
 *
 * Bounded, coalesced refresh for the desktop-wide agent roster — the one
 * choke point every roster read routes through (`hermes:agents:roster` →
 * enumerateRegistryAgentSources, per #104227).
 *
 * Why this exists: several renderer surfaces fetch this roster — the sidebar
 * on mount/focus/registry change (fleet-roster.ts), the skills pane, and every
 * profile-routed SDK call. Each fetch enumerates EVERY registered connection,
 * and for a source that is not yet in the backend pool that enumeration dials
 * or spawns a backend. When the profile pool cap is reached the coordinator
 * refuses the spawn after its 30s slot wait; the refused source keeps the
 * roster "incomplete", so the sidebar store re-asks on its 5s retry window and
 * the next fetch re-attempts the refused spawn. That is the reported storm:
 * refresh → refused spawn → refresh, cascading across sources and hammering
 * the model API while wedging the pool.
 *
 * Three bounds, all applied at that one seam:
 *  - single-flight: concurrent refreshes (boot fan-out, several panes) share
 *    ONE enumeration, so N callers cannot become N spawns;
 *  - cooldown: a refresh that arrives inside the current window is served the
 *    last snapshot instead of starting a new enumeration — the periodic poll
 *    keeps its data, the backends are left alone;
 *  - backoff: a refresh whose enumeration threw, or which came back DEGRADED
 *    (a source reported a real refusal/dial error — the pool-cap case, where
 *    the enumeration itself still resolves), doubles the cooldown up to a
 *    ceiling. The first clean refresh resets it.
 *
 * Fail open, never latched: with no snapshot to serve, a caller still runs
 * (the renderer must be able to recover), and a later success clears every
 * failure. Dependency-free (clock injected) so the windows are asserted
 * directly, same pattern as backend-dial-claim.ts / pool-eviction.ts.
 */

/** Floor between two enumerations. Matches the sidebar store's 5s retry
 *  window for an incomplete roster (fleet-roster.ts), so a healthy fleet
 *  refreshes exactly as often as it always did — only the duplicate and
 *  post-failure refreshes collapse away. */
export const ROSTER_REFRESH_COOLDOWN_MS = 5_000

/** Ceiling the failure backoff doubles up to — one refresh a minute while the
 *  pool is capped or a source is refusing to come up. */
export const ROSTER_REFRESH_MAX_COOLDOWN_MS = 60_000

export interface RosterRefreshThrottleOptions {
  /** Floor between two enumerations, in milliseconds. */
  cooldownMs?: number
  /** Ceiling for the failure backoff, in milliseconds. */
  maxCooldownMs?: number
  /** Injectable clock (tests). */
  now?: () => number
}

/** Did an enumeration that RESOLVED still report a failed source? A predicate
 *  rather than a fixed shape because only the caller knows which source errors
 *  are failures and which are deliberate skips. */
export type RosterRefreshDegraded<T> = (value: T) => boolean

export interface RosterRefreshThrottle {
  /** The cooldown currently enforced — grows on failure, resets on success. */
  readonly cooldownMs: number
  /** Whether an enumeration is in flight (test/diagnostic seam). */
  readonly inFlight: boolean
  /**
   * Run one refresh through the throttle. Concurrent callers share the
   * in-flight enumeration; a caller inside the cooldown is handed the last
   * snapshot without enumerating. Rejections propagate to every waiter.
   */
  run<T>(producer: () => Promise<T> | T, degraded?: RosterRefreshDegraded<T>): Promise<T>
}

export function createRosterRefreshThrottle(options: RosterRefreshThrottleOptions = {}): RosterRefreshThrottle {
  const baseCooldownMs = options.cooldownMs ?? ROSTER_REFRESH_COOLDOWN_MS
  const maxCooldownMs = Math.max(baseCooldownMs, options.maxCooldownMs ?? ROSTER_REFRESH_MAX_COOLDOWN_MS)
  const now = options.now ?? Date.now

  let inflight: Promise<unknown> | null = null
  let snapshot: { value: unknown } | null = null
  let nextAllowedAt = 0
  let cooldownMs = baseCooldownMs
  let failures = 0

  function backOff() {
    failures += 1
    cooldownMs = Math.min(maxCooldownMs, baseCooldownMs * 2 ** failures)
  }

  return {
    get cooldownMs() {
      return cooldownMs
    },

    get inFlight() {
      return inflight !== null
    },

    run<T>(producer: () => Promise<T> | T, degraded?: RosterRefreshDegraded<T>): Promise<T> {
      if (inflight) {
        return inflight as Promise<T>
      }

      // A warm snapshot inside the window is cheaper AND kinder than another
      // enumeration; with nothing to serve, the caller runs (fail open).
      if (snapshot && now() < nextAllowedAt) {
        return Promise.resolve(snapshot.value as T)
      }

      let pending: Promise<T>

      try {
        pending = Promise.resolve(producer())
      } catch (error) {
        pending = Promise.reject(error)
      }

      inflight = pending

      const release = () => {
        if (inflight === pending) {
          inflight = null
        }
      }

      const onResolved = (value: T) => {
        if (degraded?.(value)) {
          backOff()
        } else {
          failures = 0
          cooldownMs = baseCooldownMs
        }

        snapshot = { value }
        nextAllowedAt = now() + cooldownMs
        release()
      }

      const onRejected = () => {
        backOff()
        nextAllowedAt = now() + cooldownMs
        release()
      }

      void pending.then(onResolved, onRejected)

      return pending
    }
  }
}
