/** Cadences for the floating-companion `companion.info` refresh timer.

Event-capable backends rely on `companion.changed` for live updates but still need
a slow backstop: the gateway seeds the companion signature silently at boot and only
broadcasts when it *moves*, and the one-shot connect pull can race a still-
warming backend (`companion.info` fail-opens to `enabled:false`).
*/

export const COMPANION_POLL_MS = 3_000
export const COMPANION_ACTIVE_REFRESH_MS = 15_000
/** Slow safety net when `companion.changed` is available. */
export const COMPANION_BACKSTOP_MS = 15_000
/** Cold-start retries after the first connect pull (fail-open recovery). */
export const COMPANION_STARTUP_RETRY_MS = [1_000, 3_000, 8_000] as const

export function companionInfoPollIntervalMs(changeEventsAvailable: boolean, active: boolean): number {
  if (changeEventsAvailable) {
    return COMPANION_BACKSTOP_MS
  }

  return active ? COMPANION_ACTIVE_REFRESH_MS : COMPANION_POLL_MS
}
