import type { DesktopAgentWork } from '@/global'

/** Same liveness window the Bot Mode roster uses for its worker pulse: the
 *  activity heartbeat refreshes a running worker's `last_activity_at` about
 *  once a minute, so 150s means "the last heartbeat is still current". */
export const FLEET_WORKER_ACTIVE_WINDOW_S = 150

/** Two missed 60s roster refreshes. Past this the rail paints the work dot
 *  hollow: the claim was true when fetched, but nobody has confirmed it since. */
export const FLEET_WORK_STALE_MS = 120_000

export type FleetWorkState = 'fresh' | 'idle' | 'stale'

/**
 * How a roster worker claim should paint right now.
 *
 * Freshness comes from the claim's own `lastActive` (the gateway heartbeat),
 * never from the fetch time: a poll that stops updating cannot keep a dot
 * green past the liveness window. `stale` means the claim is still inside that
 * window but the roster itself is old enough that we cannot vouch for it.
 */
export function fleetWorkState(
  work: DesktopAgentWork | undefined,
  rosterFetchedAt: number,
  nowMs: number
): FleetWorkState {
  if (!work) {
    return 'idle'
  }

  const lastActiveMs = work.lastActive * 1000

  if (!Number.isFinite(lastActiveMs) || nowMs - lastActiveMs >= FLEET_WORKER_ACTIVE_WINDOW_S * 1000) {
    return 'idle'
  }

  if (!rosterFetchedAt || nowMs - rosterFetchedAt > FLEET_WORK_STALE_MS) {
    return 'stale'
  }

  return 'fresh'
}
