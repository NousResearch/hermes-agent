'use strict'

/**
 * update-gate.ts
 *
 * Pure, dependency-injected gate that parks local backend spawns while an
 * in-app update is running (#73822, #50238).
 *
 * Two independent signals mean "an update owns the venv right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section.
 *
 * The marker alone is NOT enough (#73822): `applyUpdates` kills its own
 * backend early (`releaseBackendLock`) but only writes the marker AFTER the
 * Windows venv-blocker scan. Killing the backend drops the renderer's
 * WebSocket, the renderer reconnects within ~1s, and a marker-only gate
 * happily spawns a fresh backend inside the update's own critical section —
 * which `scanVenvBlockers` then reports as a blocker, aborting every update
 * attempt forever. Consulting the flag closes that window. On the success
 * path the marker is written BEFORE the flag clears in `applyUpdates`'
 * `finally`, so there is no instant where both signals are false and a
 * waiter could slip through mid-update.
 */

import { UPDATE_MARKER_MAX_AGE_MS } from './update-marker'

export type UpdateGateReason = 'marker' | 'update-in-flight' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
}

/** One process-local mutex shared by normal update and bootstrap recovery. */
export class UpdateInFlightTransaction {
  private active = false

  readonly isActive = (): boolean => this.active

  async run<T>(operation: () => T | Promise<T>): Promise<T> {
    if (this.active) {
      throw new Error('An update is already in progress.')
    }

    this.active = true

    try {
      return await operation()
    } finally {
      this.active = false
    }
  }
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  return null
}

export type UpdateClearanceOutcome = 'clear' | 'finished' | 'timeout'

export interface WaitForUpdateClearanceOptions {
  timeoutMs: number
  pollMs: number
  /** Invoked once per poll while parked (boot progress / logging). */
  onWaitTick?: (reason: Exclude<UpdateGateReason, null>) => void | Promise<void>
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

/**
 * Park until no update signal remains, or the deadline passes.
 *
 * Returns 'clear' when the gate was already open (no wait happened),
 * 'finished' when it opened during the wait, and 'timeout' when the deadline
 * expired with the gate still closed (callers proceed anyway — matching the
 * long-standing marker-gate behavior, since a wedged updater must not brick
 * the app forever).
 */
export async function waitForUpdateClearance(
  deps: UpdateGateDeps,
  options: WaitForUpdateClearanceOptions
): Promise<UpdateClearanceOutcome> {
  const now = options.now || Date.now
  const sleep = options.sleep || (ms => new Promise<void>(r => setTimeout(r, ms)))

  let reason = updateGateReason(deps)

  if (!reason) {
    return 'clear'
  }

  const deadline = now() + options.timeoutMs

  while (reason && now() < deadline) {
    if (options.onWaitTick) {
      await options.onWaitTick(reason)
    }

    await sleep(options.pollMs)
    reason = updateGateReason(deps)
  }

  return reason ? 'timeout' : 'finished'
}

/**
 * Keep local backend startup parked across bounded UI wait windows.
 *
 * The park is bounded: after `blockedBudgetMs` (default: the marker's own
 * 20-minute age ceiling) of consecutive closed-gate windows the outcome is
 * 'timeout' and the caller must report a failed startup attempt without spawning
 * a backend. An unbounded loop parked the backend forever
 * behind an unreadable, malformed, future-dated or cleanup-race marker that
 * nothing could self-heal.
 */
export async function waitForLocalBackendClearance(
  deps: UpdateGateDeps,
  options: WaitForUpdateClearanceOptions & {
    onStillBlocked?: (reason: Exclude<UpdateGateReason, null>) => void | Promise<void>
    /** Total consecutive blocked time before giving up; defaults to UPDATE_MARKER_MAX_AGE_MS. */
    blockedBudgetMs?: number
  }
): Promise<UpdateClearanceOutcome> {
  const now = options.now || Date.now
  const blockedBudgetMs = Math.max(0, options.blockedBudgetMs ?? UPDATE_MARKER_MAX_AGE_MS)
  const blockedSince = now()
  let waited = false

  while (true) {
    const outcome = await waitForUpdateClearance(deps, options)

    if (outcome === 'clear') {
      return waited ? 'finished' : 'clear'
    }

    if (outcome === 'finished') {
      return 'finished'
    }

    waited = true
    const reason = updateGateReason(deps)

    if (reason) {
      await options.onStillBlocked?.(reason)
    }

    if (now() - blockedSince >= blockedBudgetMs) {
      return 'timeout'
    }
  }
}
