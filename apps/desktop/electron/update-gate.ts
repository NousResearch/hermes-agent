'use strict'

import { runBackendStartStep } from './backend-start-cancellation'

/**
 * update-gate.ts
 *
 * Pure, dependency-injected gate that parks local backend spawns while an
 * in-app update is running (#73822, #50238).
 *
 * Three independent signals mean "an update owns the venv right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section, and
 *  - the successful detached hand-off state, which remains true while this
 *    Desktop is waiting to quit after the wrapper has handed control away.
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

export type UpdateGateReason = 'marker' | 'update-in-flight' | 'handoff' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
  /** True after a detached updater hand-off is viable and this Desktop will quit. */
  isHandoffActive: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  if (deps.isHandoffActive()) {
    return 'handoff'
  }

  return null
}

export type UpdateClearanceOutcome = 'clear' | 'finished' | 'timeout' | 'cancelled'

export interface WaitForUpdateClearanceOptions {
  signal?: AbortSignal
  isCancelled?: () => boolean
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

  const isCancelled = () => options.signal?.aborted || options.isCancelled?.()

  if (isCancelled()) {
    return 'cancelled'
  }

  let reason = updateGateReason(deps)

  if (!reason) {
    return 'clear'
  }

  const deadline = now() + options.timeoutMs

  while (reason && now() < deadline) {
    if (isCancelled()) {
      return 'cancelled'
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    try {
      if (options.onWaitTick) {
        await runBackendStartStep(options.signal, () => options.onWaitTick!(reason!))
      }

      if (isCancelled()) {
        return 'cancelled'
      }

      await runBackendStartStep(options.signal, () =>
        options.sleep
          ? sleep(options.pollMs)
          : new Promise<void>(resolve => {
              timer = setTimeout(resolve, options.pollMs)
            })
      )
    } catch (error) {
      if (isCancelled()) {
        return 'cancelled'
      }

      throw error
    } finally {
      clearTimeout(timer)
    }

    if (isCancelled()) {
      return 'cancelled'
    }

    reason = updateGateReason(deps)
  }

  return reason ? 'timeout' : 'finished'
}
