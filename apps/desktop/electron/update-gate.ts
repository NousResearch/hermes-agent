'use strict'

import { runBackendStartStep } from './backend-start-cancellation'

/**
 * update-gate.ts
 *
 * Pure, dependency-injected gate that parks local backend spawns while an
 * in-app update is running (#73822, #50238).
 *
 * Five independent signals mean "an update owns the local runtime right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section, and
 *  - the successful detached hand-off state, which remains true while this
 *    Desktop is waiting to quit after the wrapper has handed control away.
 *  - the host fleet-restart obligation record, armed by EXTERNAL updates
 *    (any profile's CLI, a maintenance script, the gateway's own /update)
 *    before they bounce the gateway fleet and discharged only when
 *    post-restart verification proves the fleet is on the pulled code
 *    (#126177).
 *
 * The marker alone is NOT enough (#73822): `applyUpdates` stops its own backend
 * early (`releaseBackendLock`) before committing the hand-off. The renderer
 * reconnects after the WebSocket closes; a marker-only gate can spawn a new
 * backend on the runtime being replaced. Consulting the flag closes that
 * window. On success the marker is written BEFORE the flag clears in `applyUpdates`'
 * `finally`, so there is no instant where both signals are false and a
 * waiter could slip through mid-update.
 *
 * Five independent signals mean "an update owns the local runtime right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section, and
 *  - the successful detached hand-off state, which remains true while this
 *    Desktop is waiting to quit after the wrapper has handed control away.
 *  - the host fleet-restart obligation record, armed by EXTERNAL updates
 *    (any profile's CLI, a maintenance script, the gateway's own /update)
 *    before they bounce the gateway fleet and discharged only when
 *    post-restart verification proves the fleet is on the pulled code
 *    (#126177). It closes the external-update window the other signals
 *    cannot see: the updater releases the live-update marker before its
 *    fleet-restart tail, and an update driven from outside this process
 *    never sets `updateInFlight`. The record is cross-process by
 *    construction — it lives in the host gateway-locks state dir, not this
 *    app's memory or this profile's home.
 *  - a terminal `failed` receipt naming a FINISHED, failed update (#122206):
 *    a failed `hermes update` releases the marker in its `finally` only when
 *    the failure happened after the lock was claimed; preparation-stage
 *    failures (uv sync network errors) can exit with the marker left pointing
 *    at a dead or recycled pid while `readLiveUpdateMarker`'s liveness probe
 *    keeps answering true on Windows, and even a cleanly released marker
 *    leaves the boot nothing to wait for. Parking the full 20-minute budget
 *    on a receipt that already says `failed` strands the window (#122206:
 *    486 polls / 20 minutes after a receipt-marked failure). The gate reports
 *    it separately so the boot path can surface the terminal failure instead
 *    of silently counting down.
 */

export type UpdateGateReason =
  | 'marker'
  | 'update-in-flight'
  | 'handoff'
  | 'fleet-restart-pending'
  | 'failed-receipt'
  | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
  /** True after a detached updater hand-off is viable and this Desktop will quit. */
  isHandoffActive: () => boolean
  /**
   * True when the latest update receipt records a terminal failure (see
   * main.ts readLatestSyncReceipt). Optional: older callers (and the pool
   * spawn path, which is not user-visible) may omit it, in which case a
   * live marker keeps the historical parking behavior.
   */
  hasFailedReceipt?: () => boolean
  /** True when the host still owes the fleet a restart onto freshly pulled code (#126177). */
  hasFleetRestartPending: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    // A failed receipt with a live marker is still the marker's wait — but the
    // receipt is what the boot path needs to know about, so it wins here.
    if (deps.hasFailedReceipt?.()) {
      return 'failed-receipt'
    }

    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  if (deps.isHandoffActive()) {
    return 'handoff'
  }

  if (deps.hasFleetRestartPending()) {
    return 'fleet-restart-pending'
  }

  return null
}

export type UpdateClearanceOutcome = 'clear' | 'finished' | 'timeout' | 'cancelled' | 'abandoned'

export interface WaitForUpdateClearanceOptions {
  signal?: AbortSignal
  isCancelled?: () => boolean
  timeoutMs: number
  pollMs: number
  /** Invoked once per poll while parked (boot progress / logging). */
  onWaitTick?: (reason: Exclude<UpdateGateReason, null>) => void | Promise<void>
  /**
   * Consulted whenever the gate is closed. Returning true makes the wait
   * return 'abandoned' immediately instead of parking. The primary boot path
   * uses this for the terminal failed-receipt signal (#122206): a receipt
   * that already says "failed" must not consume the parking budget.
   */
  abandonOn?: (reason: Exclude<UpdateGateReason, null>) => boolean
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

/**
 * Park until no update signal remains, or the deadline passes.
 *
 * Returns 'clear' when the gate was already open (no wait happened),
 * 'finished' when it opened during the wait, 'abandoned' when `abandonOn`
 * accepted the closed-gate reason, and 'timeout' when the deadline
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

  if (options.abandonOn?.(reason)) {
    return 'abandoned'
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

    if (reason && options.abandonOn?.(reason)) {
      return 'abandoned'
    }
  }

  return reason ? 'timeout' : 'finished'
}
