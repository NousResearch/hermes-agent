'use strict'

import { runBackendStartStep } from './backend-start-cancellation'

/**
 * update-gate.ts
 *
 * Pure, dependency-injected gate that parks local backend spawns while an
 * in-app update is running (#73822, #50238).
 *
 * Three independent signals mean "an update owns the local runtime right now":
 *
 *  - the on-disk marker (`HERMES_HOME/.hermes-update-in-progress`), written
 *    by the updater — and by the desktop itself just before hand-off — and
 *  - the in-process `updateInFlight` flag, true for the whole
 *    `applyUpdates()` critical section, and
 *  - the successful detached hand-off state, which remains true while this
 *    Desktop is waiting to quit after the wrapper has handed control away.
 *
 * The marker alone is NOT enough (#73822): `applyUpdates` stops its backend
 * early (`releaseBackendLock`) before committing the hand-off. The renderer
 * reconnects after the WebSocket closes; a marker-only gate can spawn a new
 * backend on the runtime being replaced. Consulting the flag closes that
 * window. On success the marker is written BEFORE the flag clears in `applyUpdates`'
 * `finally`, so there is no instant where both signals are false and a
 * waiter could slip through mid-update.
 *
 * A finished, failed update (`logs/update_receipts/latest.json` says `failed`)
 * never outranks a live marker: `latest.json` is written only at finalize, so a
 * retry after a failed update still reads `failed` while the new update runs
 * (desktop V2). The marker's owner liveness (pid + creation time, see
 * update-marker.ts) is the only signal for "an update is running"; a crashed
 * updater's marker reads dead and opens the gate by itself.
 */

// How long the launch parks on an in-process update signal (in-flight /
// hand-off) before starting the backend anyway. A LIVE marker owner is waited
// out with no deadline (owner liveness, never age); past this the boot copy
// says the update is still running.
export const UPDATE_WAIT_TIMEOUT_MS = 20 * 60 * 1000
export const UPDATE_WAIT_POLL_MS = 1000
// How long the desktop lingers on the "updating, don't reopen" overlay after
// spawning the detached updater, before it quits to release the venv shim. The
// old 600ms was long enough to register the child process but far too short for
// the user to READ the overlay — the window just vanished, looked like a crash,
// and the user relaunched mid-update (the #50238 restart-loop trigger). A
// couple of seconds lets the message land and bridges the gap until the
// updater's own progress window appears. (#50419)
export const UPDATE_HANDOFF_DWELL_MS = 2500

export type UpdateGateReason = 'marker' | 'update-in-flight' | 'handoff' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean | Promise<boolean>
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
  /** True after a detached updater hand-off is viable and this Desktop will quit. */
  isHandoffActive: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export async function updateGateReason(deps: UpdateGateDeps): Promise<UpdateGateReason> {
  if (await deps.hasLiveMarker()) {
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
  onWaitTick?: (reason: Exclude<UpdateGateReason, null>, waitedMs: number) => void | Promise<void>
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

/**
 * Park until no update signal remains.
 *
 * Returns 'clear' when the gate was already open (no wait happened),
 * 'finished' when it opened during the wait, and 'timeout' when the deadline
 * expired with an in-process signal (in-flight / hand-off) still set. A LIVE
 * marker has no deadline here: owner liveness decides when an update is over
 * (C1 rule 3) — booting into a half-replaced runtime at minute 21 of a slow
 * Windows update is exactly the failure the gate exists for. The marker reader
 * still bounds the wait: an owner whose creation time cannot be read stops
 * reading live at the 20-minute ceiling (A1), so a reused pid never parks
 * boot forever. A dead marker whose checkout a leftover process still holds
 * (`held`) never ages out: the caller shows a blocked boot screen whose only
 * way through while the hold lasts is a confirmed, logged Start anyway
 * (update-marker-gate.ts, review R8 D3).
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

  let reason = await updateGateReason(deps)

  if (!reason) {
    return 'clear'
  }

  const startedAt = now()
  const deadline = startedAt + options.timeoutMs

  while (reason && (reason === 'marker' || now() < deadline)) {
    if (isCancelled()) {
      return 'cancelled'
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    try {
      if (options.onWaitTick) {
        const tickReason = reason
        await runBackendStartStep(options.signal, () => options.onWaitTick!(tickReason, now() - startedAt))
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

    reason = await updateGateReason(deps)
  }

  return reason ? 'timeout' : 'finished'
}
