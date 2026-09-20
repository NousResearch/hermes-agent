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
 *  - `isHandoffLaunched`: true from the moment this process detaches a
 *    hand-off updater until it exits.
 *
 * The marker alone is NOT enough (#73822): `applyUpdates` kills its own
 * backend early (`releaseBackendLock`) but only writes the marker AFTER the
 * Windows venv-blocker scan. Killing the backend drops the renderer's
 * WebSocket, the renderer reconnects within ~1s, and a marker-only gate
 * happily spawns a fresh backend inside the update's own critical section —
 * which `scanVenvBlockers` then reports as a blocker, aborting every update
 * attempt forever. Consulting the flag closes that window.
 *
 * The flag alone is NOT enough either, on Windows. There the hand-off child
 * is a `cmd.exe /d /s /c start` WRAPPER (updater-process.ts
 * wrapHandoffForDetachedConsole) that exits 0 within milliseconds, so:
 *
 *  - the placeholder marker is written with that wrapper's pid
 *    (main.ts writeUpdateMarker right after the spawn) and therefore
 *    self-heals as stale on the very next read, and
 *  - `observeUpdaterHandoff` treats a clean exit inside its settle window as
 *    success (updater-process.ts, "clean exit 0 ... is expected for wrapper
 *    shapes"), so `applyUpdates` returns — and its `finally` clears
 *    `updateInFlight` — ~50 ms after launching the script, while the script
 *    itself only claims the marker with its own live pid ~1-2 s later.
 *
 * Both signals are then false for that whole gap, so a backend spawn that
 * arrives in it (the boot path's `waitForUpdateToFinish`, the renderer's
 * reconnect, the pool path, or the supervisor's respawn of the backend
 * `releaseBackendLockForUpdate` just killed) goes through. That backend then
 * holds the venv and keeps the Electron main process alive, which is exactly
 * what makes the hand-off script's "wait for the Desktop to exit" step fail:
 * it aborts after 30 s with "the Hermes window (pid …) did not exit within
 * 30s" and exit code 4, having changed nothing. `isHandoffLaunched` — wired
 * to the process's own quit-for-handoff latch, which is set before
 * `updateInFlight` clears and never needs clearing because the process dies
 * with it — keeps the gate closed across that gap so the hand-off is the last
 * thing this process does.
 */

export type UpdateGateReason = 'marker' | 'update-in-flight' | 'handoff' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
  /**
   * True once this process has detached a hand-off updater (it is quitting
   * and must not spawn a backend again). See the module docstring.
   */
  isHandoffLaunched: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  if (deps.isHandoffLaunched()) {
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
