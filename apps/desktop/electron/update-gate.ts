'use strict'

import { UPDATE_MARKER_MAX_AGE_MS } from './update-marker'
import { runBackendStartStep } from './backend-start-cancellation'

/** Backend starts wait for marker ownership, in-process work, and accepted handoff.
 * A terminal receipt is diagnostic evidence only; it cannot override an owner.
 * Authoritatively cleared markers permit immediate recovery after a failed update.
 */
export type UpdateGateReason = 'marker' | 'update-in-flight' | 'handoff' | 'failed-receipt' | null

export interface UpdateGateDeps {
  /** True when a live on-disk update marker exists (see update-marker.ts). */
  hasLiveMarker: () => boolean
  /** True while this process is inside applyUpdates()' critical section. */
  isUpdateInFlight: () => boolean
  /** True after a detached updater hand-off is viable and this Desktop will quit. */
  isHandoffActive?: () => boolean
  /**
   * Retained for call-site compatibility and receipt diagnostics. A terminal
   * failure does not change gate ownership or authorize a backend start.
   */
  hasFailedReceipt?: () => boolean
}

/** Why the gate is closed right now, or null when it is open. */
export function updateGateReason(deps: UpdateGateDeps): UpdateGateReason {
  if (deps.hasLiveMarker()) {
    return 'marker'
  }

  if (deps.isUpdateInFlight()) {
    return 'update-in-flight'
  }

  if (deps.isHandoffActive?.()) {
    return 'handoff'
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
   * return 'abandoned' immediately instead of parking. Callers must never
   * use a failed receipt to bypass an authoritative marker owner.
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
 * expired with the gate still closed. Backend-start callers must refuse
 * startup on timeout; authoritative marker recovery opens the gate safely.
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
    if (outcome === 'cancelled' || outcome === 'abandoned') { return outcome }

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
