/**
 * Pool limits — how many bot backends may stay spawned, and how long an
 * unused one survives before it is shut down.
 *
 * A device-local preference (each machine trades RAM against switching
 * speed for itself). The MAIN process is authoritative: it owns the pool
 * and the persisted copy, and applies a new max immediately by evicting
 * least-recently-used idle backends — no restart. This store mirrors the
 * live values for the Settings rows and feeds prewarmProfileBackend's
 * saturation guard.
 */

import { atom } from 'nanostores'

export interface PoolLimits {
  /** Max concurrently spawned non-primary profile backends. */
  maxBackends: number
  /** Idle lifetime of an unused pool backend, in milliseconds. */
  idleMs: number
}

export const POOL_LIMITS_DEFAULTS: PoolLimits = {
  maxBackends: 8,
  idleMs: 10 * 60_000
}

export const $poolLimits = atom<PoolLimits>({ ...POOL_LIMITS_DEFAULTS })

// Electron IPC rejects with a plain Error in the renderer, so the stable
// coordinator phrase is the cross-process discriminator. Keep it narrow: other
// backend failures must retain their existing recovery path.
const LOCAL_BACKEND_SLOT_TIMEOUT_PREFIX = 'Local backend start for "'
const LOCAL_BACKEND_SLOT_TIMEOUT_SUFFIX = ' timed out while waiting for a free slot.'
const LOCAL_BACKEND_SLOT_TIMEOUT_BACKGROUND_SUFFIX = ' (background)'

export function isLocalBackendSlotWaitTimeout(error: unknown): boolean {
  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return (
    message.includes(LOCAL_BACKEND_SLOT_TIMEOUT_PREFIX) &&
    message.includes(LOCAL_BACKEND_SLOT_TIMEOUT_SUFFIX) &&
    !message.endsWith(LOCAL_BACKEND_SLOT_TIMEOUT_BACKGROUND_SUFFIX)
  )
}

/** Navigation intent consumed by the shell, which owns React Router. */
export const $poolLimitsSettingsRequest = atom(0)

export function requestPoolLimitsSettings(): void {
  $poolLimitsSettingsRequest.set($poolLimitsSettingsRequest.get() + 1)
}

/** Seed from main's authoritative state once at startup; no-op without the
 *  bridge (web/older builds just keep the defaults for the UI). */
export async function loadPoolLimits(): Promise<void> {
  try {
    const limits = await window.hermesDesktop?.getPoolLimits?.()

    if (limits) {
      $poolLimits.set(limits)
    }
  } catch {
    // Keep defaults — Settings rows still render and can retry on save.
  }
}

/** Push new limits to main; adopt the post-clamp values it reports. */
export async function savePoolLimits(next: { maxBackends?: number; idleMs?: number }): Promise<void> {
  const current = $poolLimits.get()

  const optimistic: PoolLimits = {
    maxBackends: next.maxBackends ?? current.maxBackends,
    idleMs: next.idleMs ?? current.idleMs
  }

  // Optimistic paint, then honest reconciliation with the clamped result.
  $poolLimits.set(optimistic)

  try {
    const result = await window.hermesDesktop?.setPoolLimits?.(next)

    if (result?.limits) {
      $poolLimits.set(result.limits)
    }
  } catch {
    $poolLimits.set(current)
    throw new Error('Applying pool limits failed')
  }
}
