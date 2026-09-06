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
  maxBackends: 3,
  idleMs: 10 * 60_000
}

export const $poolLimits = atom<PoolLimits>({ ...POOL_LIMITS_DEFAULTS })

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
