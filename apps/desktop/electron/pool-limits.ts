/**
 * Pool limits — how many bot backends may stay spawned, and how long an
 * unused one survives.
 *
 * A device-local preference (each machine trades RAM against switching
 * speed for itself), stored in userData like keep-awake. The main process
 * is authoritative: it owns the pool AND the persisted copy, and applies a
 * new max IMMEDIATELY by evicting least-recently-used idle backends — no
 * app restart. The renderer mirrors the values for its UI and prewarm
 * guard over IPC.
 *
 * Defaults preserve the historical hard-coded behavior (3 backends, 10min
 * idle) so machines that never open Settings behave exactly as before.
 */

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

/** Hard floors — match the clamps the env-var path always applied. */
export const POOL_LIMITS_MIN: PoolLimits = {
  maxBackends: 1,
  idleMs: 60_000
}

/** Shared bounds for both pool knobs — imported by the Settings UI so the
 *  advertised input ranges can never drift from what main actually clamps
 *  to. idleMs has no ceiling: a user who wants backends kept warm all week
 *  may have exactly that. */
export const POOL_LIMITS_BOUNDS = {
  maxBackendsMax: 64,
  /** 7 days, matching the UI's suggestion ceiling. */
  idleMsMax: 7 * 24 * 60 * 60_000
} as const

const MAX_BACKENDS_CEILING = POOL_LIMITS_BOUNDS.maxBackendsMax
const IDLE_MS_CEILING = POOL_LIMITS_BOUNDS.idleMsMax

/** Clamp a raw partial to the floors/ceilings; missing keys fall to defaults. */
export function clampPoolLimits(raw: Partial<PoolLimits>): PoolLimits {
  const maxBackends = Number.isFinite(raw.maxBackends)
    ? Math.min(MAX_BACKENDS_CEILING, Math.max(POOL_LIMITS_MIN.maxBackends, Math.floor(Number(raw.maxBackends))))
    : POOL_LIMITS_DEFAULTS.maxBackends

  const idleMs = Number.isFinite(raw.idleMs)
    ? Math.min(IDLE_MS_CEILING, Math.max(POOL_LIMITS_MIN.idleMs, Math.floor(Number(raw.idleMs))))
    : POOL_LIMITS_DEFAULTS.idleMs

  return { maxBackends, idleMs }
}

function clampLimits(raw: Partial<PoolLimits>): PoolLimits {
  return clampPoolLimits(raw)
}

/** Parse + clamp a persisted JSON blob; anything unreadable falls back to
 *  defaults so a corrupted file can never wedge the pool. */
export function parsePoolLimits(json: string | null | undefined): PoolLimits {
  if (!json) {
    return { ...POOL_LIMITS_DEFAULTS }
  }

  try {
    const parsed = JSON.parse(json)

    return clampLimits({
      maxBackends: typeof parsed?.maxBackends === 'number' ? parsed.maxBackends : undefined,
      idleMs: typeof parsed?.idleMs === 'number' ? parsed.idleMs : undefined
    })
  } catch {
    return { ...POOL_LIMITS_DEFAULTS }
  }
}
