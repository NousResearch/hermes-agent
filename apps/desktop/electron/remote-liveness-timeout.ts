/**
 * Remote liveness/dispatch probe timeout — how long the desktop CLIENT waits
 * for a remote backend's /api/health (or /api/status) probe before treating
 * it as unreachable.
 *
 * A device-local preference (each client machine trades probe patience
 * against how fast it notices a truly dead remote), stored in userData like
 * pool limits. The main process is authoritative: it owns the live value
 * (`remote-liveness.ts`'s getter/setter) and the persisted copy, and a change
 * applies immediately — no restart. `HERMES_REMOTE_LIVENESS_TIMEOUT_MS`
 * remains the initial-value fallback for scripted/headless setups (a
 * double-clicked, Finder/Dock-launched app never sees this env var — see
 * backend-env.ts), the same relationship `HERMES_DESKTOP_POOL_MAX`/
 * `HERMES_DESKTOP_POOL_IDLE_MS` have with pool-limits.json.
 *
 * Default preserves the historical hard-coded 10s so a machine that never
 * opens Settings and never sets the env var behaves exactly as before.
 */

export const REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS = 10_000

/** Hard floor/ceiling — a typo (or a fat-fingered Settings row) can't hang
 *  dispatch/reconnect forever, nor make every probe fail instantly. */
export const REMOTE_LIVENESS_TIMEOUT_BOUNDS = {
  min: 1_000,
  max: 120_000
} as const

/** Clamp a raw value to the floor/ceiling; anything non-numeric falls back
 *  to the default. */
export function clampRemoteLivenessTimeoutMs(raw: unknown): number {
  const n = Number(raw)

  if (!Number.isFinite(n) || n <= 0) {
    return REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS
  }

  return Math.min(REMOTE_LIVENESS_TIMEOUT_BOUNDS.max, Math.max(REMOTE_LIVENESS_TIMEOUT_BOUNDS.min, Math.floor(n)))
}

/** Legacy scripted/headless fallback, consulted only when no persisted
 *  preference exists yet: honours HERMES_REMOTE_LIVENESS_TIMEOUT_MS when it
 *  parses as a positive integer. */
export function resolveRemoteLivenessTimeoutMsFromEnv(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.HERMES_REMOTE_LIVENESS_TIMEOUT_MS

  if (raw == null || raw === '') {
    return REMOTE_LIVENESS_TIMEOUT_DEFAULT_MS
  }

  return clampRemoteLivenessTimeoutMs(raw)
}

/** Parse + clamp a persisted JSON blob; anything unreadable returns null so
 *  the caller falls through to the env-var fallback. */
export function parsePersistedRemoteLivenessTimeoutMs(json: null | string | undefined): null | number {
  if (!json) {
    return null
  }

  try {
    const parsed = JSON.parse(json)

    return typeof parsed?.timeoutMs === 'number' ? clampRemoteLivenessTimeoutMs(parsed.timeoutMs) : null
  } catch {
    return null
  }
}
