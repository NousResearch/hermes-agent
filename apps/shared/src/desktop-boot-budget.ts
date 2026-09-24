/**
 * Cold-start connect budgets shared by Electron main and the renderer.
 *
 * Main waits `DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS` for `HERMES_BACKEND_READY` /
 * `HERMES_DASHBOARD_READY` before it gives up on a still-starting child. The
 * renderer `getConnection()` IPC used to expire at 45s, so a slow Windows
 * cold start failed the boot overlay while main was still waiting and the
 * backend then came up healthy.
 *
 * Both sides read this module so those deadlines cannot drift. The renderer
 * wait covers the announce deadline and stays finite: a dead child rejects
 * the IPC when it exits or when the announce deadline passes, and a wedged
 * round-trip still ends. This is one wait, not a retry interval.
 */

export const DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 90_000

// Never trust a deadline tighter than the warm-start path needs. Floor at the
// historical 45s default so a malformed override can't reintroduce the
// kill-and-respawn loop (#50209).
export const MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45_000

type AnnounceTimeoutEnv = {
  HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS?: string
}

/**
 * Port-announcement deadline. Honors `HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS`
 * for slow disks / aggressive AV, clamped to the warm-start floor so a bad
 * value can't make boot flakier than the historical default.
 */
export function resolvePortAnnounceTimeoutMs(env: AnnounceTimeoutEnv = process.env): number {
  const parsed = Number(env.HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS)

  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(MIN_PORT_ANNOUNCE_TIMEOUT_MS, Math.round(parsed))
  }

  return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS
}

/**
 * Renderer connect timeout for a primary cold boot.
 *
 * Covers `announceTimeoutMs` so the IPC is still in flight when main's
 * port-announce wait ends. Non-finite or non-positive inputs fall back to the
 * default announce deadline — never 0 (an immediate fail that a caller could
 * spin on) and never Infinity (a dead backend with no recovery overlay).
 */
export function resolveRendererBootWaitMs(announceTimeoutMs: number): number {
  if (!Number.isFinite(announceTimeoutMs) || announceTimeoutMs <= 0) {
    return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS
  }

  return announceTimeoutMs
}
