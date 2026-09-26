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
 * wait covers the announce deadline plus the health poll after it and stays
 * finite: a dead child rejects the IPC when it exits or when a deadline
 * passes, and a wedged round-trip still ends. This is one wait, not a retry
 * interval.
 */

export const DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 90_000

// Never trust a deadline tighter than the warm-start path needs. Floor at the
// historical 45s default so a malformed override can't reintroduce the
// kill-and-respawn loop (#50209).
export const MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45_000

// Main's `waitForHermes` health-poll deadline after the port is announced
// (electron/backend-health.ts re-exports it).
export const DEFAULT_BACKEND_READY_TIMEOUT_MS = 45_000

// Work outside those two deadlines: pre-spawn gating (release gate, runtime
// resolution, pool claim) starts the renderer clock before main's announce
// timer, and token adoption plus the first WS upgrade run after health. A
// Windows cold start can stall that upgrade 12-28s (#96177).
export const BOOT_CONNECT_MARGIN_MS = 30_000

type AnnounceTimeoutEnv = {
  HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS?: string
}

/**
 * Port-announcement deadline. Honors `HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS`
 * for slow disks / aggressive AV, clamped to the warm-start floor so a bad
 * value can't make boot flakier than the historical default. `env` is
 * required: apps/shared has no Node types, so Electron passes `process.env`.
 */
export function resolvePortAnnounceTimeoutMs(env: AnnounceTimeoutEnv): number {
  const parsed = Number(env.HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS)

  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(MIN_PORT_ANNOUNCE_TIMEOUT_MS, Math.round(parsed))
  }

  return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS
}

/**
 * Renderer connect timeout for a primary cold boot.
 *
 * Covers main's whole cold-start chain: the port-announce wait, the health
 * poll after it, and `BOOT_CONNECT_MARGIN_MS` for the work around them, so a
 * backend that announces near the deadline and then turns healthy still
 * reaches the renderer. A missing, non-finite or non-positive announce input
 * falls back to the default deadline, so the result is never an immediate
 * fail and never Infinity (a dead backend with no recovery overlay).
 */
export function resolveRendererBootWaitMs(announceTimeoutMs?: number): number {
  const announce =
    typeof announceTimeoutMs === 'number' && Number.isFinite(announceTimeoutMs) && announceTimeoutMs > 0
      ? announceTimeoutMs
      : DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS

  return announce + DEFAULT_BACKEND_READY_TIMEOUT_MS + BOOT_CONNECT_MARGIN_MS
}
