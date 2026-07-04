/**
 * liveness-streak.cjs
 *
 * Consecutive-failure tracking for the cached remote backend's liveness
 * probe (hermes:connection:revalidate in main.cjs). A single slow or dropped
 * GET /api/status — routine over Tailscale right after a wake, or while the
 * remote host is briefly loaded — must NOT tear down the cached connection:
 * dropping it forces a full backend teardown/rebuild on every renderer
 * reconnect tick, producing the ready → probe-fail → restart churn seen in
 * desktop.log. Only a STREAK of consecutive failures against the same base
 * URL is treated as a dead remote.
 *
 * Kept in a standalone cjs module so it can be unit-tested with `node --test`
 * without dragging in the electron runtime (same pattern as backend-probes.cjs
 * and hardening.cjs).
 */

/**
 * Create a consecutive-failure tracker.
 *
 * @param {object} [options]
 * @param {number} [options.threshold] - Consecutive failures (against the
 *   same key) required before `recordFailure` reports `drop: true`.
 *   Clamped to >= 1; a threshold of 1 restores drop-on-first-failure.
 * @returns {{
 *   recordFailure(key?: string): { drop: boolean, count: number, threshold: number, firstOfStreak: boolean },
 *   recordSuccess(): void
 * }}
 */
function createLivenessStreak(options = {}) {
  const threshold = Math.max(1, Math.floor(Number(options.threshold) || 1))
  let count = 0
  let streakKey = null

  return {
    /**
     * Record one probe failure against `key` (the probed base URL). A key
     * change means a different backend is being probed — the old streak is
     * meaningless, so the count restarts at 1. When the streak reaches the
     * threshold, `drop` is true and the streak resets so the next failure
     * starts a fresh streak against the rebuilt connection.
     */
    recordFailure(key = '') {
      const k = String(key || '')
      if (k !== streakKey) {
        streakKey = k
        count = 0
      }
      count += 1
      const result = { drop: count >= threshold, count, threshold, firstOfStreak: count === 1 }
      if (result.drop) {
        count = 0
        streakKey = null
      }
      return result
    },

    /** A successful probe ends any in-progress streak. */
    recordSuccess() {
      count = 0
      streakKey = null
    }
  }
}

module.exports = { createLivenessStreak }
