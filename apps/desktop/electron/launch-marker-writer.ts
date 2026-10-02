/**
 * Deferred persistence for the launch-time recovery markers.
 *
 * The sandbox / GPU ladders decide before `app ready`, because Chromium
 * command-line switches only apply pre-launch — that part cannot wait for the
 * single-instance lock. Persisting the markers cannot: a second launch that
 * loses the lock exits without ever starting a GPU or sandbox child, so its
 * `booting` write is indistinguishable from an aborted boot to the next
 * launch's two-strike counter. A few ordinary relaunches (a launcher button,
 * `gtk-launch`, a `hermes://` deeplink) were enough to promote the marker to
 * `fallback/boot-loop` and pin `--no-sandbox` on a host that never crashed.
 *
 * So the writes are queued here and flushed once the lock is resolved: a
 * primary instance persists them, a secondary one drops them and leaves the
 * markers exactly as the running instance left them.
 *
 * Pure and dependency-free so it can be unit-tested without Electron.
 */

export interface LaunchMarkerWriter {
  /** Queue a marker write for after the single-instance lock resolves. */
  queue(write: () => void): void
  /**
   * Persist everything queued, or nothing when this launch is not the lock
   * owner. Returns how many writes ran.
   */
  flush(isPrimaryInstance: boolean): number
  /** Writes still waiting — a primary instance that never flushed. */
  pending(): number
}

export function createLaunchMarkerWriter(): LaunchMarkerWriter {
  const queued: Array<() => void> = []

  return {
    queue(write) {
      if (typeof write === 'function') {
        queued.push(write)
      }
    },

    flush(isPrimaryInstance) {
      if (!isPrimaryInstance) {
        queued.length = 0

        return 0
      }

      const writes = queued.splice(0, queued.length)
      let ran = 0

      for (const write of writes) {
        try {
          write()
          ran += 1
        } catch {
          // A marker that cannot be persisted must not take the launch with
          // it: the ladder re-reads whatever is on disk next time.
        }
      }

      return ran
    },

    pending() {
      return queued.length
    }
  }
}
