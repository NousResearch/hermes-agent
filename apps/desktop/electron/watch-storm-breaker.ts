/**
 * Rate breaker for raw fs.watch handles (#118974).
 *
 * On Windows, ReadDirectoryChangesW can flood a single fs.watch handle with
 * hundreds of thousands of `rename` events per second (observed: 180k/s on
 * the desktop-plugins dir). The JS debounce in main.ts only guards the IPC
 * dispatch; libuv still calls the listener for every event, pinning a core
 * until the watcher is closed.
 *
 * guardedWatch counts events per window. Past the threshold it closes the
 * FSWatcher and falls back to a slow poll of a cheap snapshot (readdir names
 * or stat), reporting a change only when the snapshot differs — the same
 * shape as the 5s readdir poll the directory watch replaced. The breaker is
 * armed on win32 only: that is where the storm happens, and elsewhere a
 * legitimate burst would needlessly demote a watch to polling.
 */

export const WATCH_STORM_MAX_EVENTS = 1000
export const WATCH_STORM_WINDOW_MS = 1000
export const WATCH_STORM_POLL_MS = 5000

export type WatchListener = (eventType: string, filename: string | Buffer | null) => void

export interface ClosableWatcher {
  close(): void
}

export interface GuardedWatchOptions {
  /** Platform as data (process.platform in production) so tests can pick. */
  platform: string
  /** Create the underlying watcher (fs.watch in production). */
  watch: (listener: WatchListener) => ClosableWatcher
  /** Forwarded for every event while in watch mode. */
  onEvent: WatchListener
  /** Cheap fingerprint of the watched target, compared between polls. */
  snapshot: () => string
  /** Called in poll mode when the snapshot changed since the last poll. */
  onPollChange: () => void
  /** Called once when the breaker trips (logging). */
  onTrip?: (info: { events: number; windowMs: number }) => void
  now?: () => number
  setInterval?: (fn: () => void, ms: number) => unknown
  clearInterval?: (handle: unknown) => void
  maxEvents?: number
  windowMs?: number
  pollMs?: number
}

export interface GuardedWatch {
  close(): void
  mode(): 'watch' | 'poll' | 'closed'
}

export function stormBreakerArmed(platform: string): boolean {
  return platform === 'win32'
}

/**
 * `null` = the snapshot failed (target removed or unreadable). Kept distinct
 * from the empty string, which is the valid snapshot of an empty directory, so
 * an empty directory vanishing still compares as a change.
 */
function safeSnapshot(snapshot: () => string): string | null {
  try {
    return snapshot()
  } catch {
    return null
  }
}

export function guardedWatch(options: GuardedWatchOptions): GuardedWatch {
  const now = options.now ?? Date.now
  const startInterval = options.setInterval ?? ((fn, ms) => setInterval(fn, ms))
  const stopInterval = options.clearInterval ?? (handle => clearInterval(handle as ReturnType<typeof setInterval>))
  const maxEvents = options.maxEvents ?? WATCH_STORM_MAX_EVENTS
  const windowMs = options.windowMs ?? WATCH_STORM_WINDOW_MS
  const pollMs = options.pollMs ?? WATCH_STORM_POLL_MS
  const armed = stormBreakerArmed(options.platform)

  let state: 'watch' | 'poll' | 'closed' = 'watch'
  let windowStart = now()
  let count = 0
  let pollHandle: unknown = null
  let watcher: ClosableWatcher | null = null

  const trip = () => {
    state = 'poll'
    watcher?.close()
    watcher = null
    options.onTrip?.({ events: count, windowMs })

    let last = safeSnapshot(options.snapshot)

    pollHandle = startInterval(() => {
      const next = safeSnapshot(options.snapshot)

      if (next !== last) {
        last = next
        options.onPollChange()
      }
    }, pollMs)
  }

  watcher = options.watch((eventType, filename) => {
    if (state !== 'watch') {
      return
    }

    if (armed) {
      const t = now()

      if (t - windowStart >= windowMs) {
        windowStart = t
        count = 0
      }

      count += 1

      if (count > maxEvents) {
        trip()

        return
      }
    }

    options.onEvent(eventType, filename)
  })

  return {
    close() {
      if (state === 'closed') {
        return
      }

      state = 'closed'
      watcher?.close()
      watcher = null

      if (pollHandle !== null) {
        stopInterval(pollHandle)
        pollHandle = null
      }
    },
    mode: () => state
  }
}
