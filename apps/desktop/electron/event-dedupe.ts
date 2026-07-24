// Cross-window de-dupe for one-shot side-effects (OS notifications, the turn-end
// sound, spoken replies). Every desktop window is its own renderer process, so N
// open windows each independently react to the same backend event. The main
// process is the one place they all share and it handles IPC serially, so it's
// the race-free owner: the first window to claim a key within the interval wins;
// peers see it's taken and stay quiet. Pure + injectable clock, so it's
// unit-testable without Electron.

const DEDUPE_INTERVAL_MS = 1000

// Returns true when `key` was already claimed within the interval (caller drops
// this one). Self-evicting: stale keys are pruned on every call, so the map
// can't grow unbounded.
export function createEventDeduper(intervalMs = DEDUPE_INTERVAL_MS) {
  const lastSeenAt = new Map<string, number>()

  return function isDuplicate(key: string, now = Date.now()): boolean {
    for (const [k, at] of lastSeenAt) {
      if (now - at >= intervalMs) {
        lastSeenAt.delete(k)
      }
    }

    if (lastSeenAt.has(key)) {
      return true
    }

    lastSeenAt.set(key, now)

    return false
  }
}
