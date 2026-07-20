// Cross-window de-dupe for one-shot side-effects (OS notifications, the turn-end
// sound, spoken replies). Every desktop window is its own renderer process, so N
// open windows each independently react to the same backend event. The main
// process is the one place they all share and it handles IPC serially, so it's
// the race-free owner: the first window to claim a key within the window wins;
// peers see it's taken and stay quiet.
//
// Pure + injectable clock so it's unit-testable without Electron.

const DEDUPE_WINDOW_MS = 1000

// Returns true when `key` was already claimed within the window (caller drops
// this one). Self-evicting: stale keys are pruned on every call, so the map
// can't grow unbounded.
function createEventDeduper(windowMs = DEDUPE_WINDOW_MS) {
  const lastSeenAt = new Map<string, number>()

  return function isDuplicate(key: string, now = Date.now()): boolean {
    for (const [k, at] of lastSeenAt) {
      if (now - at >= windowMs) {
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

export { createEventDeduper, DEDUPE_WINDOW_MS }
