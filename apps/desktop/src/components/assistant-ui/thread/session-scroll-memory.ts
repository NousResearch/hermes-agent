// Per-session scroll memory for the thread viewport (#70101).
//
// Switching sessions swaps the transcript inside ONE long-lived scroller, so
// "where was I in session A?" is state the DOM cannot hold for us. This module
// remembers, per session key, whether the user was parked at the bottom
// (sticky-follow) or reading at an exact offset — and list.tsx reapplies that
// state after the switch relayout stabilizes (async markdown / highlight /
// image loads land over many frames, so a one-shot scrollTop write restores to
// a height that immediately changes underneath it).
//
// Offsets are stored as distance-from-bottom, not scrollTop: the switch
// relayout mostly reshapes content ABOVE the on-screen turns (placeholder →
// real heights), and bottom-anchored math keeps the restored view steady under
// that churn — the same reason "Show earlier" restores from the bottom edge in
// list.tsx.

export type ScrollMetrics = {
  clientHeight: number
  scrollHeight: number
  scrollTop: number
}

export type SessionScrollState = { kind: 'bottom' } | { fromBottom: number; kind: 'offset' }

export const BOTTOM: SessionScrollState = { kind: 'bottom' }

// Within this many pixels of the bottom edge counts as "parked at the bottom".
// Deliberately tight: use-stick-to-bottom's own near-bottom band re-locks lazy
// scrollers anyway, and recording a small real offset as `bottom` would yank a
// reader who stopped just shy of the edge.
export const STICKY_BOTTOM_THRESHOLD_PX = 8

// Bounded so a marathon runtime that touches hundreds of sessions doesn't grow
// the map forever. Map iteration order gives LRU eviction — recording re-inserts
// the key at the back, so the front is always the least-recently-visited.
export const SCROLL_MEMORY_LIMIT = 64

const memory = new Map<string, SessionScrollState>()

/** Distance from the bottom edge; clamped to 0 for overscroll bounce. */
export function distanceFromBottom(metrics: ScrollMetrics): number {
  return Math.max(0, metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight)
}

/** Classify live metrics as sticky-bottom or an exact reading offset. */
export function stateFromMetrics(metrics: ScrollMetrics, threshold = STICKY_BOTTOM_THRESHOLD_PX): SessionScrollState {
  const fromBottom = distanceFromBottom(metrics)

  return fromBottom <= threshold ? BOTTOM : { fromBottom, kind: 'offset' }
}

/** The scrollTop that reapplies `state` at the current content height. */
export function targetScrollTop(state: SessionScrollState, metrics: Pick<ScrollMetrics, 'clientHeight' | 'scrollHeight'>): number {
  const max = Math.max(0, metrics.scrollHeight - metrics.clientHeight)

  return state.kind === 'bottom' ? max : Math.max(0, max - state.fromBottom)
}

export function recordSessionScroll(sessionKey: string | null | undefined, state: SessionScrollState): void {
  if (!sessionKey) {
    return
  }

  memory.delete(sessionKey)
  memory.set(sessionKey, state)

  if (memory.size > SCROLL_MEMORY_LIMIT) {
    const oldest = memory.keys().next().value

    if (oldest !== undefined) {
      memory.delete(oldest)
    }
  }
}

/** Unknown sessions restore to the bottom — the pre-#70101 behavior. */
export function recallSessionScroll(sessionKey: string | null | undefined): SessionScrollState {
  return (sessionKey ? memory.get(sessionKey) : undefined) ?? BOTTOM
}

export function resetSessionScrollMemory(): void {
  memory.clear()
}
