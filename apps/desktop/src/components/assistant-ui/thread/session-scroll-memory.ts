/**
 * Per-session transcript reading position.
 *
 * The thread scroller pins to the bottom on every sessionKey change. That is
 * correct for sessions the user left following the tail, and wrong for sessions
 * they left mid-read: coming back jumps to newest and the place is lost
 * (#45562).
 *
 * Store distance-from-bottom (`scrollHeight - scrollTop`), the same anchor
 * "Show earlier" already uses, so prepends and late markdown layout do not
 * invalidate it. Sessions within a few pixels of the edge are remembered as
 * `bottom` so a return still follows new turns.
 *
 * In-memory only (session switches in this window). Bounded LRU so a long
 * profile does not grow without cap.
 */
export const SESSION_SCROLL_BOTTOM_PX = 8
export const SESSION_SCROLL_MEMORY_CAP = 64

export type SessionScrollState = { kind: 'bottom' } | { kind: 'offset'; fromBottom: number }

export type ScrollMetrics = {
  clientHeight: number
  scrollHeight: number
  scrollTop: number
}

export function classifySessionScroll(metrics: ScrollMetrics, thresholdPx = SESSION_SCROLL_BOTTOM_PX): SessionScrollState {
  const remaining = metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight

  if (remaining <= thresholdPx) {
    return { kind: 'bottom' }
  }

  return { kind: 'offset', fromBottom: metrics.scrollHeight - metrics.scrollTop }
}

export function sessionScrollTargetTop(state: SessionScrollState, metrics: Pick<ScrollMetrics, 'clientHeight' | 'scrollHeight'>): number {
  if (state.kind === 'bottom') {
    return Math.max(0, metrics.scrollHeight - metrics.clientHeight)
  }

  return Math.max(0, metrics.scrollHeight - state.fromBottom)
}

/** True when the current content is tall enough to place an offset without clamping to 0. */
export function sessionScrollOffsetPlaceable(state: SessionScrollState, scrollHeight: number): boolean {
  if (state.kind === 'bottom') {
    return true
  }

  return scrollHeight >= state.fromBottom
}

export function createSessionScrollMemory(cap = SESSION_SCROLL_MEMORY_CAP) {
  const map = new Map<string, SessionScrollState>()

  const remember = (key: string | null | undefined, state: SessionScrollState): void => {
    if (!key) {
      return
    }

    if (map.has(key)) {
      map.delete(key)
    }

    map.set(key, state)

    while (map.size > cap) {
      const oldest = map.keys().next().value

      if (oldest === undefined) {
        break
      }

      map.delete(oldest)
    }
  }

  const recall = (key: string | null | undefined): SessionScrollState | null => {
    if (!key) {
      return null
    }

    const state = map.get(key)

    if (!state) {
      return null
    }

    map.delete(key)
    map.set(key, state)

    return state
  }

  const forget = (key: string | null | undefined): void => {
    if (!key) {
      return
    }

    map.delete(key)
  }

  const reset = (): void => {
    map.clear()
  }

  return { forget, recall, remember, reset, size: () => map.size }
}

export const sessionScrollMemory = createSessionScrollMemory()
