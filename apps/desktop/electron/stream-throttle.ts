// Stream-aware background throttling for chat windows.
//
// Chat windows must paint the live transcript while blurred, occluded, or
// minimized — but a static `backgroundThrottling: false` in webPreferences
// costs far more than that feature needs: it pins the renderer's
// `document.visibilityState` to 'visible' for the life of the window, which
// turns every visibility-gated poll and clock tick in the renderer into an
// always-on timer. An idle, hidden Hermes burned ~20% CPU forever.
//
// So throttling is a runtime dial instead: each renderer already reports
// "which chats are mid-turn" for the quit guard (`hermes:active-work`). Only
// the reporting window gets `setBackgroundThrottling(false)` while it streams;
// sibling profiles and secondary windows remain eligible for Chromium's normal
// background throttling. When that window's turns finish, a short trailing
// delay lets its final coalesced flush land before throttling returns.
//
// Pure and Electron-free (timers + the WebContents surface are injected) so it
// can be unit-tested, mirroring session-windows.ts.

/** How long after the last turn ends before throttling is restored. Covers the
 * stream queue's final coalesced flush and the settle writes that trail a
 * turn's completion, so re-throttling never strands a visible delta. */
const RETHROTTLE_DELAY_MS = 5_000

export interface ThrottleWindowLike {
  isDestroyed(): boolean
  webContents?: {
    isDestroyed(): boolean
    setBackgroundThrottling(allowed: boolean): void
  } | null
}

interface TimersLike {
  clearTimeout(handle: unknown): void
  setTimeout(fn: () => void, ms: number): unknown
}

export interface StreamThrottle {
  /** True while this window (or any window when omitted) is unthrottled. */
  isUnthrottled(win?: ThrottleWindowLike): boolean
  /** Track a chat window; applies the current state immediately and stops
   * tracking on close. */
  register(win: ThrottleWindowLike & { on?: (event: string, fn: () => void) => void }): void
  /** Report whether this window has a turn in flight. */
  update(win: ThrottleWindowLike, busy: boolean): void
}

interface WindowThrottleState {
  trailing: unknown | null
  unthrottled: boolean
}

export function createStreamThrottle(
  timers: TimersLike = { clearTimeout: handle => clearTimeout(handle as never), setTimeout },
  delayMs: number = RETHROTTLE_DELAY_MS
): StreamThrottle {
  const windows = new Map<ThrottleWindowLike, WindowThrottleState>()

  function remove(win: ThrottleWindowLike) {
    const state = windows.get(win)

    if (state && state.trailing !== null) {
      timers.clearTimeout(state.trailing)
    }

    windows.delete(win)
  }

  function apply(win: ThrottleWindowLike, state: WindowThrottleState) {
    if (win.isDestroyed()) {
      remove(win)

      return
    }

    const contents = win.webContents

    if (!contents || contents.isDestroyed()) {
      remove(win)

      return
    }

    try {
      contents.setBackgroundThrottling(!state.unthrottled)
    } catch {
      // A window mid-teardown can throw; it's about to leave the set anyway.
    }
  }

  return {
    isUnthrottled: win =>
      win ? (windows.get(win)?.unthrottled ?? false) : [...windows.values()].some(state => state.unthrottled),

    register(win) {
      if (windows.has(win)) {
        return
      }

      const state: WindowThrottleState = { trailing: null, unthrottled: false }
      windows.set(win, state)
      win.on?.('closed', () => remove(win))
      apply(win, state)
    },

    update(win, busy) {
      const state = windows.get(win)

      if (!state) {
        return
      }

      if (busy) {
        if (state.trailing !== null) {
          timers.clearTimeout(state.trailing)
          state.trailing = null
        }

        if (!state.unthrottled) {
          state.unthrottled = true
          apply(win, state)
        }

        return
      }

      if (!state.unthrottled || state.trailing !== null) {
        return
      }

      // Trailing edge: keep full cadence briefly so the final flush paints.
      state.trailing = timers.setTimeout(() => {
        state.trailing = null
        state.unthrottled = false
        apply(win, state)
      }, delayMs)
    }
  }
}
