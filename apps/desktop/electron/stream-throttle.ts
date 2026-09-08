// Stream-aware background throttling for chat windows.
//
// Chat windows must paint the live transcript while blurred, occluded, or
// minimized — but a static `backgroundThrottling: false` in webPreferences
// costs far more than that feature needs: it pins the renderer's
// `document.visibilityState` to 'visible' for the life of the window, which
// turns every visibility-gated poll and clock tick in the renderer into an
// always-on timer. An idle, hidden Hermes burned ~20% CPU forever.
//
// So throttling is a runtime dial instead: the renderers already report
// "which chats are mid-turn" for the quit guard (`hermes:active-work`), and
// this controller rides the merged edge of those reports. Any turn in flight →
// every registered chat window gets `setBackgroundThrottling(false)`, exactly
// the streaming behavior the static flag used to provide. All turns done →
// after a short trailing delay (so tail flushes land at full cadence) Chromium's
// default throttling returns and hidden windows go quiet.
//
// Pure and Electron-free (timers + the WebContents surface are injected) so it
// can be unit-tested, mirroring session-windows.ts.

/** How long after the last turn ends before throttling is restored. Covers the
 * stream queue's final coalesced flush and the settle writes that trail a
 * turn's completion, so re-throttling never strands a visible delta. */
const RETHROTTLE_DELAY_MS = 5_000

export interface ThrottleWindowLike {
  isDestroyed(): boolean
  /** Optional window-state probes. When absent the window is treated as
   * off-screen, which preserves the pre-existing busy-only semantics. */
  isMinimized?(): boolean
  isVisible?(): boolean
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
  /** True while windows are currently unthrottled (streaming or trailing). */
  isUnthrottled(): boolean
  /** Track a chat window; applies the current state immediately and stops
   * tracking on close. */
  register(win: ThrottleWindowLike & { on?: (event: string, fn: () => void) => void }): void
  /** Report whether any turn is in flight across all renderers. */
  update(busy: boolean): void
}

export function createStreamThrottle(
  timers: TimersLike = { clearTimeout: handle => clearTimeout(handle as never), setTimeout },
  delayMs: number = RETHROTTLE_DELAY_MS
): StreamThrottle {
  const windows = new Set<ThrottleWindowLike>()
  let unthrottled = false
  let trailing: unknown = null

  /** A window the user can actually see. Chromium throttles rAF on a
   * throttled webContents, so an on-screen-but-blurred window stops animating
   * while IPC-driven DOM updates still land -- it reads as a freeze. Anything
   * on screen therefore never gets throttled, regardless of the busy dial.
   * Missing probes mean "not on screen" so injected test fakes keep their
   * busy-only behaviour. */
  function isOnScreen(win: ThrottleWindowLike): boolean {
    try {
      return win.isVisible?.() === true && win.isMinimized?.() !== true
    } catch {
      return false
    }
  }

  function apply(win: ThrottleWindowLike) {
    if (win.isDestroyed()) {
      windows.delete(win)

      return
    }

    const contents = win.webContents

    if (!contents || contents.isDestroyed()) {
      return
    }

    try {
      contents.setBackgroundThrottling(!unthrottled && !isOnScreen(win))
    } catch {
      // A window mid-teardown can throw; it's about to leave the set anyway.
    }
  }

  function applyAll() {
    for (const win of windows) {
      apply(win)
    }
  }

  return {
    isUnthrottled: () => unthrottled,

    register(win) {
      windows.add(win)
      win.on?.('closed', () => windows.delete(win))

      // Visibility transitions change the throttling verdict on their own, so
      // re-apply rather than waiting for the next busy edge.
      for (const event of ['minimize', 'restore', 'show', 'hide']) {
        win.on?.(event, () => apply(win))
      }

      apply(win)
    },

    update(busy) {
      if (busy) {
        if (trailing !== null) {
          timers.clearTimeout(trailing)
          trailing = null
        }

        if (!unthrottled) {
          unthrottled = true
          applyAll()
        }

        return
      }

      if (!unthrottled || trailing !== null) {
        return
      }

      // Trailing edge: keep full cadence briefly so the final flush paints.
      trailing = timers.setTimeout(() => {
        trailing = null
        unthrottled = false
        applyAll()
      }, delayMs)
    }
  }
}
