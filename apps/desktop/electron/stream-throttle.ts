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
// On native Wayland only, a fullscreen chat may be misclassified as occluded
// after sitting idle (#94865). The caller can opt into keeping the fleet
// unthrottled while any registered window is fullscreen. The policy is
// deliberately opt-in so macOS/Windows/X11 keep the idle-CPU contract.
//
// Pure and Electron-free (timers + the WebContents surface are injected) so it
// can be unit-tested, mirroring session-windows.ts.

/** How long after the last turn ends before throttling is restored. Covers the
 * stream queue's final coalesced flush and the settle writes that trail a
 * turn's completion, so re-throttling never strands a visible delta. */
const RETHROTTLE_DELAY_MS = 5_000

export interface ThrottleWindowLike {
  isDestroyed(): boolean
  isFullScreen?: () => boolean
  webContents?: {
    isDestroyed(): boolean
    setBackgroundThrottling(allowed: boolean): void
  } | null
}

interface TimersLike {
  clearTimeout(handle: unknown): void
  setTimeout(fn: () => void, ms: number): unknown
}

export interface StreamThrottleOptions {
  /** Native-Wayland workaround for #94865. False by default so other
   * platforms retain Chromium's idle background throttling. */
  keepFullscreenPainting?: boolean
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
  delayMs: number = RETHROTTLE_DELAY_MS,
  options: StreamThrottleOptions = {}
): StreamThrottle {
  const windows = new Set<ThrottleWindowLike>()
  const keepFullscreenPainting = options.keepFullscreenPainting === true
  let unthrottled = false
  let trailing: unknown = null
  let lastBusy = false

  function anyFullscreen(): boolean {
    if (!keepFullscreenPainting) {
      return false
    }

    for (const win of windows) {
      try {
        if (!win.isDestroyed() && win.isFullScreen?.()) {
          return true
        }
      } catch {
        // A window mid-teardown can throw from geometry queries; skip it.
      }
    }

    return false
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
      contents.setBackgroundThrottling(!unthrottled)
    } catch {
      // A window mid-teardown can throw; it's about to leave the set anyway.
    }
  }

  function applyAll() {
    for (const win of windows) {
      apply(win)
    }
  }

  function armRethrottleIfIdle() {
    if (lastBusy || anyFullscreen() || !unthrottled || trailing !== null) {
      return
    }

    trailing = timers.setTimeout(() => {
      trailing = null
      unthrottled = false
      applyAll()
    }, delayMs)
  }

  return {
    isUnthrottled: () => unthrottled,

    register(win) {
      windows.add(win)
      win.on?.('closed', () => {
        windows.delete(win)
        armRethrottleIfIdle()
      })

      if (keepFullscreenPainting) {
        win.on?.('enter-full-screen', () => {
          if (trailing !== null) {
            timers.clearTimeout(trailing)
            trailing = null
          }
          if (!unthrottled) {
            unthrottled = true
            applyAll()
          }
        })
        win.on?.('leave-full-screen', () => {
          armRethrottleIfIdle()
        })
      }

      apply(win)
    },

    update(busy) {
      lastBusy = busy

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

      if (anyFullscreen()) {
        return
      }

      // Trailing edge: keep full cadence briefly so the final flush paints.
      armRethrottleIfIdle()
    }
  }
}
