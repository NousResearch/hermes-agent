// Stream-aware background throttling for chat windows.
//
// Visible chat windows must paint the live transcript while blurred or occluded,
// but hidden and minimized windows need not paint it. A static
// `backgroundThrottling: false` pins `document.visibilityState` to 'visible',
// turning visibility-gated polls and clocks into always-on timers. An idle,
// hidden Hermes burned ~20% CPU forever.
//
// So throttling is a runtime dial instead: the renderers already report
// "which chats are mid-turn" for the quit guard (`hermes:active-work`), and
// this controller rides the merged edge of those reports. Any turn in flight →
// visible, non-minimized chat windows get `setBackgroundThrottling(false)`.
// Native visibility transitions reapply that policy; bounded activity holds
// can still wake one hidden owner. All turns done → after a short trailing
// delay, Chromium's default throttling returns.
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
  isVisible(): boolean
  isMinimized(): boolean
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
  /** True while global paint demand is active; hidden windows may still throttle. */
  isUnthrottled(): boolean
  /** Track a chat window; applies the current state immediately and stops
   * tracking on close. */
  register(win: ThrottleWindowLike & { on?: (event: string, fn: () => void) => void }): void
  /** Keep one window's timers awake for bounded native work; release is idempotent. */
  holdWindow(win: ThrottleWindowLike): () => void
  /** Speech output still consumes frame-published text; preserve its streaming cadence. */
  setVoiceActive(win: ThrottleWindowLike, active: boolean): void
  /** Report whether any turn is in flight across all renderers. */
  update(busy: boolean): void
}

export function createStreamThrottle(
  timers: TimersLike = { clearTimeout: handle => clearTimeout(handle as never), setTimeout },
  delayMs: number = RETHROTTLE_DELAY_MS,
  options: StreamThrottleOptions = {}
): StreamThrottle {
  const windows = new Set<ThrottleWindowLike>()
  const windowHolds = new WeakMap<ThrottleWindowLike, number>()
  const voiceWindows = new WeakSet<ThrottleWindowLike>()
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
      windowHolds.delete(win)

      return
    }

    const contents = win.webContents

    if (!contents || contents.isDestroyed()) {
      return
    }

    try {
      const needsStreamPaint = unthrottled && ((win.isVisible() && !win.isMinimized()) || voiceWindows.has(win))
      contents.setBackgroundThrottling(!(needsStreamPaint || windowHolds.has(win)))
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

    setVoiceActive(win, active) {
      if (!windows.has(win) || voiceWindows.has(win) === active) {
        return
      }

      if (active) {
        voiceWindows.add(win)
      } else {
        voiceWindows.delete(win)
      }

      apply(win)
    },

    holdWindow(win) {
      if (!windows.has(win) || win.isDestroyed() || !win.webContents || win.webContents.isDestroyed()) {
        throw new Error('Preview owner is not a live registered window')
      }

      const held = windowHolds.get(win) ?? 0
      windowHolds.set(win, held + 1)

      if (!held) {
        apply(win)
      }

      let released = false

      return () => {
        if (released) {
          return
        }

        released = true
        const count = windowHolds.get(win)

        if (count === undefined) {
          return
        }

        if (count > 1) {
          windowHolds.set(win, count - 1)
        } else {
          windowHolds.delete(win)

          apply(win)
        }
      }
    },

    register(win) {
      windows.add(win)
      win.on?.('closed', () => {
        windows.delete(win)
        windowHolds.delete(win)
        armRethrottleIfIdle()
      })
      for (const event of ['show', 'hide', 'minimize', 'restore']) {
        win.on?.(event, () => apply(win))
      }

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
