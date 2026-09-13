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
// Fullscreen is the same disease with a nastier symptom (#94865): on Hyprland
// (and other Wayland compositors) Chromium can treat a fullscreened surface as
// occluded/backgrounded once the user goes idle. Frame submission stops, the
// compositor keeps showing the last (or an empty) buffer — a white page —
// until a geometry change (leaving fullscreen) forces full damage. A fullscreen
// chat is definitionally the surface the user is looking at, so while ANY
// tracked window is fullscreen this controller keeps its renderer unthrottled.
// Like streaming, that state is scoped and reversible: it ends the moment the
// window leaves fullscreen. It deliberately does NOT pin visibilityState for
// hidden windows — only the fullscreen window itself stays live.
//
// Pure and Electron-free (timers + the WebContents surface are injected) so it
// can be unit-tested, mirroring session-windows.ts.

/** How long after the last turn ends before throttling is restored. Covers the
 * stream queue's final coalesced flush and the settle writes that trail a
 * turn's completion, so re-throttling never strands a visible delta. */
const RETHROTTLE_DELAY_MS = 5_000

export interface ThrottleWindowLike {
  isDestroyed(): boolean
  /** Optional so pure fakes can omit Electron-only geometry methods; real
   * BrowserWindows always have them. */
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

export interface StreamThrottle {
  /** True while windows are currently unthrottled (streaming or trailing). */
  isUnthrottled(): boolean
  /** Track a chat window; applies the current state immediately, follows the
   * window through fullscreen transitions, and stops tracking on close. */
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
  let lastBusy = false

  function anyFullscreen(): boolean {
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

    let allowed = !unthrottled

    if (!allowed && win.isFullScreen?.()) {
      // Streaming settled, but this window is fullscreen: keep painting.
      // Occlusion-driven white-screen (#94865) beats idle CPU savings for
      // the one surface the user pinned to their whole screen.
      allowed = false
    }

    try {
      contents.setBackgroundThrottling(allowed)
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
      // Follow the compositor: entering fullscreen must lift throttling even
      // when no turn is in flight, leaving it may restore throttling once the
      // stream settles. Both events re-evaluate every window because the
      // "any fullscreen" predicate spans the whole set.
      win.on?.('enter-full-screen', () => {
        if (trailing !== null) {
          timers.clearTimeout(trailing)
          trailing = null
        }
        unthrottled = true
        applyAll()
      })
      win.on?.('leave-full-screen', () => {
        // Idle exit: nothing needs full cadence anymore. If a settle report
        // already passed through while fullscreen held the lift, arm the
        // trailing re-throttle now (same delay as a stream tail).
        armRethrottleIfIdle()
      })
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

      // Settling during fullscreen: stay unthrottled — the fullscreen window
      // still needs frames — and skip arming the trailing re-throttle until
      // no window is fullscreen anymore.
      if (anyFullscreen()) {
        return
      }

      // Trailing edge: keep full cadence briefly so the final flush paints.
      armRethrottleIfIdle()
    }
  }
}
