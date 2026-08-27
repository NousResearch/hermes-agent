// Frozen-frame recovery for chat windows.
//
// Symptom (Linux/Wayland sessions, XWayland windows): the window's viewable
// area freezes on a single solid color and only a manual resize restores it.
// Chromium has silently stopped presenting frames — typically after a GPU
// process death (the browser restarts the process but does not always resume
// presenting), or when the compositor misses a reveal transition — and
// nothing short of a surface reconfiguration makes it present again. A resize
// forces exactly that reconfiguration, which is why users discover it as the
// fix. This controller automates that resize: a guarded, transient bounds
// nudge that forces a fresh buffer without leaving the window resized.
//
// It is deliberately narrow, matching how stream-throttle.ts treats the
// related throttling concern:
// - Nudges only on reveal transitions ('show' / 'restore') and on explicit
//   kick() calls (the GPU-process-gone recovery in main.ts). It never nudges
//   on plain focus, so alt-tabbing between healthy windows does not reflow
//   content.
// - Skips hidden, maximized, and fullscreen windows — a nudge there is
//   meaningless and can fight the window manager.
// - The transient size is restored ~80 ms later, well inside the geometry
//   persistence debounce, so the blip is never recorded as the window state.
//
// Pure and Electron-free (timers + the window surface are injected) so it can
// be unit-tested, mirroring stream-throttle.ts.

/** How many DIPs wider the window gets during a nudge. Large enough to
 * survive display-scale rounding (a 1-DIP nudge can round to zero physical
 * pixels on a scaled XWayland surface), small enough to be invisible. */
const NUDGE_DELTA_DIPS = 2

/** How long the transient nudge persists before the original bounds return. */
const NUDGE_RESTORE_DELAY_MS = 80

export interface RepaintWindowBounds {
  x: number
  y: number
  width: number
  height: number
}

export interface RepaintWindowLike {
  isDestroyed(): boolean
  isVisible(): boolean
  isMaximized(): boolean
  isFullScreen(): boolean
  getBounds(): RepaintWindowBounds
  setBounds(bounds: RepaintWindowBounds): void
  on?(event: string, listener: (...args: unknown[]) => void): unknown
}

export interface WindowRepaintController {
  /** Track a chat window: nudge on reveal transitions, stop on close. */
  register(win: RepaintWindowLike): void
  /** Nudge one window immediately (used after GPU-process death). */
  kick(win: RepaintWindowLike): void
  /** Nudge every tracked window (used after GPU-process death). */
  kickAll(): void
}

interface TimersLike {
  clearTimeout(handle: unknown): void
  setTimeout(fn: () => void, ms: number): unknown
}

export function createWindowRepaintController(
  timers: TimersLike = { clearTimeout: handle => clearTimeout(handle as never), setTimeout }
): WindowRepaintController {
  const windows = new Set<RepaintWindowLike>()
  // Windows whose nudge restore is still pending. Guards the show+restore
  // double-fire: the second reveal arrives while the first nudge is still
  // active, and a second nudge would restore to the *nudged* size.
  const restoring = new Set<RepaintWindowLike>()

  function nudge(win: RepaintWindowLike) {
    if (win.isDestroyed() || !win.isVisible() || win.isMaximized() || win.isFullScreen()) {
      return
    }

    if (restoring.has(win)) {
      return
    }

    let bounds: RepaintWindowBounds

    try {
      bounds = win.getBounds()
    } catch {
      return
    }

    restoring.add(win)

    try {
      win.setBounds({ ...bounds, width: bounds.width + NUDGE_DELTA_DIPS })
    } catch {
      restoring.delete(win)

      return
    }

    timers.setTimeout(() => {
      restoring.delete(win)

      if (win.isDestroyed()) {
        return
      }

      try {
        win.setBounds(bounds)
      } catch {
        // Teardown race: the window closed mid-nudge; nothing to restore into.
      }
    }, NUDGE_RESTORE_DELAY_MS)
  }

  return {
    register(win) {
      windows.add(win)
      win.on?.('show', () => nudge(win))
      win.on?.('restore', () => nudge(win))
      win.on?.('closed', () => windows.delete(win))
    },

    kick(win) {
      nudge(win)
    },

    kickAll() {
      for (const win of windows) {
        nudge(win)
      }
    }
  }
}
