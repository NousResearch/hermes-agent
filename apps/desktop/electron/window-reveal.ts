type WindowRevealTarget = {
  isDestroyed: () => boolean
  isVisible: () => boolean
  show: () => void
}

type TimerHandle = ReturnType<typeof setTimeout>

type WindowRevealOptions = {
  onRevealed?: () => void
  /** Called once when the window fails to reach first paint (see fail below). */
  onRevealFailed?: (reason: string) => void
  delayMs?: number
  setTimer?: (callback: () => void, delayMs: number) => TimerHandle
  clearTimer?: (timer: TimerHandle) => void
}

export const WINDOW_REVEAL_FALLBACK_MS = 4_000

export function createWindowRevealController(
  window: WindowRevealTarget,
  {
    onRevealed = () => {},
    onRevealFailed = () => {},
    delayMs = WINDOW_REVEAL_FALLBACK_MS,
    setTimer = (callback, delay) => setTimeout(callback, delay),
    clearTimer = timer => clearTimeout(timer)
  }: WindowRevealOptions = {}
) {
  let disposed = false
  let revealed = false
  let failed = false
  let fallbackTimer: TimerHandle | null = null

  const cancelFallback = () => {
    if (fallbackTimer === null) {
      return
    }

    clearTimer(fallbackTimer)
    fallbackTimer = null
  }

  const reveal = () => {
    if (disposed || revealed || failed || window.isDestroyed()) {
      return false
    }

    revealed = true
    cancelFallback()

    if (!window.isVisible()) {
      window.show()
    }

    onRevealed()

    return true
  }

  const scheduleFallback = () => {
    if (disposed || revealed || failed || fallbackTimer !== null || window.isDestroyed()) {
      return
    }

    fallbackTimer = setTimer(() => {
      fallbackTimer = null
      reveal()
    }, delayMs)
  }

  /**
   * The reveal gate has no success-shaped event left to hang on: the main
   * frame failed to load or the render process died before first paint, so
   * neither `ready-to-show` nor `did-finish-load` will ever fire — and a
   * `show: false` window would stay hidden forever while callers keep
   * claiming it is open (#108230). Hand it to `onRevealFailed` once and
   * disarm every other path. A window that already revealed keeps its own
   * caller-chosen lifecycle: a post-reveal crash is not this branch's
   * business.
   */
  const fail = (reason: string) => {
    if (disposed || revealed || failed || window.isDestroyed()) {
      return false
    }

    failed = true
    cancelFallback()
    onRevealFailed(reason)

    return true
  }

  const dispose = () => {
    disposed = true
    cancelFallback()
  }

  return {
    dispose,
    fail,
    reveal,
    scheduleFallback
  }
}

/** Minimal event-emitter shape wireWindowReveal listens on (a BrowserWindow
 *  satisfies it; tests pass fakes). */
type RevealEvents = {
  once(event: string, listener: (...args: unknown[]) => void): unknown
  on(event: string, listener: (...args: unknown[]) => void): unknown
}

export type WindowRevealWindow = WindowRevealTarget &
  RevealEvents & {
    webContents: RevealEvents & {
      on(
        event: 'did-fail-load',
        listener: (
          event: unknown,
          errorCode: number,
          errorDescription: string,
          validatedURL: string,
          isMainFrame: boolean
        ) => void
      ): unknown
      on(event: 'render-process-gone', listener: (event: unknown, details: { reason?: string }) => void): unknown
    }
  }

/**
 * Wire a `show: false` window's reveal to the load lifecycle: reveal on
 * `ready-to-show`, arm the bounded fallback on `did-finish-load`, and — only
 * for callers that pass `onRevealFailed` — catch the two pre-paint death
 * modes (`did-fail-load` on the main frame, `render-process-gone`) so a
 * failed window is handed to the caller once instead of staying hidden
 * forever.
 */
export function wireWindowReveal(
  win: WindowRevealWindow,
  {
    show,
    onRevealed,
    onRevealFailed,
    ...timing
  }: {
    show?: () => void
    onRevealed?: () => void
    onRevealFailed?: (reason: string) => void
  } & Omit<WindowRevealOptions, 'onRevealed' | 'onRevealFailed'> = {}
) {
  const controller = createWindowRevealController(
    {
      isDestroyed: () => win.isDestroyed(),
      isVisible: () => win.isVisible(),
      show: show ?? (() => win.show())
    },
    { onRevealed, onRevealFailed, ...timing }
  )

  win.once('ready-to-show', controller.reveal)
  win.webContents.once('did-finish-load', controller.scheduleFallback)
  win.on('closed', controller.dispose)

  if (onRevealFailed) {
    win.webContents.on('did-fail-load', (_event, errorCode, errorDescription, _validatedURL, isMainFrame) => {
      if (isMainFrame) {
        controller.fail(`main frame failed to load (${errorCode}: ${errorDescription})`)
      }
    })

    win.webContents.on('render-process-gone', (_event, details) => {
      controller.fail(`render process gone (${details?.reason ?? 'unknown'})`)
    })
  }

  return controller
}
