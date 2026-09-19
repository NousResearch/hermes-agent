type WindowRevealTarget = {
  isDestroyed: () => boolean
  isVisible: () => boolean
  show: () => void
}

type TimerHandle = ReturnType<typeof setTimeout>

type WindowRevealOptions = {
  onRevealed?: () => void
  onFailed?: () => void
  delayMs?: number
  setTimer?: (callback: () => void, delayMs: number) => TimerHandle
  clearTimer?: (timer: TimerHandle) => void
}

export const WINDOW_REVEAL_FALLBACK_MS = 4_000

const TERMINAL_RENDERER_GONE_REASONS = new Set([
  'abnormal-exit',
  'crashed',
  'oom',
  'launch-failed',
  'integrity-failure',
  'memory-eviction'
])

export function isTerminalFailedLoadBeforeReveal(errorCode: unknown, isMainFrame: unknown): boolean {
  return isMainFrame === true && String(errorCode) !== '-3'
}

export function isTerminalRendererGoneBeforeReveal(reason: unknown): boolean {
  return TERMINAL_RENDERER_GONE_REASONS.has(String(reason ?? ''))
}

export function createWindowRevealController(
  window: WindowRevealTarget,
  {
    onRevealed = () => {},
    onFailed = () => {},
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

  const fail = () => {
    if (disposed || revealed || failed || window.isDestroyed() || window.isVisible()) {
      return false
    }

    failed = true
    cancelFallback()
    onFailed()

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
