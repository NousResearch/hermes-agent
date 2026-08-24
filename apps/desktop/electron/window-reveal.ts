type WindowRevealTarget = {
  isDestroyed: () => boolean
  isVisible: () => boolean
  show: () => void
}

type TimerHandle = ReturnType<typeof setTimeout>

type WindowRevealOptions = {
  onRevealed?: () => void
  delayMs?: number
  setTimer?: (callback: () => void, delayMs: number) => TimerHandle
  clearTimer?: (timer: TimerHandle) => void
}

export const WINDOW_REVEAL_FALLBACK_MS = 4_000

export function createWindowRevealController(
  window: WindowRevealTarget,
  {
    onRevealed = () => {},
    delayMs = WINDOW_REVEAL_FALLBACK_MS,
    setTimer = (callback, delay) => setTimeout(callback, delay),
    clearTimer = timer => clearTimeout(timer)
  }: WindowRevealOptions = {}
) {
  let disposed = false
  let revealed = false
  let fallbackTimer: TimerHandle | null = null

  const cancelFallback = () => {
    if (fallbackTimer === null) {
      return
    }

    clearTimer(fallbackTimer)
    fallbackTimer = null
  }

  const reveal = () => {
    if (disposed || revealed || window.isDestroyed()) {
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
    if (disposed || revealed || fallbackTimer !== null || window.isDestroyed()) {
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
    reveal,
    scheduleFallback
  }
}
