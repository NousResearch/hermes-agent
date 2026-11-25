const DEFAULT_TIMEOUT_MS = 10_000

type TimerHandle = unknown

type Schedule = (callback: () => void, timeoutMs: number) => TimerHandle

type Cancel = (handle: TimerHandle) => void

export interface QuitFinalizationOptions {
  enabled: boolean
  hardExit: (code: number) => void
  timeoutMs?: number
  schedule?: Schedule
  cancel?: Cancel
}

export interface QuitFinalization {
  arm: () => void
  cancel: () => void
}

/**
 * Platforms where Electron can stall after JS quit teardown has been admitted
 * and leave a zero-window process alive instead of emitting `quit`.
 */
export function shouldArmQuitFinalization(platform: NodeJS.Platform | string): boolean {
  return platform === 'win32' || platform === 'darwin'
}

/**
 * Provides a bounded escape hatch for an Electron process that has finished
 * admitted quit teardown (or entered `will-quit`) but never emits `quit`.
 *
 * The fallback is armed only after normal before-quit teardown is complete and
 * admitted, never while user confirmation or managed remote updates are active.
 * A successful `quit` event cancels it.
 */
export function createQuitFinalization({
  enabled,
  hardExit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  schedule = (callback, delay) => setTimeout(callback, delay),
  cancel = handle => clearTimeout(handle as ReturnType<typeof setTimeout>)
}: QuitFinalizationOptions): QuitFinalization {
  let timer: TimerHandle | null = null
  let finished = false

  return {
    arm() {
      if (!enabled || finished || timer !== null) {
        return
      }

      timer = schedule(() => {
        timer = null

        if (finished) {
          return
        }

        finished = true
        hardExit(0)
      }, timeoutMs)
    },

    cancel() {
      if (finished) {
        return
      }

      finished = true

      if (timer !== null) {
        cancel(timer)
        timer = null
      }
    }
  }
}
