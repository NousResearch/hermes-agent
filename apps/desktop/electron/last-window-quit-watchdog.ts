/**
 * Bounded fail-safe for the last-window quit path.
 *
 * Electron normally exits after window-all-closed on Windows/Linux, but its
 * async backend teardown can be interrupted or wedged after the windows are
 * already gone. The single-instance lock then leaves every later launch
 * routing into a process that has no window. This watchdog gives graceful
 * teardown time to finish, then releases the stale process only if it is still
 * windowless.
 */

interface TimersLike {
  clearTimeout(handle: unknown): void
  setTimeout(fn: () => void, ms: number): unknown
}

interface LastWindowQuitWatchdogOptions {
  hasWindows: () => boolean
  forceExit: () => void
  onForcedExit?: () => void
  delayMs?: number
  timers?: TimersLike
}

export const LAST_WINDOW_QUIT_WATCHDOG_MS = 15_000

export function createLastWindowQuitWatchdog({
  hasWindows,
  forceExit,
  onForcedExit = () => {},
  delayMs = LAST_WINDOW_QUIT_WATCHDOG_MS,
  timers = { clearTimeout: handle => clearTimeout(handle as never), setTimeout }
}: LastWindowQuitWatchdogOptions) {
  let timer: unknown = null

  const cancel = () => {
    if (timer === null) {
      return
    }

    timers.clearTimeout(timer)
    timer = null
  }

  return {
    arm() {
      if (timer !== null) {
        return
      }

      timer = timers.setTimeout(() => {
        timer = null

        if (hasWindows()) {
          return
        }

        onForcedExit()
        forceExit()
      }, delayMs)
    },
    cancel
  }
}
