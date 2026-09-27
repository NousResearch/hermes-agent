const DEFAULT_TIMEOUT_MS = 10_000

type TimerHandle = unknown

type Schedule = (callback: () => void, timeoutMs: number) => TimerHandle

type Cancel = (handle: TimerHandle) => void

export interface QuitFinalizationOptions {
  isWindows: boolean
  hardExit: (code: number) => void
  timeoutMs?: number
  schedule?: Schedule
  cancel?: Cancel
}

export interface QuitFinalization {
  /** Windows-only watchdog armed from `will-quit`. */
  arm: () => void
  /**
   * Quit teardown has already aborted the backend. Arm on every platform.
   * If Electron never emits `quit` — a beforeunload that never returns, or a
   * follow-up `app.quit()` that a window cancels — force the process out.
   * A successful `quit` event cancels it.
   */
  armAfterSealedTeardown: (delayMs?: number) => void
  cancel: () => void
}

export interface ManagedUpdateQuitCoordinatorOptions {
  hasInFlightUpdates: () => boolean
  waitForUpdates: () => Promise<void>
  requestQuit: () => void
  armSealedTeardown: (delayMs?: number) => void
}

export interface ManagedUpdateQuitCoordinator {
  handleBeforeQuit: (event: { preventDefault: () => void }) => boolean
  armSealedQuitExit: (delayMs?: number) => void
  isWaitDone: () => boolean
}

/**
 * Ensures in-flight managed updates or recoveries are joined before before-quit
 * teardown seals the backend.
 *
 * While updates are in flight, before-quit is prevented and sealed-teardown
 * watchdogs are suppressed. Once the update operations settle, app.quit() is
 * re-requested so normal teardown can proceed and arm the watchdog.
 */
export function createManagedUpdateQuitCoordinator({
  hasInFlightUpdates,
  waitForUpdates,
  requestQuit,
  armSealedTeardown
}: ManagedUpdateQuitCoordinatorOptions): ManagedUpdateQuitCoordinator {
  let waitPromise: Promise<void> | null = null
  let waitDone = false

  function isInFlight(): boolean {
    return !waitDone && (waitPromise !== null || hasInFlightUpdates())
  }

  return {
    handleBeforeQuit(event) {
      if (!isInFlight()) {
        return false
      }

      event.preventDefault()

      if (!waitPromise) {
        waitPromise = waitForUpdates().finally(() => {
          waitDone = true
          requestQuit()
        })
      }

      return true
    },

    armSealedQuitExit(delayMs?: number) {
      if (isInFlight()) {
        return
      }

      armSealedTeardown(delayMs)
    },

    isWaitDone() {
      return waitDone
    }
  }
}

/**
 * Provides a bounded escape hatch for an Electron process that has admitted
 * quit but never emits the completed quit event.
 *
 * `arm` is Windows-only and is called from `will-quit` (#116376: the process
 * stayed resident with 0 windows). `armAfterSealedTeardown` covers the earlier
 * stall: `before-quit` aborts the backend, then the follow-up `app.quit()`
 * never reaches `will-quit`, and the still-open renderer reports "Hermes
 * couldn't start". A successful `quit` event cancels whichever timer is live.
 */
export function createQuitFinalization({
  isWindows,
  hardExit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  schedule = (callback, delay) => setTimeout(callback, delay),
  cancel = handle => clearTimeout(handle as ReturnType<typeof setTimeout>)
}: QuitFinalizationOptions): QuitFinalization {
  let timer: TimerHandle | null = null
  let finished = false

  function scheduleExit(delayMs: number): void {
    if (finished || timer !== null) {
      return
    }

    timer = schedule(() => {
      timer = null

      if (finished) {
        return
      }

      finished = true
      hardExit(0)
    }, delayMs)
  }

  return {
    arm() {
      if (!isWindows) {
        return
      }

      scheduleExit(timeoutMs)
    },

    armAfterSealedTeardown(delayMs: number = timeoutMs) {
      scheduleExit(delayMs)
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
