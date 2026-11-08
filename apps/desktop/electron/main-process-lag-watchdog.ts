type IntervalHandle = ReturnType<typeof setInterval>

interface MainProcessLagWatchdogOptions {
  cadenceMs: number
  thresholdMs: number
  now: () => number
  log: (message: string) => void
  setInterval: (callback: () => void, delayMs: number) => IntervalHandle
  clearInterval: (timer: IntervalHandle) => void
}

/**
 * Records delayed main-process timer callbacks after the event loop resumes.
 * It intentionally observes only; a stall must never change window or tray
 * behavior while the application is recovering.
 */
export function createMainProcessLagWatchdog({
  cadenceMs,
  thresholdMs,
  now,
  log,
  setInterval,
  clearInterval
}: MainProcessLagWatchdogOptions) {
  let timer: IntervalHandle | undefined
  let expectedAt = 0
  let suspended = false

  const tick = () => {
    if (!timer) {
      return
    }

    const observedAt = now()
    const lagMs = Math.max(0, observedAt - expectedAt)

    if (lagMs >= thresholdMs) {
      log(
        `[diagnostics] main-process event loop lagged ${lagMs}ms (expected tick at ${expectedAt}ms, observed at ${observedAt}ms)`
      )
    }

    // Rebase after every callback: one late tick is evidence, not a reason to
    // report the same delay again on every healthy future cadence.
    expectedAt = observedAt + cadenceMs
  }

  const start = () => {
    if (timer) {
      return
    }

    expectedAt = now() + cadenceMs
    timer = setInterval(tick, cadenceMs)
  }

  const stop = () => {
    if (!timer) {
      return
    }

    clearInterval(timer)
    timer = undefined
  }

  return {
    start,
    stop: () => {
      suspended = false
      stop()
    },
    // System sleep freezes this process, so the first tick after wake lands
    // minutes "late" and would be logged as a stall that never happened (in
    // the field every long lag lined up with a Sleep entry in `pmset -g log`).
    // Stand down for the suspend; resume restarts with a fresh baseline, but
    // only a watchdog that was running when the machine went to sleep.
    suspend: () => {
      if (timer) {
        stop()
        suspended = true
      }
    },
    resume: () => {
      if (suspended) {
        suspended = false
        start()
      }
    }
  }
}
