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

  return {
    start: () => {
      if (timer) {
        return
      }

      expectedAt = now() + cadenceMs
      timer = setInterval(tick, cadenceMs)
    },
    stop: () => {
      if (!timer) {
        return
      }

      clearInterval(timer)
      timer = undefined
    }
  }
}
