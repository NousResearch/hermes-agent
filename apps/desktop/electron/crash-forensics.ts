/**
 * Last-chance forensics for the Electron main process.
 *
 * Electron installs its own `uncaughtException` listener and only warns on
 * unhandled rejections, so the app usually survives — but the reason lands on
 * stderr alone, which is discarded entirely when the app is launched from
 * Finder or the Start menu. Without a record in desktop.log, a main-process
 * fault is invisible in a `hermes debug share` bundle and the user is left
 * describing symptoms instead of showing a stack.
 */

export interface CrashForensicsTarget {
  on: (event: 'uncaughtException' | 'unhandledRejection', listener: (value: unknown) => void) => unknown
}

export interface CrashForensicsOptions {
  flush: () => void
  log: (message: string) => void
  target?: CrashForensicsTarget
}

/** Render a thrown value for the log, preferring a stack over a bare message. */
export function describeCrashReason(reason: unknown): string {
  if (reason instanceof Error) {
    return reason.stack || reason.message || reason.name || 'Error'
  }

  if (typeof reason === 'string') {
    return reason
  }

  try {
    return JSON.stringify(reason) ?? String(reason)
  } catch {
    return String(reason)
  }
}

/** Property stamped on intentional app-transition sentinel errors. */
const EXPECTED_TRANSITION_FLAG = '__hermesExpectedTransition'

/**
 * Stamp a sentinel Error as an expected, intentional app transition (#119409).
 *
 * The shutdown path aborts in-flight work on purpose; the sentinel's rejection
 * reaches the unhandledRejection hook on every clean quit. Marking it lets
 * crash forensics record the transition in one line instead of a crash-shaped
 * stack, so error-shaped entries in desktop.log keep meaning "something died".
 */
export function markExpectedTransition<T extends Error>(error: T): T {
  ;(error as T & { [EXPECTED_TRANSITION_FLAG]?: boolean })[EXPECTED_TRANSITION_FLAG] = true

  return error
}

/** True when a rejection reason is a marked intentional-transition sentinel. */
export function isExpectedTransition(reason: unknown): reason is Error {
  if (!(reason instanceof Error)) {
    return false
  }

  return (reason as unknown as Record<string, unknown>)[EXPECTED_TRANSITION_FLAG] === true
}

/**
 * Record main-process faults to desktop.log and flush synchronously, since a
 * fault that does prove fatal leaves no chance for the batched async flush.
 */
export function installCrashForensics({ flush, log, target = process }: CrashForensicsOptions): void {
  const record = (label: string) => (reason: unknown) => {
    if (isExpectedTransition(reason)) {
      log(`[main] ${label}: expected shutdown transition — ${reason.message}`)
      flush()

      return
    }

    log(`[main] ${label}: ${describeCrashReason(reason)}`)
    flush()
  }

  target.on('uncaughtException', record('Uncaught exception'))
  target.on('unhandledRejection', record('Unhandled rejection'))
}
