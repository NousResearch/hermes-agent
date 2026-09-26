export interface QuitTeardownTask {
  /** Whether Electron must defer this quit until the task settles. */
  waitForCompletion: boolean
  /** Starts teardown. This is invoked synchronously from before-quit. */
  run: () => Promise<unknown> | unknown
}

export interface QuitTeardownCoordinator {
  /**
   * Starts teardown once and reports whether the current quit must be
   * prevented. Re-entrant quit requests are held until all required teardown
   * settles; the coordinator then requests exactly one final quit.
   */
  begin: (tasks: readonly QuitTeardownTask[]) => boolean
}

export interface BackendQuitActivity {
  connectionPending: boolean
  poolPending: boolean
  processAttached: boolean
  shutdownPending: boolean
}

export function backendQuitNeedsWait(activity: BackendQuitActivity): boolean {
  return activity.shutdownPending || activity.processAttached || activity.connectionPending || activity.poolPending
}

/**
 * What a deliberate primary-backend teardown intends to happen next.
 *
 * `'reconnect'`: a backend comes back (update hand-off, bundle swap, re-home).
 * `'quit'`: nothing comes back — the app is exiting, or being uninstalled.
 */
export type BackendTeardownIntent = 'quit' | 'reconnect'

/**
 * The `soft` option a deliberate teardown must pass to
 * `teardownPrimaryBackendAndWait()`.
 *
 * `soft: true` is what stops `resetHermesConnectionState()` from rewriting the
 * boot-progress overlay — the step that writes `[boot] Restarting desktop
 * connection` into desktop.log and pushes `hermes:boot-progress` to the
 * renderer. A reconnect may announce that; a quit must not, because the
 * announcement is false, it can overwrite the renderer's own "Update in
 * progress…" copy, and a reader of desktop.log then attributes a shutdown to a
 * re-home.
 */
export function backendTeardownOptions(intent: BackendTeardownIntent): { soft: boolean } {
  return { soft: intent === 'quit' }
}

function runTask(task: QuitTeardownTask): Promise<unknown> {
  try {
    return Promise.resolve(task.run())
  } catch (error) {
    return Promise.reject(error)
  }
}

/**
 * Coordinates Electron's before-quit teardown without cancelling a quit that
 * has no asynchronous work to wait for.
 */
export function createQuitTeardownCoordinator(requestFinalQuit: () => void): QuitTeardownCoordinator {
  let started = false
  let finished = false

  return {
    begin(tasks): boolean {
      if (finished) {
        return false
      }

      if (started) {
        return true
      }

      started = true
      const executions = tasks.map(task => ({ promise: runTask(task), waitForCompletion: task.waitForCompletion }))
      const mustWait = executions.some(task => task.waitForCompletion)

      if (!mustWait) {
        // Observe asynchronous no-wait cleanup so a late rejection cannot
        // become unhandled, but let Electron continue the original quit.
        void Promise.allSettled(executions.map(task => task.promise))
        finished = true

        return false
      }

      void Promise.allSettled(executions.map(task => task.promise)).then(() => {
        finished = true
        requestFinalQuit()
      })

      return true
    }
  }
}
