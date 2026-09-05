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
  let completion: Promise<void> | null = null
  let finished = false

  return {
    begin(tasks): boolean {
      if (finished) {
        return false
      }

      if (completion) {
        return true
      }

      const executions = tasks.map(task => ({ promise: runTask(task), waitForCompletion: task.waitForCompletion }))
      const mustWait = executions.some(task => task.waitForCompletion)

      if (!mustWait) {
        // Observe asynchronous no-wait cleanup so a late rejection cannot
        // become unhandled, but let Electron continue the original quit.
        void Promise.allSettled(executions.map(task => task.promise))
        finished = true

        return false
      }

      completion = Promise.allSettled(executions.map(task => task.promise)).then(() => {
        finished = true
        requestFinalQuit()
      })

      return true
    }
  }
}
