type ModuleLoader = () => Promise<unknown>

interface IdleWarmupOptions {
  gapMs?: number
  idleTimeoutMs?: number
  initialDelayMs?: number
}

/**
 * Load code chunks one at a time after the foreground settles. Dynamic imports
 * are cached by the module loader, so opening a warmed surface avoids its
 * first-click parse/load pause without putting every route on the startup path.
 */
export function scheduleIdleWarmup(loaders: readonly ModuleLoader[], options: IdleWarmupOptions = {}): () => void {
  const initialDelayMs = options.initialDelayMs ?? 1_500
  const gapMs = options.gapMs ?? 250
  const idleTimeoutMs = options.idleTimeoutMs ?? 5_000
  let cancelled = false
  let idleHandle: null | number = null
  let timeoutHandle: null | number = null
  let index = 0

  const schedule = (delayMs: number) => {
    timeoutHandle = window.setTimeout(() => {
      timeoutHandle = null

      const run = () => {
        idleHandle = null

        if (cancelled || index >= loaders.length) {
          return
        }

        const loader = loaders[index]

        index += 1

        void Promise.resolve()
          .then(loader)
          .catch(() => undefined)
          .finally(() => {
            if (!cancelled && index < loaders.length) {
              schedule(gapMs)
            }
          })
      }

      if (window.requestIdleCallback) {
        idleHandle = window.requestIdleCallback(run, { timeout: idleTimeoutMs })
      } else {
        run()
      }
    }, delayMs)
  }

  if (loaders.length > 0) {
    schedule(initialDelayMs)
  }

  return () => {
    cancelled = true

    if (timeoutHandle !== null) {
      window.clearTimeout(timeoutHandle)
    }

    if (idleHandle !== null) {
      window.cancelIdleCallback(idleHandle)
    }
  }
}
