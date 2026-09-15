export type MermaidTheme = 'dark' | 'default'

interface MermaidRenderCacheOptions {
  defer?: (signal?: AbortSignal) => Promise<void>
  maxEntries: number
  render: (code: string, theme: MermaidTheme) => Promise<string>
  yieldToMainThread?: () => Promise<void>
}

interface MermaidRenderCache {
  render: (code: string, theme: MermaidTheme, signal?: AbortSignal) => Promise<string>
}

interface PendingRender {
  controller: AbortController
  observers: number
  promise: Promise<string>
  uncancellable: boolean
}

function abortError(): DOMException {
  return new DOMException('Mermaid render cancelled', 'AbortError')
}

function deferUntilIdle(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError())

      return
    }

    let idleId: number | undefined
    let timerId: ReturnType<typeof setTimeout> | undefined

    const cleanup = () => {
      signal?.removeEventListener('abort', onAbort)
    }

    const finish = () => {
      cleanup()
      resolve()
    }

    const onAbort = () => {
      if (idleId !== undefined && typeof cancelIdleCallback === 'function') {
        cancelIdleCallback(idleId)
      }

      if (timerId !== undefined) {
        clearTimeout(timerId)
      }

      cleanup()
      reject(abortError())
    }

    signal?.addEventListener('abort', onAbort, { once: true })

    if (typeof requestIdleCallback === 'function') {
      idleId = requestIdleCallback(finish, { timeout: 100 })
    } else {
      timerId = setTimeout(finish, 0)
    }
  })
}

function yieldTask(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0))
}

function observePending(entry: PendingRender, signal?: AbortSignal): Promise<string> {
  if (!signal) {
    entry.uncancellable = true

    return entry.promise
  }

  if (signal.aborted) {
    return Promise.reject(abortError())
  }

  entry.observers += 1

  return new Promise((resolve, reject) => {
    let observing = true

    const release = () => {
      if (!observing) {
        return
      }

      observing = false
      signal.removeEventListener('abort', onAbort)
      entry.observers -= 1

      if (entry.observers === 0 && !entry.uncancellable) {
        entry.controller.abort()
      }
    }

    const onAbort = () => {
      release()
      reject(abortError())
    }

    signal.addEventListener('abort', onAbort, { once: true })
    entry.promise.then(
      value => {
        if (!observing) {
          return
        }

        release()
        resolve(value)
      },
      error => {
        if (!observing) {
          return
        }

        release()
        reject(error)
      }
    )
  })
}

export function createRetryableLoader<T>(load: () => Promise<T>): () => Promise<T> {
  let current: Promise<T> | null = null

  return () => {
    if (current) {
      return current
    }

    const attempt = load()
    current = attempt
    void attempt.catch(() => {
      if (current === attempt) {
        current = null
      }
    })

    return attempt
  }
}

export function createMermaidRenderCache({
  defer = deferUntilIdle,
  maxEntries,
  render: renderSvg,
  yieldToMainThread = yieldTask
}: MermaidRenderCacheOptions): MermaidRenderCache {
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new Error('Mermaid render cache must contain at least one entry')
  }

  const completed = new Map<string, Promise<string>>()
  const inFlight = new Map<string, PendingRender>()
  let renderMutex = Promise.resolve()

  return {
    render(code, theme, signal) {
      if (signal?.aborted) {
        return Promise.reject(abortError())
      }

      const key = JSON.stringify([theme, code])
      const cached = completed.get(key)

      if (cached) {
        completed.delete(key)
        completed.set(key, cached)

        return cached
      }

      const existing = inFlight.get(key)

      if (existing?.controller.signal.aborted) {
        inFlight.delete(key)
      } else if (existing) {
        return observePending(existing, signal)
      }

      const controller = new AbortController()
      const admitted = defer(controller.signal)

      const promise = admitted.then(() => {
        const rendered = renderMutex.then(() => {
          if (controller.signal.aborted) {
            throw abortError()
          }

          return renderSvg(code, theme)
        })

        renderMutex = rendered.then(yieldToMainThread, yieldToMainThread).then(
          () => undefined,
          () => undefined
        )

        return rendered
      })

      const entry: PendingRender = { controller, observers: 0, promise, uncancellable: false }
      inFlight.set(key, entry)

      void promise.then(
        () => {
          if (inFlight.get(key) !== entry) {
            return
          }

          inFlight.delete(key)
          completed.set(key, promise)

          while (completed.size > maxEntries) {
            const oldest = completed.keys().next().value

            if (oldest === undefined) {
              break
            }

            completed.delete(oldest)
          }
        },
        () => {
          if (inFlight.get(key) === entry) {
            inFlight.delete(key)
          }
        }
      )

      return observePending(entry, signal)
    }
  }
}
