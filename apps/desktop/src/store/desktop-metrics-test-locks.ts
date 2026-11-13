import { vi } from 'vitest'

import type { captureDesktopMetricsGate, DesktopMetricsGate } from './desktop-metrics'

/** Use the production consent seam, bound to the renderer module actually under test. */
export async function setMetricsGate(
  capture: typeof captureDesktopMetricsGate,
  next: DesktopMetricsGate,
  decided = true
): Promise<boolean> {
  return (await capture())(next, decided)
}

/** Deterministic idle Web Lock acquisition for tests that advance the wall clock synchronously. */
export function immediateMetricsLocks(): void {
  const waiting: (() => void)[] = []
  let owned = false

  const release = () => {
    owned = false
    waiting.shift()?.()
  }

  vi.stubGlobal('navigator', { locks: {
    request: (_key: string, run: () => unknown) => {
      const result = Promise.withResolvers<unknown>()

      const acquire = () => {
        owned = true

        try {
          const value = run()

          if (value instanceof Promise) {
            void value.then(result.resolve, result.reject).finally(release)
          } else {
            result.resolve(value)
            release()
          }
        } catch (error) {
          result.reject(error)
          release()
        }
      }

      if (owned) {
        waiting.push(acquire)
      } else {
        acquire()
      }

      return result.promise
    }
  } })
}

/** Only the browser transaction API is doubled: real metric journals, snapshots and backend-latch
 * assertions remain in the tests. Requests acquire ownership asynchronously, one transaction at
 * a time, and completion/abort release it before the next transaction's request can run. */
export function indexedDbMetricsLocks(): { abortNext: (after?: number) => void; close: () => void } {
  interface Request { onsuccess?: () => void }
  interface Transaction {
    aborted: boolean
    onabort?: () => void
    oncomplete?: () => void
    request?: Request
  }
  const waiting: Transaction[] = []
  let busy = false
  let abortAfter: number | undefined

  const pump = () => {
    if (busy || waiting.length === 0) {
      return
    }

    busy = true
    const transaction = waiting.shift()!

    setTimeout(() => {
      if (abortAfter === 0) {
        abortAfter = undefined
        transaction.aborted = true
      } else {
        if (abortAfter !== undefined) {
          abortAfter -= 1
        }

        transaction.request?.onsuccess?.()
      }

      setTimeout(() => {
        if (transaction.aborted) {
          transaction.onabort?.()
        } else {
          transaction.oncomplete?.()
        }

        busy = false
        pump()
      }, 0)
    }, 0)
  }

  const database: {
    close: () => void
    createObjectStore: () => void
    onversionchange?: () => void
    transaction: () => unknown
  } = {
    close: () => undefined,
    createObjectStore: () => undefined,
    transaction: () => {
      const transaction: Transaction = { aborted: false }

      return Object.assign(transaction, {
        abort: () => { transaction.aborted = true },
        objectStore: () => ({ get: () => {
          const request: Request = {}

          transaction.request = request
          waiting.push(transaction)
          pump()

          return request
        } })
      })
    }
  }

  vi.stubGlobal('navigator', {})
  vi.stubGlobal('indexedDB', { open: () => {
    const request: Request & { onupgradeneeded?: () => void; result: typeof database } = { result: database }

    setTimeout(() => {
      request.onupgradeneeded?.()
      request.onsuccess?.()
    }, 0)

    return request
  } })

  return {
    abortNext: (after = 0) => { abortAfter = after },
    close: () => database.onversionchange?.()
  }
}
