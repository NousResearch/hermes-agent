import type { ShikiWorkerRequest, ShikiWorkerResponse, ShikiWorkerToken } from './shiki-worker'

export interface ShikiHighlightJob {
  dispose: () => void
  promise: Promise<ShikiWorkerToken[][]>
}

let nextRequestId = 0
let worker: Worker | null = null
let consumers = 0

const pending = new Map<
  number,
  { reject: (reason?: unknown) => void; resolve: (tokens: ShikiWorkerToken[][]) => void }
>()

function rejectPending(error: Error) {
  for (const request of pending.values()) {
    request.reject(error)
  }

  pending.clear()
}

function terminateWorker(error: Error) {
  const activeWorker = worker
  worker = null

  if (activeWorker) {
    activeWorker.onmessage = null
    activeWorker.onerror = null
    activeWorker.terminate()
  }

  rejectPending(error)
}

function getWorker(): Worker {
  if (worker) {
    return worker
  }

  worker = new Worker(new URL('./shiki-worker.ts', import.meta.url), { type: 'module' })

  worker.onmessage = (event: MessageEvent<ShikiWorkerResponse>) => {
    const request = pending.get(event.data.id)

    if (!request) {
      return
    }

    pending.delete(event.data.id)

    if (event.data.error || !event.data.tokens) {
      request.reject(new Error(event.data.error || 'Shiki worker returned no tokens'))
    } else {
      request.resolve(event.data.tokens)
    }
  }

  worker.onerror = event => {
    terminateWorker(new Error(event.message || 'Shiki worker failed'))
  }

  return worker
}

export function startShikiHighlight(code: string, language: string): ShikiHighlightJob {
  if (typeof Worker === 'undefined') {
    return {
      dispose: () => {},
      promise: Promise.reject(new Error('Web Workers are unavailable'))
    }
  }

  const id = ++nextRequestId
  consumers += 1
  let disposed = false

  const promise = new Promise<ShikiWorkerToken[][]>((resolve, reject) => {
    pending.set(id, { reject, resolve })

    try {
      getWorker().postMessage({ code, id, language } satisfies ShikiWorkerRequest)
    } catch (error) {
      pending.delete(id)
      reject(error)
    }
  })

  return {
    promise,
    dispose: () => {
      if (disposed) {
        return
      }

      disposed = true
      const request = pending.get(id)

      if (request) {
        pending.delete(id)
        request.reject(new Error('Shiki highlight cancelled'))
      }

      consumers = Math.max(0, consumers - 1)

      if (consumers === 0) {
        terminateWorker(new Error('Shiki worker has no active consumers'))
      }
    }
  }
}
