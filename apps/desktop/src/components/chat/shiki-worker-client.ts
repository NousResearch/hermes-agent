import type { ShikiWorkerRequest, ShikiWorkerResponse, ShikiWorkerToken } from './shiki-worker'

export interface ShikiHighlightJob {
  dispose: () => void
  promise: Promise<ShikiWorkerToken[][]>
}

interface WorkerGeneration {
  id: number
  leases: number
  pendingIds: Set<number>
  retired: boolean
  worker: Worker
}

interface PendingRequest {
  generation: WorkerGeneration
  reject: (reason?: unknown) => void
  resolve: (tokens: ShikiWorkerToken[][]) => void
}

export class ShikiWorkerRetryableError extends Error {
  readonly retryable = true

  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'ShikiWorkerRetryableError'
  }
}

let nextGenerationId = 0
let nextRequestId = 0
let currentGeneration: WorkerGeneration | null = null
const pending = new Map<number, PendingRequest>()

function rejectGenerationPending(generation: WorkerGeneration, error: Error) {
  for (const id of generation.pendingIds) {
    const request = pending.get(id)

    if (request?.generation === generation) {
      pending.delete(id)
      request.reject(error)
    }
  }

  generation.pendingIds.clear()
}

function retireGeneration(generation: WorkerGeneration, error: Error) {
  if (generation.retired) {
    return
  }

  generation.retired = true

  if (currentGeneration === generation) {
    currentGeneration = null
  }

  generation.worker.onmessage = null
  generation.worker.onerror = null
  generation.worker.terminate()
  rejectGenerationPending(generation, error)
}

function createGeneration(): WorkerGeneration {
  let worker: Worker

  try {
    worker = new Worker(new URL('./shiki-worker.ts', import.meta.url), { type: 'module' })
  } catch (cause) {
    throw new ShikiWorkerRetryableError('Shiki worker construction failed', { cause })
  }

  const generation: WorkerGeneration = {
    id: ++nextGenerationId,
    leases: 0,
    pendingIds: new Set(),
    retired: false,
    worker
  }

  generation.worker.onmessage = (event: MessageEvent<ShikiWorkerResponse>) => {
    const request = pending.get(event.data.id)

    if (!request || request.generation !== generation) {
      return
    }

    pending.delete(event.data.id)
    generation.pendingIds.delete(event.data.id)

    if (event.data.error || !event.data.tokens) {
      request.reject(new Error(event.data.error || 'Shiki worker returned no tokens'))
    } else {
      request.resolve(event.data.tokens)
    }
  }

  generation.worker.onerror = event => {
    retireGeneration(generation, new ShikiWorkerRetryableError(event.message || 'Shiki worker failed'))
  }

  currentGeneration = generation

  return generation
}

function getGeneration(): WorkerGeneration {
  return currentGeneration ?? createGeneration()
}

export function isShikiWorkerRetryableError(error: unknown): error is ShikiWorkerRetryableError {
  return (
    error instanceof ShikiWorkerRetryableError ||
    (error instanceof Error && 'retryable' in error && error.retryable === true)
  )
}

/** Narrow behavior seam for lifecycle tests; never exposes worker instances. */
export function getShikiWorkerClientSnapshotForTests() {
  return {
    activeGeneration: currentGeneration?.id ?? null,
    activeLeases: currentGeneration?.leases ?? 0,
    pending: pending.size
  }
}

export function startShikiHighlight(code: string, language: string): ShikiHighlightJob {
  if (typeof Worker === 'undefined') {
    return {
      dispose: () => {},
      promise: Promise.reject(new Error('Web Workers are unavailable'))
    }
  }

  let generation: WorkerGeneration

  try {
    generation = getGeneration()
  } catch (error) {
    return { dispose: () => {}, promise: Promise.reject(error) }
  }

  const id = ++nextRequestId
  generation.leases += 1
  let disposed = false

  const promise = new Promise<ShikiWorkerToken[][]>((resolve, reject) => {
    pending.set(id, { generation, reject, resolve })
    generation.pendingIds.add(id)

    try {
      generation.worker.postMessage({ code, id, language } satisfies ShikiWorkerRequest)
    } catch (error) {
      retireGeneration(generation, new ShikiWorkerRetryableError('Shiki worker postMessage failed', { cause: error }))
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

      if (request?.generation === generation) {
        pending.delete(id)
        generation.pendingIds.delete(id)
        request.reject(new Error('Shiki highlight cancelled'))
      }

      generation.leases = Math.max(0, generation.leases - 1)

      if (generation.leases === 0 && !generation.retired) {
        retireGeneration(generation, new Error('Shiki worker has no active consumers'))
      }
    }
  }
}
