/**
 * Hook for off-main-thread Shiki tokenization via Web Worker
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { BundledLanguage, ThemedToken } from 'shiki'

interface WorkerMessage<T = unknown> {
  type: 'tokenize' | 'tokenizeChunk' | 'cancel' | 'init' | 'reinit'
  id: number
  payload?: T
}

interface WorkerResponse<T = unknown> {
  type: 'result' | 'error' | 'progress' | 'init'
  id: number
  payload?: T
  error?: string
  success?: boolean
}

interface TokenizeResult {
  tokens: ThemedToken[][]
}

interface TokenizeChunkResult {
  tokens: ThemedToken[][]
  chunkIndex: number
}

interface UseShikiWorkerOptions {
  language: BundledLanguage
  theme?: 'github-dark-dimmed' | 'github-light-default'
}

interface UseShikiWorkerReturn {
  tokenize: (code: string) => Promise<ThemedToken[][] | null>
  tokenizeChunks: (chunks: Array<{ code: string; chunkIndex: number }>) => Promise<Map<number, ThemedToken[][]>>
  isReady: boolean
  error: string | null
  terminate: () => void
  cancel: (chunkIndex?: number) => void
}

// Use a more reliable worker URL - absolute path from origin
const getWorkerUrl = () => {
  const base = typeof window !== 'undefined' ? window.location.origin : ''

  return `${base}/shiki-worker.js`
}

const WORKER_URL = typeof window !== 'undefined' ? getWorkerUrl() : new URL('./shiki-worker.ts', import.meta.url).toString()

// Request ID with timestamp to prevent collisions
const generateRequestId = () => {
  return Date.now() << 10 | (Math.random() * 1024) >>> 0
}

// Max chunk size to prevent DoS
const MAX_CHUNK_SIZE = 50000 // ~50KB per chunk
const MAX_CONCURRENT_CHUNKS = 20

export function useShikiWorker({ language, theme = 'github-dark-dimmed' }: UseShikiWorkerOptions): UseShikiWorkerReturn {
  const workerRef = useRef<Worker | null>(null)
  const pendingRef = useRef<Map<number, { resolve: (value: ThemedToken[][] | null) => void; reject: (error: Error) => void; type: 'tokenize' | 'tokenizeChunk'; chunkIndex?: number }>>(new Map())
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isInitializedRef = useRef(false)
  const initIdRef = useRef(0)

  // Initialize worker
  // eslint-disable-next-line no-restricted-syntax -- Worker initialization guard ref is not an atom mirror
  useEffect(() => {
    if (isInitializedRef.current) {return}
    isInitializedRef.current = true

    const worker = new Worker(WORKER_URL, { type: 'module' })
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const { type, id, payload, error: workerError, success } = event.data

      if (type === 'init') {
        if (success) {
          setIsReady(true)
          setError(null)
        } else {
          setError(workerError || 'Worker initialization failed')
          setIsReady(false)
        }

        return
      }

      const pending = pendingRef.current.get(id)

      if (!pending) {
        return
      }

      pendingRef.current.delete(id)

      if (type === 'result') {
        const typedPayload = payload as TokenizeResult | TokenizeChunkResult
        pending.resolve(typedPayload.tokens)
      } else if (type === 'error') {
        pending.reject(new Error(workerError || 'Unknown worker error'))
      }
    }

    worker.onerror = (err) => {
      setError(`Worker error: ${err.message}`)
      setIsReady(false)
      // Reject all pending promises
      pendingRef.current.forEach(({ reject }) => {
        reject(new Error(`Worker error: ${err.message}`))
      })
      pendingRef.current.clear()
    }

    // Handle worker termination - reject all pending
    const handleWorkerClose = () => {
      setIsReady(false)
      pendingRef.current.forEach(({ reject }) => {
        reject(new Error('Worker terminated'))
      })
      pendingRef.current.clear()
    }

    worker.addEventListener('error', handleWorkerClose)
    worker.addEventListener('messageerror', handleWorkerClose)

    // Initialize worker
    initIdRef.current = generateRequestId()
    worker.postMessage({ type: 'init', id: initIdRef.current } as WorkerMessage)

    return () => {
      worker.removeEventListener('error', handleWorkerClose)
      worker.removeEventListener('messageerror', handleWorkerClose)
      worker.terminate()
      workerRef.current = null
      setIsReady(false)
      // Reject all pending on cleanup
      pendingRef.current.forEach(({ reject }) => {
        reject(new Error('Worker terminated'))
      })
      pendingRef.current.clear()
    }
  }, [])

  // Re-initialize worker with new theme/language
  const reinitWorker = useCallback(() => {
    if (!workerRef.current || !isReady) {return}

    const newInitId = generateRequestId()
    initIdRef.current = newInitId
    workerRef.current.postMessage({ 
      type: 'reinit', 
      id: newInitId,
      payload: { language, theme }
    } as WorkerMessage)
  }, [language, theme, isReady])

  // Watch for theme/language changes
  useEffect(() => {
    if (isReady) {
      reinitWorker()
    }
  }, [language, theme, isReady, reinitWorker])

  // Single tokenization
  const tokenize = useCallback(async (code: string): Promise<ThemedToken[][] | null> => {
    if (!workerRef.current || !isReady) {
      throw new Error('Worker not ready')
    }

    if (code.length > MAX_CHUNK_SIZE) {
      throw new Error(`Code exceeds max chunk size (${MAX_CHUNK_SIZE} chars)`)
    }

    const id = generateRequestId()

    return new Promise<ThemedToken[][] | null>((resolve, reject) => {
      pendingRef.current.set(id, { resolve, reject, type: 'tokenize' })

      workerRef.current!.postMessage({
        type: 'tokenize',
        id,
        payload: { code, language, theme }
      } as WorkerMessage)
    }) as Promise<ThemedToken[][] | null>
  }, [language, theme, isReady])

  // Chunk tokenization - processes multiple chunks in parallel with limits
  const tokenizeChunks = useCallback(async (
    chunks: Array<{ code: string; chunkIndex: number }>
  ): Promise<Map<number, ThemedToken[][]>> => {
    if (!workerRef.current || !isReady) {
      throw new Error('Worker not ready')
    }

    // Filter empty chunks and validate size
    const validChunks = chunks
      .filter(({ code }) => code.length > 0 && code.length <= MAX_CHUNK_SIZE)
      .slice(0, MAX_CONCURRENT_CHUNKS)

    if (validChunks.length === 0) {
      return new Map()
    }

    const results = new Map<number, ThemedToken[][]>()

    const promises = validChunks.map(({ code, chunkIndex }) => {
      const id = generateRequestId()

      return new Promise<{ chunkIndex: number; tokens: ThemedToken[][] }>((resolve, reject) => {
        pendingRef.current.set(id, { 
          resolve: (payload) => {
            const typed = payload as unknown as TokenizeResult | TokenizeChunkResult
            resolve({ chunkIndex: 'chunkIndex' in typed ? typed.chunkIndex : chunkIndex, tokens: typed.tokens })
          }, 
          reject,
          type: 'tokenizeChunk',
          chunkIndex
        })

        workerRef.current!.postMessage({
          type: 'tokenizeChunk',
          id,
          payload: { code, language, theme, chunkIndex }
        } as WorkerMessage)
      })
    })

    const chunkResults = await Promise.all(promises)
    chunkResults.forEach(({ chunkIndex, tokens }) => {
      results.set(chunkIndex, tokens)
    })

    return results
  }, [language, theme, isReady])

  // Cancel specific chunk or all pending
  const cancel = useCallback((chunkIndex?: number) => {
    if (!workerRef.current || !isReady) {return}

    if (chunkIndex !== undefined) {
      // Cancel specific chunk - find by chunkIndex
      pendingRef.current.forEach((pending, id) => {
        if (pending.chunkIndex === chunkIndex) {
          pending.reject(new Error('Cancelled'))
          pendingRef.current.delete(id)
          workerRef.current!.postMessage({ type: 'cancel', id, payload: { chunkIndex } } as WorkerMessage)
        }
      })
    } else {
      // Cancel all
      pendingRef.current.forEach((_, id) => {
        workerRef.current!.postMessage({ type: 'cancel', id } as WorkerMessage)
      })
      pendingRef.current.forEach(({ reject }) => {
        reject(new Error('Cancelled'))
      })
      pendingRef.current.clear()
    }
  }, [isReady])

  const terminate = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate()
      workerRef.current = null
      setIsReady(false)
      pendingRef.current.forEach(({ reject }) => {
        reject(new Error('Worker terminated'))
      })
      pendingRef.current.clear()
    }
  }, [])

  return { tokenize, tokenizeChunks, isReady, error, terminate, cancel }
}

// Simpler hook for single-shot tokenization (used by DiffLines compact mode)
export function useShikiTokenize() {
  const { tokenize, isReady, error, terminate, cancel } = useShikiWorker({ language: 'txt' as BundledLanguage })
  
  return { tokenize, isReady, error, terminate, cancel }
}