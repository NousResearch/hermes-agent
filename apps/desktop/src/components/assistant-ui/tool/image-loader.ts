import { useCallback, useEffect, useRef, useState } from 'react'

import { type DesktopFileOrigin, readDesktopFileDataUrl } from '@/lib/desktop-fs'
import { isInlineMediaSrc } from '@/lib/media'
import {
  getSessionOwnerHint,
  getSessionOwnerHints,
  ownerLookupSessionRows,
  sessionMatchesStoredId
} from '@/store/session'
import { $sessionTiles, knownOwnerForSession } from '@/store/session-states'

export const TOOL_IMAGE_READ_CONCURRENCY = 3

interface ImageReadJob {
  run: () => Promise<string>
  signal: AbortSignal
  resolve: (value: string) => void
  reject: (error: unknown) => void
  cancel: () => void
}

// Shared across galleries: opening several tool rows must not multiply the I/O limit.
const pendingReads: ImageReadJob[] = []
let activeReads = 0

function drainReads() {
  while (activeReads < TOOL_IMAGE_READ_CONCURRENCY && pendingReads.length) {
    const job = pendingReads.shift()!
    job.signal.removeEventListener('abort', job.cancel)

    if (job.signal.aborted) {
      job.reject(new DOMException('Image read cancelled', 'AbortError'))

      continue
    }

    activeReads += 1
    void Promise.resolve()
      .then(job.run)
      .then(job.resolve, job.reject)
      .finally(() => {
        activeReads -= 1
        drainReads()
      })
  }
}

function scheduleRead(run: () => Promise<string>, signal: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    const job: ImageReadJob = {
      run,
      signal,
      resolve,
      reject,
      cancel: () => {
        const index = pendingReads.indexOf(job)

        if (index >= 0) {
          pendingReads.splice(index, 1)
        }

        reject(new DOMException('Image read cancelled', 'AbortError'))
      }
    }

    if (signal.aborted) {
      job.cancel()

      return
    }

    signal.addEventListener('abort', job.cancel, { once: true })
    pendingReads.push(job)
    drainReads()
  })
}

export interface ToolImageContext {
  sessionId: string | null
  runtimeId: string | null
  origin?: DesktopFileOrigin
  revision?: number
}

function toolImageOrigin({ sessionId, runtimeId }: ToolImageContext): DesktopFileOrigin {
  if (!sessionId) {
    throw new Error('Image source session is unavailable')
  }

  const tiles = $sessionTiles.get().filter(tile => tile.storedSessionId === sessionId)
  const tileOwner = runtimeId ? tiles.find(tile => tile.runtimeId === runtimeId)?.ownerRoute : undefined

  const hint = getSessionOwnerHint(sessionId)

  const candidates = new Set([
    ...tiles.flatMap(tile =>
      tile.ownerRoute
        ? [JSON.stringify([tile.ownerRoute.connectionId ?? null, tile.ownerRoute.profile || 'default'])]
        : []
    ),
    ...getSessionOwnerHints(sessionId).map(owner =>
      JSON.stringify([owner.connectionId ?? null, owner.profile || 'default'])
    ),
    ...ownerLookupSessionRows()
      .filter(row => sessionMatchesStoredId(row, sessionId))
      .map(row => JSON.stringify([row.connection_id ?? null, row.profile || 'default']))
  ])

  // Cloned DBs can contain the same stored id on different hosts. Never choose an arbitrary row.
  if (!tileOwner && !hint && candidates.size > 1) {
    throw new Error('Image source owner is ambiguous')
  }

  const owner = tileOwner || hint || knownOwnerForSession(sessionId)

  if (!owner) {
    throw new Error('Image source session is unavailable')
  }

  return {
    sessionId,
    connectionId: typeof owner === 'string' ? undefined : owner.connectionId,
    profile: typeof owner === 'string' ? owner : owner.targetProfile || owner.profile
  }
}

export async function readToolImage(source: string, context: ToolImageContext): Promise<string> {
  if (isInlineMediaSrc(source)) {
    return source
  }

  const dataUrl = await readDesktopFileDataUrl(source, context.origin ?? toolImageOrigin(context))

  if (!dataUrl.startsWith('data:image/')) {
    throw new Error('Not an image')
  }

  return dataUrl
}

export interface ToolImageState {
  status: 'loading' | 'ready' | 'error'
  src?: string
}

/** Own only the current thumbnail page. Page changes evict bytes; collapse retains that bounded page. */
export function useToolImagePage(sources: string[], active: boolean, context: ToolImageContext) {
  const cache = useRef(new Map<string, ToolImageState>())
  const reads = useRef(new Map<string, AbortController>())
  const { sessionId, runtimeId, revision = 0 } = context
  const cacheRevision = useRef({ value: revision })
  const [snapshot, setSnapshot] = useState(() => ({ revision, images: new Map<string, ToolImageState>() }))
  const originSession = context.origin?.sessionId
  const originConnection = context.origin?.connectionId
  const originProfile = context.origin?.profile

  const load = useCallback(
    (source: string) => {
      const previous = cache.current.get(source)

      if (previous?.status === 'ready' || previous?.status === 'loading') {
        return
      }

      const controller = new AbortController()
      const signal = controller.signal
      reads.current.set(source, controller)
      cache.current.set(source, { status: 'loading' })
      setSnapshot({ revision, images: new Map(cache.current) })

      const read = () =>
        readToolImage(source, {
          sessionId,
          runtimeId,
          origin:
            originSession && originProfile
              ? { sessionId: originSession, connectionId: originConnection, profile: originProfile }
              : undefined
        })

      const promise = isInlineMediaSrc(source) ? read() : scheduleRead(read, signal)
      void promise
        .then(
          src => {
            if (signal.aborted) {
              return
            }

            cache.current.set(source, { status: 'ready', src })
            setSnapshot({ revision, images: new Map(cache.current) })
          },
          () => {
            if (signal.aborted) {
              return
            }

            cache.current.set(source, { status: 'error' })
            setSnapshot({ revision, images: new Map(cache.current) })
          }
        )
        .finally(() => {
          if (reads.current.get(source) === controller) {
            reads.current.delete(source)
          }
        })
    },
    [runtimeId, sessionId, originSession, originConnection, originProfile, revision]
  )

  useEffect(() => {
    const inFlight = reads.current

    if (cacheRevision.current.value !== revision) {
      cache.current.clear()
      cacheRevision.current.value = revision
    }

    for (const [source, state] of cache.current) {
      if (!sources.includes(source) || state.status !== 'ready') {
        cache.current.delete(source)
      }
    }

    setSnapshot({ revision, images: new Map(cache.current) })

    if (active) {
      sources.forEach(load)
    }

    return () => {
      for (const controller of inFlight.values()) {
        controller.abort()
      }

      inFlight.clear()
    }
  }, [active, load, sources, revision])

  const retry = useCallback(
    (source: string) => {
      if (active && sources.includes(source) && cache.current.get(source)?.status === 'error') {
        load(source)
      }
    },
    [active, load, sources]
  )

  const fail = useCallback(
    (source: string) => {
      cache.current.set(source, { status: 'error' })
      setSnapshot({ revision, images: new Map(cache.current) })
    },
    [revision]
  )

  return { images: snapshot.revision === revision ? snapshot.images : new Map<string, ToolImageState>(), retry, fail }
}
