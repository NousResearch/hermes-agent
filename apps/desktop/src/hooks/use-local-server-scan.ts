import type { ModelOptionsResult } from '@hermes/shared'
import { hashKey, type QueryKey, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef } from 'react'

import { modelOptionsQueryKey, type ModelOptionsRequest, requestModelOptions } from '@/lib/model-options'

interface LocalServerScan extends Pick<ModelOptionsRequest, 'gateway' | 'profile' | 'request' | 'sessionId'> {
  enabled: boolean
  ownerConnectionId?: null | string
}

// Catalog keys with an in-flight fetch that carries `detectLocal`, counted per key.
const scanningFetches = new Map<string, number>()

/** A picker's catalog query fn: a `detectLocal` read recorded while in flight,
 *  so an opened picker skips its own scan only behind work known to include it.
 *  The composer pill fetches the same key without scanning. */
export async function scanningModelOptions(
  queryKey: QueryKey,
  request: Omit<ModelOptionsRequest, 'detectLocal'>
): Promise<ModelOptionsResult> {
  const key = hashKey(queryKey)
  scanningFetches.set(key, (scanningFetches.get(key) ?? 0) + 1)

  try {
    return await requestModelOptions({ ...request, detectLocal: true })
  } finally {
    const left = (scanningFetches.get(key) ?? 1) - 1

    if (left > 0) {
      scanningFetches.set(key, left)
    } else {
      scanningFetches.delete(key)
    }
  }
}

/** An opened model picker scans this machine for local model servers (Ollama,
 *  LM Studio, vLLM). With nothing cached the picker's own catalog query asks
 *  for the scan (`detectLocal`). The composer pill subscribes to the same key
 *  WITHOUT scanning, though, so when it already cached the catalog (or is
 *  fetching it, a request the picker joins) the picker would just reuse it:
 *  this one-shot read on open scans and writes the answer, `Detected:` rows
 *  included, into that shared query. */
export function useLocalServerScan({
  enabled,
  gateway,
  ownerConnectionId,
  profile,
  request,
  sessionId
}: LocalServerScan): void {
  const queryClient = useQueryClient()
  // Callers may pass a fresh dispatcher every render; one scan per open, not per render.
  const route = useRef({ gateway, request })
  route.current = { gateway, request }

  useEffect(() => {
    if (!enabled) {
      return
    }

    const queryKey = modelOptionsQueryKey(profile, sessionId, ownerConnectionId)

    // A picker's own query is already fetching with `detectLocal`: a second read would repeat it.
    if (scanningFetches.has(hashKey(queryKey))) {
      return
    }

    const idle = () => queryClient.isFetching({ queryKey }) === 0

    // No catalog and no fetch: the picker's own query starts the scanning one.
    if (idle() && queryClient.getQueryData(queryKey) === undefined) {
      return
    }

    let live = true

    const scan = () =>
      requestModelOptions({ detectLocal: true, ...route.current, profile, sessionId })
        .then(next => {
          if (live) {
            queryClient.setQueryData<ModelOptionsResult>(queryKey, next)
          }
        })
        // A failed scan leaves the catalog the picker already shows.
        .catch(() => undefined)

    if (idle()) {
      void scan()

      return () => {
        live = false
      }
    }

    // A non-scanning fetch (the composer's) is in flight and the picker joined it: scan once it ends.
    const unsubscribe = queryClient.getQueryCache().subscribe(() => {
      if (live && idle()) {
        unsubscribe()
        void scan()
      }
    })

    return () => {
      live = false
      unsubscribe()
    }
  }, [enabled, ownerConnectionId, profile, queryClient, sessionId])
}
