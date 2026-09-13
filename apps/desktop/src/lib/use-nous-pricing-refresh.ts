import { type Query, type QueryClient, type QueryKey, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import type { ModelOptionProvider } from '@/types/hermes'

const refreshOwners = new WeakMap<Query, { subscribers: number; stop: () => void }>()
const REFRESH_MS = 300_000
const PENDING_MS = 1_500
const MAX_PENDING_READS = 10

/** One completion-driven timer per cached catalog, shared by every editor of
 * that query. Query state owns data/errors and suppresses overlap with manual
 * refreshes, retries and offline requests. */
function watchPricing(query: Query, client: QueryClient) {
  let timer: ReturnType<typeof setTimeout> | undefined
  let pendingReads = 0

  const schedule = () => {
    clearTimeout(timer)

    if (query.state.fetchStatus !== 'idle') {
      return
    }

    const data = query.state.data as { providers?: ModelOptionProvider[] } | undefined
    const nous = data?.providers?.find(provider => provider.slug === 'nous' && !provider.free_tier_row)

    if (!nous) {
      return
    }

    const pending = !!(nous.pricing_pending || nous.free_tier_pending)

    if (!pending) {
      pendingReads = 0
    }

    const followUp = pending && pendingReads < MAX_PENDING_READS

    timer = setTimeout(
      () => {
        pendingReads = followUp ? pendingReads + 1 : 0
        // Keep a slow request running, including one started by another observer.
        // Background errors remain available on the catalog query for its UI.
        void query.fetch(undefined, { cancelRefetch: false }).catch(() => undefined)
      },
      followUp ? PENDING_MS : REFRESH_MS
    )
  }

  const unsubscribe = client.getQueryCache().subscribe(event => {
    if (event.query === query && event.type === 'updated') {
      schedule()
    }
  })

  schedule()

  return () => {
    clearTimeout(timer)
    unsubscribe()
  }
}

/** Subscribe only while the editor is active. The last subscriber releases
 * the timer; an owner/profile change subscribes to a different query object. */
export function useNousPricingRefresh({ queryKey, enabled = true }: { queryKey: QueryKey; enabled?: boolean }) {
  const client = useQueryClient()
  const query = client.getQueryCache().find({ queryKey, exact: true })

  useEffect(() => {
    if (!enabled || !query) {
      return
    }

    let owner = refreshOwners.get(query)

    if (!owner) {
      owner = { subscribers: 0, stop: watchPricing(query, client) }
      refreshOwners.set(query, owner)
    }

    owner.subscribers += 1

    return () => {
      owner.subscribers -= 1

      if (owner.subscribers === 0) {
        owner.stop()
        refreshOwners.delete(query)
      }
    }
  }, [client, enabled, query])
}
