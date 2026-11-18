import { useEffect, useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'

import type { ConnectorCardModel } from '../types'

import { connectorToolsQueryOptions } from './queries'

const PREFETCH_LIMIT = 3

// Only a connected hosted account has tools to list; an unconnected card's read is a wasted gateway call.
export function hasConnectedTools(card: ConnectorCardModel): boolean {
  return card.ways.hosted?.state === 'connected'
}

export function prefetchConnectorTools(scope: ProfileScope, card: ConnectorCardModel): void {
  if (hasConnectedTools(card)) {
    void queryClient.prefetchQuery(connectorToolsQueryOptions(scope, card.slug))
  }
}

export function usePrefetchConnectedTools(scope: ProfileScope, cards: readonly ConnectorCardModel[]): void {
  const slugs = useMemo(() => cards.filter(hasConnectedTools).map(card => card.slug), [cards])

  useEffect(() => {
    let stopped = false
    const queue = [...slugs]

    const next = async (): Promise<void> => {
      const slug = queue.shift()

      if (slug === undefined || stopped) {
        return
      }

      await queryClient.prefetchQuery(connectorToolsQueryOptions(scope, slug))

      return next()
    }

    void Promise.all(Array.from({ length: Math.min(PREFETCH_LIMIT, queue.length) }, () => next()))

    return () => {
      stopped = true
    }
  }, [scope, slugs])
}
