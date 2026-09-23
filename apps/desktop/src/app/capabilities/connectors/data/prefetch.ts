import { useEffect, useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'

import type { ConnectorCardModel } from '../types'

import { connectorToolsQueryOptions } from './queries'

const PREFETCH_LIMIT = 3

export function prefetchConnectorTools(scope: ProfileScope, slug: string): void {
  void queryClient.prefetchQuery(connectorToolsQueryOptions(scope, slug))
}

export function usePrefetchConnectedTools(scope: ProfileScope, cards: readonly ConnectorCardModel[]): void {
  const slugs = useMemo(
    () => cards.filter(card => card.ways.hosted?.state === 'connected').map(card => card.slug),
    [cards]
  )

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
