import { useInfiniteQuery } from '@tanstack/react-query'

import { type ProfileScope, profileScopeKey } from '@/api/client'
import { searchProvider } from '@/api/skill-catalog'
import type { CatalogSource } from '@/types/skill-catalog'

import { parseCatalog } from './catalog-data'

/** Provider pages have their own scoped cache; public discovery never waits on them. */
export function useProviderCatalog(sources: CatalogSource[], profile: ProfileScope, query: string) {
  const ready = sources.filter(source => source.state === 'ready' && source.searchable)

  const result = useInfiniteQuery({
    queryKey: ['skill-provider-catalog', profileScopeKey(profile), ready.map(source => source.id), query],
    queryFn: async ({ pageParam }) => Promise.all(Object.entries(pageParam).map(async ([source, cursor]) => {
      try { return { source, ...await searchProvider(profile, source, query, cursor) } }
      catch (error) { return { source, results: [], next_cursor: null, provider_errors: [{ source, code: 'provider_unavailable', message: String(error instanceof Error ? error.message : error) }] } }
    })),
    initialPageParam: Object.fromEntries(ready.map(source => [source.id, null])) as Record<string, string | null>,
    getNextPageParam: (last, pages) => {
      const next = Object.fromEntries(last.filter(page => page.next_cursor && !pages.slice(0, -1).flat().some(previous => previous.source === page.source && previous.next_cursor === page.next_cursor)).map(page => [page.source, page.next_cursor!]))

      return Object.keys(next).length ? next : undefined
    },
    enabled: ready.length > 0,
    staleTime: 60_000,
    retry: false
  })

  const pages = result.data?.pages.flat() ?? []

  const entries = parseCatalog('skills', pages.flatMap(page => page.results).map(row => ({
    ...row, author: row.provider?.author, version: row.provider?.version, installIdentifier: row.identifier
  })))

  const failures = pages.flatMap(page => [
    ...(page.provider_errors ?? []).map(error => error.message),
    ...(page.timed_out ?? [])
  ])

  return { ...result, entries, failures }
}
