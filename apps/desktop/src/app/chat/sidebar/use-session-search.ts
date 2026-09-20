import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { searchSessions, type SessionSearchResult } from '@/hermes'
import { $connection } from '@/store/session'
import { $sessionTagsRevision } from '@/store/session-tags'

export function useSessionSearch(trimmedQuery: string) {
  const tagsRevision = useStore($sessionTagsRevision)
  const connection = useStore($connection)
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [searchPending, setSearchPending] = useState(false)
  // Full-text search across *all* sessions (not just the loaded page) so 699
  // sessions stay findable. Debounced; loaded sessions are matched instantly
  // client-side and merged ahead of the server hits.
  useEffect(() => {
    if (!trimmedQuery) {
      setServerMatches([])
      setSearchPending(false)

      return
    }

    let cancelled = false

    setSearchPending(true)

    const id = window.setTimeout(() => {
      void searchSessions(trimmedQuery)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
          }
        })
        .catch(() => undefined)
        .finally(() => {
          if (!cancelled) {
            setSearchPending(false)
          }
        })
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [trimmedQuery, tagsRevision, connection?.connectionId, connection?.profile])

  return { serverMatches, searchPending }
}
