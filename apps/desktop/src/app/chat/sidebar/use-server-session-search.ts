import { useEffect, useState } from 'react'

import { searchSessions, type SessionSearchResult } from '@/hermes'
import { ALL_PROFILES, normalizeProfileKey } from '@/store/profile'

const SEARCH_DEBOUNCE_MS = 200

/** A failed request is "could not search": "No sessions match" would claim the sessions do not exist. */
export function searchEmptyLabel(
  copy: { noMatch: (query: string) => string; searchFailed: string },
  query: string,
  failed: boolean
): string {
  return failed ? copy.searchFailed : copy.noMatch(query)
}

export interface ServerSessionSearch {
  serverMatches: SessionSearchResult[]
  searchPending: boolean
  /** The request failed: an empty list here means "could not search", not "no matches". */
  searchFailed: boolean
}

/**
 * Debounced full-text search across *all* sessions (not just the loaded page)
 * so old sessions stay findable. Hits come from the profile the sidebar shows:
 * an unscoped request searches the primary backend's launch profile, whatever
 * profile is on screen. "All profiles" asks the primary as before.
 */
export function useServerSessionSearch(query: string, profileScope: string): ServerSessionSearch {
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [searchPending, setSearchPending] = useState(false)
  const [searchFailed, setSearchFailed] = useState(false)
  const profile = profileScope === ALL_PROFILES ? null : normalizeProfileKey(profileScope)

  useEffect(() => {
    if (!query) {
      setServerMatches([])
      setSearchPending(false)
      setSearchFailed(false)

      return
    }

    let cancelled = false

    setSearchPending(true)

    const id = window.setTimeout(() => {
      void searchSessions(query, profile)
        .then(res => {
          if (!cancelled) {
            setServerMatches(res.results)
            setSearchFailed(false)
          }
        })
        .catch(() => {
          // Never keep the previous query's hits under this query's name.
          if (!cancelled) {
            setServerMatches([])
            setSearchFailed(true)
          }
        })
        .finally(() => {
          if (!cancelled) {
            setSearchPending(false)
          }
        })
    }, SEARCH_DEBOUNCE_MS)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [query, profile])

  return { searchFailed, searchPending, serverMatches }
}
