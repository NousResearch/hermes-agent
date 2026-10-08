import { useEffect, useMemo, useState } from 'react'

import { searchAllProfileSessions, searchSessions, type SessionSearchResult } from '@/hermes'
import { ALL_PROFILES, normalizeProfileKey } from '@/store/profile'
import type { ProfileInfo } from '@/types/hermes'

const SEARCH_DEBOUNCE_MS = 200

export interface ServerSessionSearch {
  serverMatches: SessionSearchResult[]
  searchPending: boolean
}

/**
 * Debounced full-text search across *all* sessions (not just the loaded page)
 * so old sessions stay findable. Hits come from the profiles the sidebar
 * shows: a concrete scope searches that profile's state.db; "All profiles"
 * fans one scoped request per profile, because the FTS endpoint is single-
 * store per call (unlike the cross-profile `/api/profiles/sessions` list) and
 * an unscoped request would read only the primary's launch profile.
 */
export function useServerSessionSearch(
  query: string,
  profileScope: string,
  profiles: readonly ProfileInfo[]
): ServerSessionSearch {
  const [serverMatches, setServerMatches] = useState<SessionSearchResult[]>([])
  const [searchPending, setSearchPending] = useState(false)
  // Only the ALL scope needs the roster; keep the list stable across renders
  // so the debounce effect doesn't re-fire on profile-store churn.
  const searchProfiles = useMemo(
    () => (profileScope === ALL_PROFILES ? profiles.map(profile => normalizeProfileKey(profile.name)) : []),
    [profileScope, profiles]
  )

  useEffect(() => {
    if (!query) {
      setServerMatches([])
      setSearchPending(false)

      return
    }

    let cancelled = false

    setSearchPending(true)

    const id = window.setTimeout(() => {
      const request =
        profileScope === ALL_PROFILES
          ? searchAllProfileSessions(query, searchProfiles)
          : searchSessions(query, normalizeProfileKey(profileScope))

      void request
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
    }, SEARCH_DEBOUNCE_MS)

    return () => {
      cancelled = true
      window.clearTimeout(id)
    }
  }, [query, profileScope, searchProfiles])

  return { searchPending, serverMatches }
}
