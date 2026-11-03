import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { capabilityScoped, hermesApi } from '@/api/client'
import type { SessionInfo, SessionSearchResponse } from '@/hermes'
import { $fleetRoster } from '@/store/fleet-roster'
import { $profiles } from '@/store/profile'

import { searchResultToSession } from './search-result-to-session'
import { useSimplifiedOwner } from './use-simplified-agents'

/** Only v2 searches the union of agents; classic keeps its existing search contract. */
export function useSimplifiedSessionSearch(query: string) {
  const profiles = useStore($profiles)
  const roster = useStore($fleetRoster)
  const owner = useSimplifiedOwner()
  const [state, setState] = useState({ rows: [] as SessionInfo[], pending: false, failed: false })
  const scopes = new Map<string, { connectionId?: string; profile: string }>()
  const connectionId = owner ?? undefined

  for (const p of profiles) {
    const scope = { connectionId, profile: p.name }
    scopes.set(JSON.stringify(scope), scope)
  }

  for (const agent of roster?.agents ?? []) {
    const scope = { connectionId: agent.connectionId, profile: agent.profile }
    scopes.set(JSON.stringify(scope), scope)
  }

  const scopeKey = JSON.stringify([...scopes.values()])
  useEffect(() => {
    let cancelled = false
    setState({ rows: [], pending: Boolean(query), failed: false })

    if (!query) {
      return
    }

    const targets: { connectionId?: string; profile: string }[] = JSON.parse(scopeKey)

    const timer = window.setTimeout(async () => {
      const results = await Promise.allSettled(
        targets.map(async scope => {
          const response = await hermesApi<SessionSearchResponse>({
            ...capabilityScoped(scope),
            path: `/api/sessions/search?q=${encodeURIComponent(query)}&profile=${encodeURIComponent(scope.profile)}`
          })

          return response.results.map(hit => ({
            ...searchResultToSession(hit),
            profile: scope.profile,
            connection_id: scope.connectionId
          }))
        })
      )

      if (!cancelled) {
        setState({
          rows: results.flatMap(result => (result.status === 'fulfilled' ? result.value : [])),
          pending: false,
          failed: results.some(result => result.status === 'rejected')
        })
      }
    }, 200)

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [query, scopeKey])

  return state
}
