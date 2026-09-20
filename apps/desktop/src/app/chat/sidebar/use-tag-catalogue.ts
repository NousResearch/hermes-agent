import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { requestGatewayForAgent } from '@/store/gateway'
import { $profiles, $profileScope, ALL_PROFILES, normalizeProfileKey } from '@/store/profile'
import { $connection, $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

export function useTagCatalogue(open: boolean) {
  const connection = useStore($connection)
  const scope = useStore($profileScope)
  const profiles = useStore($profiles)
  const sessions = useStore($sessions)
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const archived = useStore($archivedSessions)
  const primary = connection?.connectionId ?? null
  const routes = new Map<string, [string | null, string]>()

  const add = (connectionId: string | null, profile: string) => {
    const route: [string | null, string] = [connectionId, normalizeProfileKey(profile)]

    if (scope === ALL_PROFILES || route[1] === scope) {
      routes.set(JSON.stringify(route), route)
    }
  }

  add(primary, connection?.profile || 'default')

  if (scope === ALL_PROFILES) {
    profiles.forEach(profile => add(primary, profile.name))
  }

  for (const row of [...sessions, ...cron, ...messaging, ...archived]) {
    add(row.connection_id ?? primary, row.profile || 'default')
  }

  const key = JSON.stringify([...routes.values()].sort())
  const [state, setState] = useState({ key: '', tags: [] as string[], error: false })
  useEffect(() => {
    if (!open) {
      return
    }

    let alive = true
    setState({ key, tags: [], error: false })
    const targets = JSON.parse(key) as [string | null, string][]
    void Promise.allSettled(
      targets.map(([id, profile]) =>
        requestGatewayForAgent<{ tags: string[] }>(id, profile, 'session.tags.list', { profile })
      )
    ).then(results => {
      if (alive) {
        setState({
          key,
          tags: [
            ...new Set(results.flatMap(result => (result.status === 'fulfilled' ? result.value.tags : [])))
          ].sort(),
          error: results.some(result => result.status === 'rejected')
        })
      }
    })

    return () => {
      alive = false
    }
  }, [key, open])

  return state.key === key ? state : { tags: [], error: false }
}
