import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { requestGatewayForAgent } from '@/store/gateway'
import { $connection, $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

export function useTagCatalogue(open: boolean) {
  const connection = useStore($connection)
  const sessions = useStore($sessions)
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const archived = useStore($archivedSessions)
  const primary = connection?.connectionId ?? null
  const routes = new Map<string, [string | null, string]>()

  const add = (connectionId: string | null, profile: string) => {
    // Catalogue scope is the server, not the selected profile or loaded rows.
    const key = JSON.stringify(connectionId)

    if (!routes.has(key)) {
      routes.set(key, [connectionId, profile])
    }
  }

  add(primary, connection?.profile || 'default')

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
