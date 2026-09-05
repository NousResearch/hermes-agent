import { host, type PluginOpenSessionOptions } from '@/sdk'
import type { AgentRow } from '@/store/agent-overview'
const noAmbientRequest = <T>(): Promise<T> => Promise.reject(new Error('An exact session owner is required.'))
import { sessionContextDrift } from '@/app/session/hooks/session-context-drift'
import { openGatewayForAgent } from '@/store/gateway'
import { $selectedStoredSessionId } from '@/store/session'
import { requestForSessionProfile } from '@/store/session-request-router'

interface ActionDependencies {
  request: (
    connectionId: string,
    profile: string,
    method: string,
    params: Record<string, unknown>,
    timeout: number
  ) => Promise<unknown>
  open: (id: string, options: PluginOpenSessionOptions) => Promise<void>
}

export function createOverviewActions({ request, open }: ActionDependencies) {
  const rpc = (row: AgentRow, method: string, params: Record<string, unknown>) =>
    // Shared remote serves multiplex profiles in the payload, not the socket URL.
    request(row.connectionId, row.profile, method, { ...params, profile: row.profile }, 20_000)

  const durableId = async (row: AgentRow): Promise<string> => {
    if (!row.canonical) {
      return row.resolvedId
    }

    const result = (await rpc(row, 'session.list', { title: 'Bot Chat', include_hidden: true })) as {
      sessions?: Array<{ id: string; resolved_id?: string; title?: string }>
    }

    const match = result.sessions?.find(session => session.title === 'Bot Chat')

    if (!match) {
      throw new Error('Canonical Bot Chat is unavailable. Retry after reconnecting its source.')
    }

    return match.resolved_id || match.id
  }

  return {
    async connect(connectionId: string) {
      let timer: ReturnType<typeof setTimeout> | undefined

      try {
        await Promise.race([
          openGatewayForAgent(connectionId, 'default'),
          new Promise<never>((_, reject) => {
            timer = setTimeout(
              () => reject(new Error('Connection timed out. Retry when this source is available.')),
              20_000
            )
          })
        ])
      } finally {
        clearTimeout(timer)
      }
    },
    async open(row: AgentRow, isCurrent: () => boolean = () => true) {
      const routeToken = () => window.location.hash.slice(1).split(/[?#]/, 1)[0] || '/'
      const startRouteToken = routeToken()
      const startSelectedStoredId = $selectedStoredSessionId.get()
      const id = await durableId(row)

      const openingIsCurrent = () =>
        isCurrent() &&
        !sessionContextDrift({
          startRouteToken,
          nowRouteToken: routeToken(),
          startSelectedStoredId,
          nowSelectedStoredId: $selectedStoredSessionId.get(),
          submitTargetStoredId: id
        })

      if (!openingIsCurrent()) {
        throw new Error('Session open was superseded by a newer selection.')
      }

      await open(id, {
        route: {
          connectionId: row.connectionId,
          profile: row.profile,
          targetProfile: row.profile,
          mode: row.connectionId === 'local' ? 'local' : 'remote'
        },
        keepAllProfilesScope: true,
        awaitHydration: true,
        forceResume: true,
        // Unknown/stale knowledge and a newly resolved canonical identity keep
        // the SDK's history paint guard. Only this known-empty chat waits on runtime.
        expectHistory: row.history !== 'known-empty' || row.stale || id !== row.resolvedId,
        isCurrent: openingIsCurrent,
        intent: 'main',
        workspaceMode: 'sessions',
        tabTitle: row.title
      })
    },
    async reply(row: AgentRow, text: string) {
      if (!text.trim()) {
        return
      }

      const id = await durableId(row)
      const result = (await rpc(row, 'session.resume', { session_id: id })) as { session_id?: string }

      if (!result.session_id) {
        throw new Error('The source did not return a live session. No reply was sent.')
      }

      await rpc(row, 'prompt.submit', { session_id: result.session_id, text })
    },
    async stop(row: AgentRow) {
      if (row.stale || !row.runtimeId || (row.attention !== 'working' && row.attention !== 'needs-you')) {
        throw new Error('No confirmed live turn to stop. Refresh this source first.')
      }

      await rpc(row, 'session.interrupt', { session_id: row.runtimeId })
    }
  }
}

export const overviewActions = createOverviewActions({
  request: (connectionId, profile, method, params, timeout) =>
    requestForSessionProfile({ connectionId, profile }, noAmbientRequest, method, params, timeout),
  open: host.openSession
})
