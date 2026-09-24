import { type ProfileScope, profileScopeKey } from '@/hermes'
import { queryClient } from '@/lib/query-client'

export type ConnectorRead = 'accounts' | 'catalog' | 'list' | 'policy' | 'tools'

export const CONNECTORS_QUERY_ROOT = 'connectors'

const SECOND = 1000
const MINUTE = 60 * SECOND
const HOUR = 60 * MINUTE

interface ReadLifetime {
  focusRefetch: boolean
  persist: boolean
  staleTime: number
}

export const CONNECTOR_LIFETIMES = {
  accounts: { focusRefetch: true, persist: false, staleTime: 5 * MINUTE },
  catalog: { focusRefetch: false, persist: true, staleTime: 24 * HOUR },
  list: { focusRefetch: true, persist: true, staleTime: 5 * MINUTE },
  policy: { focusRefetch: false, persist: false, staleTime: 60 * SECOND },
  tools: { focusRefetch: false, persist: true, staleTime: 24 * HOUR }
} satisfies Record<ConnectorRead, ReadLifetime>

export const CONNECTOR_GC_TIME = Number.POSITIVE_INFINITY

const scoped = (scope: ProfileScope, ...rest: string[]) =>
  [CONNECTORS_QUERY_ROOT, profileScopeKey(scope), ...rest] as const

export const connectorsListQueryKey = (scope: ProfileScope) => scoped(scope, 'list')

export const connectorsCatalogQueryKey = (scope: ProfileScope) => scoped(scope, 'catalog')

export const connectorsAccountsQueryKey = (scope: ProfileScope) => scoped(scope, 'accounts')

export const connectorsPolicyQueryKey = (scope: ProfileScope) => scoped(scope, 'policy')

export const connectorToolsQueryKey = (scope: ProfileScope, slug: string) => scoped(scope, 'tools', slug)

export const pluginServersQueryKey = (scope: ProfileScope) => scoped(scope, 'plugin-servers')

export const PLUGIN_SERVERS_STALE_TIME = 5 * MINUTE

export const pluginProbeQueryKey = (scope: ProfileScope, name: string) => scoped(scope, 'plugin-probe', name)

export function invalidatePluginProbe(scope: ProfileScope, name: string): void {
  void queryClient.invalidateQueries({ queryKey: pluginProbeQueryKey(scope, name) })
}

export function invalidateConnectors(scope: ProfileScope, ...reads: [ConnectorRead, ...ConnectorRead[]]): void {
  for (const read of reads) {
    void queryClient.invalidateQueries({ queryKey: scoped(scope, read) })
  }
}

export function invalidateConnectorApp(scope: ProfileScope, slug: string): void {
  void queryClient.invalidateQueries({ queryKey: connectorToolsQueryKey(scope, slug) })
  void queryClient.invalidateQueries({ queryKey: connectorsListQueryKey(scope) })
}
