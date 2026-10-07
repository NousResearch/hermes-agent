import { type ProfileScope, profileScopeKey } from '@/rabbit'
import { queryClient } from '@/lib/query-client'

export const CONNECTORS_QUERY_ROOT = 'connectors'

const MINUTE = 60 * 1000

export const CONNECTOR_GC_TIME = Number.POSITIVE_INFINITY

const scoped = (scope: ProfileScope, ...rest: string[]) =>
  [CONNECTORS_QUERY_ROOT, profileScopeKey(scope), ...rest] as const

export const pluginServersQueryKey = (scope: ProfileScope) => scoped(scope, 'plugin-servers')

export const PLUGIN_SERVERS_STALE_TIME = 5 * MINUTE

export const pluginProbeQueryKey = (scope: ProfileScope, name: string) => scoped(scope, 'plugin-probe', name)

export function invalidatePluginProbe(scope: ProfileScope, name: string): void {
  void queryClient.invalidateQueries({ queryKey: pluginProbeQueryKey(scope, name) })
}
