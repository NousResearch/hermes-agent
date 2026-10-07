import { useQuery } from '@tanstack/react-query'

import type { ProfileScope } from '@/rabbit'

import type { LocalServerInput } from '../types'

import { pluginServerRows } from './join'
import { CONNECTOR_GC_TIME, PLUGIN_SERVERS_STALE_TIME, pluginServersQueryKey } from './keys'
import { listMcpServers, mcpServerStatus } from './rpc'

const NO_PLUGIN_SERVERS: LocalServerInput[] = []

export function usePluginServers(scope: ProfileScope): LocalServerInput[] {
  const plugins = useQuery({
    gcTime: CONNECTOR_GC_TIME,
    queryFn: async () => {
      const [list, runtime] = await Promise.all([listMcpServers(scope), mcpServerStatus(scope)])

      return pluginServerRows({ runtime: runtime.servers, servers: list.servers })
    },
    queryKey: pluginServersQueryKey(scope),
    retry: false,
    staleTime: PLUGIN_SERVERS_STALE_TIME
  })

  return plugins.data ?? NO_PLUGIN_SERVERS
}
