import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import type { AgentPluginRow, GatewayRequest } from '@/store/agent-plugins'
import { requestGatewayForAgent } from '@/store/gateway'

interface PluginInventory {
  plugins: AgentPluginRow[]
  /** Missing/null means this backend cannot report its runtime state. */
  restart_required?: boolean | null
}

interface ToggleResult {
  ok: boolean
  plugin?: AgentPluginRow | null
  restart_required?: boolean | null
}

interface PluginToggle {
  key: string
  enable: boolean
  request: GatewayRequest
  queryKey: (string | null)[]
}

interface PluginSettingsScope {
  connectionId: string | null
  profile: string
  activeProfile: string
  enabled: boolean
}

/** Each inventory belongs to one backend, not the ambient settings-search cache. */
export function useAgentPluginSettings({ connectionId, profile, activeProfile, enabled }: PluginSettingsScope) {
  const { gateway } = useGatewayRequest()
  const client = useQueryClient()
  const queryKey = ['agent-plugin-settings', connectionId, profile]

  const request: GatewayRequest = useCallback(
    <T>(method: string, params: Record<string, unknown> = {}) => {
      if (connectionId) {
        return requestGatewayForAgent<T>(connectionId, profile, method, params)
      }

      if (!gateway) {
        return Promise.reject(new Error('Hermes gateway unavailable'))
      }

      return gateway.request<T>(method, profile === activeProfile ? params : { ...params, profile })
    },
    [connectionId, profile, activeProfile, gateway]
  )

  const inventory = useQuery({
    queryKey,
    queryFn: () => request<PluginInventory>('plugins.manage', { action: 'list' }),
    enabled,
    // Runtime status is cheap and must catch a restart performed outside Desktop.
    refetchInterval: enabled ? 5_000 : false
  })

  const toggle = useMutation({
    onMutate: (variables: PluginToggle) => client.cancelQueries({ queryKey: variables.queryKey }),
    mutationFn: async ({ key, enable, request }: PluginToggle) => {
      const result = await request<ToggleResult>('plugins.manage', { action: 'toggle', key, enable })

      if (!result.ok) {
        throw new Error('Plugin change was not saved')
      }

      return result
    },
    onSuccess: (result, variables) => {
      client.setQueryData<PluginInventory>(
        variables.queryKey,
        previous =>
          previous && {
            ...previous,
            restart_required: 'restart_required' in result ? result.restart_required : previous.restart_required,
            plugins: previous.plugins.map(row => (row.key === variables.key ? { ...row, ...result.plugin } : row))
          }
      )

      if (!result.plugin) {
        void client.invalidateQueries({ queryKey: variables.queryKey })
      }
    }
  })

  return {
    inventory,
    request,
    toggle: {
      isPending: toggle.isPending,
      mutateAsync: ({ key, enable }: { key: string; enable: boolean }) =>
        toggle.mutateAsync({ key, enable, request, queryKey })
    },
    recheck: async () => {
      await client.cancelQueries({ queryKey })

      const result = await client.fetchQuery({
        queryKey,
        queryFn: () => request<PluginInventory>('plugins.manage', { action: 'list' }),
        staleTime: 0
      })

      return result.restart_required === false
    }
  }
}
