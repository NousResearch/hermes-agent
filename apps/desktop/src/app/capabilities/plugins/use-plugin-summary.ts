import { useStore } from '@nanostores/react'
import { useEffect, useMemo } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { $pluginRecords } from '@/contrib/plugins-store'
import type { ProfileScope } from '@/hermes'
import {
  $agentPlugins,
  $agentPluginsStatus,
  type GatewayRequest,
  isDesktopRelevantPlugin,
  loadAgentPlugins
} from '@/store/agent-plugins'
import { requestGatewayForAgent } from '@/store/gateway'

import { mergePluginPackages } from './plugin-packages'

export interface PluginSummary {
  active: number
  loading: boolean
  total: number
}

const profileParam = (scope: ProfileScope): null | string => {
  if (!scope) {
    return null
  }

  return typeof scope === 'string' ? scope : (scope.profile ?? null)
}

/** Profile-aware plugin counts used by the Capabilities submenu.
 *
 * `active / total` means packages effective for the selected profile / packages
 * available on this Desktop+agent pair. A Desktop half is app-global, but when
 * it is enabled it is effective while using every profile; an Agent half is
 * scoped to the selected profile. Registry-scoped profiles route through their
 * owning connection instead of borrowing the active window gateway.
 */
export function usePluginSummary(profile: ProfileScope): PluginSummary {
  const { requestGateway } = useGatewayRequest()
  const desktopRecords = useStore($pluginRecords)
  const agentRows = useStore($agentPlugins)
  const status = useStore($agentPluginsStatus)
  const scope = profileParam(profile)

  const scopedRequest = useMemo<GatewayRequest>(() => {
    if (!profile || typeof profile === 'string') {
      return requestGateway
    }

    const connectionId = (profile.connectionId ?? '').trim() || null
    const profileName = (profile.profile ?? '').trim() || 'default'

    return (method, params = {}, timeoutMs) =>
      requestGatewayForAgent(connectionId, profileName, method, params, timeoutMs)
  }, [profile, requestGateway])

  useEffect(() => {
    void loadAgentPlugins(scopedRequest, scope)
  }, [scope, scopedRequest])

  const packages = useMemo(
    () => mergePluginPackages(Object.values(desktopRecords), agentRows.filter(isDesktopRelevantPlugin)),
    [agentRows, desktopRecords]
  )

  const active = packages.filter(pkg => {
    const desktopOn = pkg.desktop ? pkg.desktop.status !== 'disabled' : false
    const agentOn = pkg.agent?.status === 'enabled'

    return desktopOn || agentOn
  }).length

  return { active, loading: status === 'idle' || status === 'loading', total: packages.length }
}
