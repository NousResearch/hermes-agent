import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import type { ProfileScope } from '@/hermes'
import { getServers, serverEnabled } from '@/lib/mcp-servers'
import {
  $agentPlugins,
  $agentPluginsOwnerKey,
  $agentPluginsProfile,
  $agentPluginsStatus,
  loadAgentPlugins
} from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useSettingsGatewayRequest } from '../settings/settings-owner-request'

import { useConnectorsListQuery } from './connectors/data/queries'

// Tab and sub-nav counts for the two capabilities whose lists are not one query (Skills and Tools
// count their own list queries). Each reads the same cache or store its page reads, so a count never
// fetches a second copy, and is undefined until known so the caller can reserve the slot.

/** Connectors that are on for this agent: signed-in hosted apps plus its enabled MCP servers. */
export function useConnectorsCount(scope: ProfileScope): number | undefined {
  const list = useConnectorsListQuery(scope)
  const config = useHermesConfigRecord(scope ?? undefined)

  if (!list.data || !config.data) {
    return undefined
  }

  const hosted = list.data.available === false ? 0 : list.data.connectors.filter(row => row.connected).length
  const servers = Object.values(getServers(config.data)).filter(serverEnabled).length

  return hosted + servers
}

const profileName = (scope: ProfileScope): null | string =>
  !scope ? null : typeof scope === 'string' ? scope : (scope.profile ?? null)

/** Plugins enabled for this agent. The plugin list is one owner-scoped store the Plugins page also
 *  loads; concurrent loads for the same owner share one request. */
export function usePluginsCount(scope: ProfileScope): number | undefined {
  const request = useSettingsGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const rows = useStore($agentPlugins)
  const ownerKey = useStore($agentPluginsOwnerKey)
  const loadedProfile = useStore($agentPluginsProfile)
  const status = useStore($agentPluginsStatus)
  const profile = profileName(scope)
  const owned = status === 'ready' && ownerKey === request.ownerKey && loadedProfile === profile

  useEffect(() => {
    if (gatewayState === 'open' && !owned) {
      void loadAgentPlugins(request, profile)
    }
  }, [gatewayState, owned, profile, request])

  return owned ? rows.filter(row => row.status === 'enabled').length : undefined
}
