import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'

import type { ProfileScope } from '@/hermes'
import { getServers, serverEnabled } from '@/lib/mcp-servers'
import {
  $agentPlugins,
  $agentPluginsOwnerKey,
  $agentPluginsProfile,
  $agentPluginsStatus,
  isDesktopRelevantPlugin,
  loadAgentPlugins
} from '@/store/agent-plugins'
import { $gatewayState } from '@/store/session'

import { useHermesConfigRecord } from '../hooks/use-config-record'
import { useSettingsGatewayRequest } from '../settings/settings-owner-request'

import { useConnectorsListQuery } from './connectors/data/queries'

// Tab and sub-nav counts for the two capabilities whose lists are not one query (Skills and Tools
// count their own list queries). Each reads the same cache or store its page reads, so a count never
// fetches a second copy. A count is `value` once known; until then the caller reserves the slot, and
// when the agent's backend cannot answer it is `failed` (with a `retry`) instead of loading forever.

export interface CapabilityCount {
  value?: number
  failed: boolean
  retry: () => void
}

/** Connectors that are on for this agent: signed-in hosted apps plus its enabled MCP servers. */
export function useConnectorsCount(scope: ProfileScope): CapabilityCount {
  const list = useConnectorsListQuery(scope)
  const config = useHermesConfigRecord(scope ?? undefined)

  const retry = () => {
    void list.refetch()
    void config.refetch()
  }

  if (!list.data || !config.data) {
    return { failed: (!list.data && list.isError) || (!config.data && config.isError), retry }
  }

  const hosted = list.data.available === false ? 0 : list.data.connectors.filter(row => row.connected).length
  const servers = Object.values(getServers(config.data)).filter(serverEnabled).length

  return { value: hosted + servers, failed: false, retry }
}

const profileName = (scope: ProfileScope): null | string =>
  !scope ? null : typeof scope === 'string' ? scope : (scope.profile ?? null)

/** Plugins turned on for this agent, counted over the rows its Plugins page lists (bundled
 *  providers, platforms and the like are enabled by default and listed nowhere, so they never
 *  count). The plugin list is one owner-scoped store the Plugins page also loads; concurrent loads
 *  for the same owner share one request. */
export function usePluginsCount(scope: ProfileScope): CapabilityCount {
  const request = useSettingsGatewayRequest()
  const gatewayState = useStore($gatewayState)
  const rows = useStore($agentPlugins)
  const ownerKey = useStore($agentPluginsOwnerKey)
  const loadedProfile = useStore($agentPluginsProfile)
  const status = useStore($agentPluginsStatus)
  const profile = profileName(scope)
  const owned = status === 'ready' && ownerKey === request.ownerKey && loadedProfile === profile
  // The load this count last asked for that failed; cleared by a retry or an owner change.
  const attempt = JSON.stringify([request.ownerKey ?? '', profile])
  const [failedAttempt, setFailedAttempt] = useState<null | string>(null)
  const failed = !owned && failedAttempt === attempt

  const load = useCallback(() => {
    setFailedAttempt(null)
    void loadAgentPlugins(request, profile).then(loaded => {
      if (loaded === false) {
        setFailedAttempt(attempt)
      }
    })
  }, [attempt, profile, request])

  useEffect(() => {
    if (gatewayState === 'open' && !owned && failedAttempt !== attempt) {
      load()
    }
    // A failure is settled until the person retries; it is not re-asked on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt, gatewayState, owned])

  return {
    value: owned ? rows.filter(row => isDesktopRelevantPlugin(row) && row.status === 'enabled').length : undefined,
    failed,
    retry: load
  }
}
