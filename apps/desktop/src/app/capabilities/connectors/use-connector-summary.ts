import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { getMcpCatalog, type McpCatalogResponse, type ProfileScope, profileScopeKey } from '@/hermes'
import { getServers } from '@/lib/mcp-servers'

import { useHermesConfigRecord } from '../../hooks/use-config-record'
import { MCP_CATALOG_KEY } from '../mcp/mcp-status'
import { deriveCards } from './derive'
import { joinBundledEntries, joinLocalServers } from './data/join'
import { useHostedConnectors, usePluginServers } from './data/queries'

export interface ConnectorSummary {
  active: number
  loading: boolean
  total: number
}

/** Lightweight connector counts for the Capabilities submenu.
 *
 * This deliberately avoids runtime probes: the badge answers configuration
 * state (usable/turned on for this profile), not health. The full Connectors
 * tab still owns probing and status details when opened.
 */
export function useConnectorSummary(profile: ProfileScope): ConnectorSummary {
  const hosted = useHostedConnectors(profile)
  const config = useHermesConfigRecord(profile)
  const servers = useMemo(() => getServers(config.data ?? null), [config.data])
  const scopeKey = profileScopeKey(profile)

  const catalog = useQuery<McpCatalogResponse>({
    queryKey: [...MCP_CATALOG_KEY, scopeKey],
    queryFn: () => getMcpCatalog(profile ?? undefined),
    staleTime: 5 * 60_000
  })
  const catalogEntries = catalog.data?.entries ?? []
  const availableCatalog = useMemo(
    () => catalogEntries.filter(entry => !entry.installed && !(entry.name in servers)),
    [catalogEntries, servers]
  )
  const local = useMemo(
    () => joinLocalServers({ catalog: catalogEntries, servers, status: {} }),
    [catalogEntries, servers]
  )
  const pluginServers = usePluginServers(profile)

  const cards = useMemo(
    () =>
      deriveCards({
        bundled: joinBundledEntries(availableCatalog),
        hosted: hosted.rows,
        local: [...local, ...pluginServers],
        titles: hosted.titles
      }),
    [availableCatalog, hosted.rows, hosted.titles, local, pluginServers]
  )

  const active = cards.filter(card => {
    const hostedOn = card.ways.hosted?.state === 'connected'
    const localOn = card.ways.local?.installed === true && card.ways.local.serverEnabled === true

    return hostedOn || localOn
  }).length

  return {
    active,
    loading: config.isLoading || catalog.isPending || hosted.phase === 'loading',
    total: cards.length
  }
}
