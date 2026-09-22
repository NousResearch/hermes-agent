import type { ConnectorAccountRow, ConnectorToolsResult } from '@hermes/shared'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'
import { isMissingRpcMethod, isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { queryClient } from '@/lib/query-client'
import { notifyError } from '@/store/notifications'

import { hostedPhase } from '../derive'
import { toolReadStatus } from '../derive-tools'
import type { HostedConnectorInput, HostedPhase, LocalServerInput, ToolInput, ToolsEditorStatus } from '../types'

import {
  type ConnectorPolicyView,
  connectorTitles,
  EMPTY_POLICY,
  joinHostedConnectors,
  pluginServerRows,
  readPolicy
} from './join'
import {
  CONNECTOR_GC_TIME,
  CONNECTOR_LIFETIMES,
  connectorsAccountsQueryKey,
  connectorsCatalogQueryKey,
  connectorsListQueryKey,
  connectorsPolicyQueryKey,
  connectorToolsQueryKey,
  invalidateConnectorApp,
  invalidateConnectors,
  PLUGIN_SERVERS_STALE_TIME,
  pluginServersQueryKey
} from './keys'
import { clearPersisted, seedOptions } from './persist'
import {
  asConnectorError,
  connectorAccounts,
  connectorCatalog,
  connectorPolicy,
  connectorTools,
  listConnectors,
  listMcpServers,
  mcpServerStatus
} from './rpc'

const read = (of: keyof typeof CONNECTOR_LIFETIMES) => ({
  gcTime: CONNECTOR_GC_TIME,
  refetchOnWindowFocus: CONNECTOR_LIFETIMES[of].focusRefetch,
  retry: false,
  staleTime: CONNECTOR_LIFETIMES[of].staleTime
})

export interface HostedConnectorsView {
  accounts: readonly ConnectorAccountRow[]
  listSlugs: ReadonlySet<string>
  phase: HostedPhase
  policy: ConnectorPolicyView
  refetch: () => void
  retryRules: () => void
  rows: HostedConnectorInput[]
  rulesFailed: boolean
  rulesSignedOut: boolean
  titles: Record<string, string>
}

export function useHostedConnectors(scope: ProfileScope): HostedConnectorsView {
  const listSeed = useMemo(() => seedOptions<Awaited<ReturnType<typeof listConnectors>>>(scope, 'list'), [scope])

  const catalogSeed = useMemo(
    () => seedOptions<Awaited<ReturnType<typeof connectorCatalog>>>(scope, 'catalog'),
    [scope]
  )

  const list = useQuery({
    ...read('list'),
    ...listSeed,
    queryFn: () => listConnectors(scope).catch(reportVersionSkew),
    queryKey: connectorsListQueryKey(scope)
  })

  const catalog = useQuery({
    ...read('catalog'),
    ...catalogSeed,
    queryFn: () => connectorCatalog(scope),
    queryKey: connectorsCatalogQueryKey(scope)
  })

  const accounts = useQuery({
    ...read('accounts'),
    queryFn: () => connectorAccounts(scope),
    queryKey: connectorsAccountsQueryKey(scope)
  })

  const policy = useQuery({
    ...read('policy'),
    queryFn: () => connectorPolicy(scope),
    queryKey: connectorsPolicyQueryKey(scope)
  })

  const listError = list.error === null ? null : asConnectorError(list.error)
  const policyError = policy.error === null ? null : asConnectorError(policy.error)
  const rulesFailed = policyError !== null

  const policyView = useMemo(() => (policy.data ? readPolicy(policy.data) : EMPTY_POLICY), [policy.data])

  const joined = useMemo(() => {
    const input = {
      accounts: accounts.data?.accounts ?? [],
      catalog: catalog.data?.connectors ?? [],
      list: list.data?.connectors ?? [],
      policy: policyView,
      rulesReadable: !rulesFailed
    }

    return { rows: joinHostedConnectors(input), titles: connectorTitles(input) }
  }, [accounts.data, catalog.data, list.data, policyView, rulesFailed])

  const phase = hostedPhase({
    available: list.data?.available,
    errored: listError !== null,
    pending: list.isPending,
    reason: listError?.reason ?? null
  })

  const blanked = phase === 'signedOut' || phase === 'unavailable'

  useEffect(() => {
    if (phase === 'signedOut') {
      clearPersisted(scope)
    }
  }, [phase, scope])

  const listSlugs = useMemo(
    () => new Set(blanked ? [] : (list.data?.connectors ?? []).flatMap(row => (row.connector ? [row.connector] : []))),
    [blanked, list.data]
  )

  const refetch = useCallback(() => invalidateConnectors(scope, 'accounts', 'catalog', 'list', 'policy'), [scope])

  const retryRules = useCallback(() => invalidateConnectors(scope, 'policy'), [scope])

  return {
    accounts: accounts.data?.accounts ?? [],
    listSlugs,
    phase,
    policy: policyView,
    refetch,
    retryRules,
    rows: blanked ? [] : joined.rows,
    rulesFailed,
    rulesSignedOut: policyError?.reason === 'NEEDS_NOUS_AUTH',
    titles: joined.titles
  }
}

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

function reportVersionSkew(cause: unknown): never {
  if (isMissingRpcMethod(cause) || isOutOfSyncRpcParams(cause instanceof Error ? cause : String(cause))) {
    notifyError(cause, translateNow('connectorsPage.page.hostedFailedTitle'))
  }

  throw cause
}

export function connectorToolsQueryOptions(scope: ProfileScope, slug: string) {
  return {
    ...read('tools'),
    ...seedOptions<ConnectorToolsResult>(scope, 'tools', slug),
    queryFn: () => connectorTools(scope, slug),
    queryKey: connectorToolsQueryKey(scope, slug)
  }
}

export interface ConnectorToolsView {
  refresh: () => void
  retry: () => void
  signedOut: boolean
  status: ToolsEditorStatus | null
  tools: ToolInput[]
}

export function useConnectorTools(scope: ProfileScope, slug: null | string, listHasApp: boolean): ConnectorToolsView {
  const key = connectorToolsQueryKey(scope, slug ?? '')
  const options = useMemo(() => connectorToolsQueryOptions(scope, slug ?? ''), [scope, slug])

  const tools = useQuery({ ...options, enabled: slug !== null })

  const revalidate = useMutation({
    mutationFn: () => connectorTools(scope, slug ?? '', true),
    onError: error => notifyError(error, translateNow('connectorsPage.page.refreshFailed')),
    onSuccess: data => queryClient.setQueryData(key, data)
  })

  const error = tools.error === null ? null : asConnectorError(tools.error)
  const reason = error === null ? null : (error.reason ?? 'CONNECTOR_REQUEST_FAILED')

  return {
    refresh: () => revalidate.mutate(),
    retry: () => invalidateConnectorApp(scope, slug ?? ''),
    signedOut: reason === 'NEEDS_NOUS_AUTH',
    status:
      slug === null
        ? null
        : toolReadStatus({ hasData: tools.data !== undefined, listHasApp, pending: tools.isPending, reason }),
    tools: tools.data?.tools ?? []
  }
}
