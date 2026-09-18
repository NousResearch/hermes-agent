// The page's reads, as TanStack queries.
//
// Two hooks, and both answer in the shapes the components already take: the
// directory gets `rows` + `titles` for `deriveCards`, the dialog gets
// `ToolInput[]` plus the `status` `useToolsEditor` expects. The container feeds
// them straight through — no reshaping above this file.

import type { ConnectorAccountRow } from '@hermes/shared'
import { useMutation, useQuery } from '@tanstack/react-query'
import { useCallback, useMemo } from 'react'

import type { ProfileScope } from '@/hermes'
import { translateNow } from '@/i18n'
import { isMissingRpcMethod, isOutOfSyncRpcParams } from '@/lib/gateway-rpc'
import { queryClient } from '@/lib/query-client'
import { notifyError } from '@/store/notifications'

import type { HostedConnectorInput, ToolInput, ToolsFreshness } from '../types'
import type { ToolsEditorStatus } from '../use-tools-editor'

import {
  connectorCategories,
  type ConnectorPolicyView,
  connectorTitles,
  EMPTY_POLICY,
  joinHostedConnectors,
  readPolicy,
  toolsFreshness
} from './join'
import {
  connectorsAccountsQueryKey,
  connectorsCatalogQueryKey,
  connectorsListQueryKey,
  connectorsPolicyQueryKey,
  connectorsScopeQueryKey,
  connectorToolsQueryKey
} from './keys'
import {
  asConnectorError,
  connectorAccounts,
  connectorCatalog,
  type ConnectorErrorReason,
  connectorPolicy,
  type ConnectorRpcError,
  connectorTools,
  listConnectors
} from './rpc'

// A typed connector failure is a verdict, not a blip: retrying "signed out" or
// "connectors are unavailable" only delays the honest state. Recovery is the
// page's own Retry.
const READ_OPTIONS = { refetchOnWindowFocus: false, retry: false } as const

/** The shelf changes on the portal's clock, not the person's. */
const CATALOG_STALE_MS = 30 * 60_000

/** The backend caches a tool list for 24 h. Holding it for half that keeps the
 *  dialog instant on reopen while leaving the cue honest, and Refresh always
 *  bypasses both caches. */
const TOOLS_STALE_MS = 12 * 60 * 60_000

/** What the hosted half of the page is doing. `unavailable` is not a failure:
 *  this account has no connectors, so the hosted groups are simply absent. */
export type HostedPhase = 'failed' | 'loading' | 'ready' | 'signedOut' | 'unavailable'

export interface HostedConnectorsView {
  accounts: readonly ConnectorAccountRow[]
  /** Hosted slug → shelf category; `joinLocalServers({ categories })` takes it so
   *  a local server filters into the same bucket as the app it backs. */
  categories: Record<string, string>
  /** The typed failure behind `phase: 'failed'`. */
  error: ConnectorRpcError | null
  phase: HostedPhase
  /** Handed to the tools hooks and to the dialog's org note. */
  policy: ConnectorPolicyView
  /** Retry, and the page's refresh hotkey. */
  refetch: () => void
  /** Straight into `deriveCards({ hosted })`. */
  rows: HostedConnectorInput[]
  /** Straight into `deriveCards({ titles })`. */
  titles: Record<string, string>
}

/**
 * The three account-level reads that make the hosted cards, plus the policy
 * that decides what is on. They are fetched in parallel because none of them
 * needs another's answer, and one failing must not hide the rest — the phase is
 * derived from all four together.
 */
export function useHostedConnectors(scope: ProfileScope): HostedConnectorsView {
  const list = useQuery({
    ...READ_OPTIONS,
    queryFn: () => listConnectors(scope).catch(reportVersionSkew),
    queryKey: connectorsListQueryKey(scope),
    staleTime: 0
  })

  const catalog = useQuery({
    ...READ_OPTIONS,
    queryFn: () => connectorCatalog(scope),
    queryKey: connectorsCatalogQueryKey(scope),
    staleTime: CATALOG_STALE_MS
  })

  const accounts = useQuery({
    ...READ_OPTIONS,
    queryFn: () => connectorAccounts(scope),
    queryKey: connectorsAccountsQueryKey(scope),
    staleTime: 0
  })

  const policy = useQuery({
    ...READ_OPTIONS,
    queryFn: () => connectorPolicy(scope),
    queryKey: connectorsPolicyQueryKey(scope),
    staleTime: 0
  })

  const errors = [list.error, catalog.error, accounts.error, policy.error]
    .filter((error): error is Error => error !== null)
    .map(asConnectorError)

  const pending = list.isPending || catalog.isPending || accounts.isPending || policy.isPending

  const policyView = useMemo(() => (policy.data ? readPolicy(policy.data.layers) : EMPTY_POLICY), [policy.data])

  const joined = useMemo(() => {
    const input = {
      accounts: accounts.data?.accounts ?? [],
      catalog: catalog.data?.connectors ?? [],
      list: list.data?.connectors ?? [],
      policy: policyView
    }

    return {
      categories: connectorCategories(input.catalog),
      rows: joinHostedConnectors(input),
      titles: connectorTitles(input)
    }
  }, [accounts.data, catalog.data, list.data, policyView])

  const refetch = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: connectorsScopeQueryKey(scope) })
  }, [scope])

  return {
    accounts: accounts.data?.accounts ?? [],
    categories: joined.categories,
    error: errors[0] ?? null,
    phase: hostedPhase(pending, errors, list.data?.available),
    policy: policyView,
    refetch,
    rows: joined.rows,
    titles: joined.titles
  }
}

/**
 * An older backend answers this page's reads with "unknown method" (-32601) or
 * refuses the renamed params shape ("out of sync (different versions)"). Neither
 * carries a `ConnectorErrorReason`, so both land on `phase: 'failed'` — a Retry
 * that can never win. Say it once, through the notice that already carries the
 * route to an update, exactly as the chat catalog store does for the same two
 * answers. The error is rethrown, so the page still shows its failed state.
 *
 * Only the list read reports: all four reads fail together against a backend
 * that predates them, and one notice is enough.
 */
function reportVersionSkew(error: unknown): never {
  if (isMissingRpcMethod(error) || isOutOfSyncRpcParams(error instanceof Error ? error : String(error))) {
    notifyError(error, translateNow('connectorsPage.page.hostedFailedTitle'))
  }

  throw error
}

/** One table, read top to bottom. "Sign in" outranks "unavailable" because a
 *  signed-out account has no verdict about connectors yet, and both outrank the
 *  generic failure. */
function hostedPhase(pending: boolean, errors: readonly ConnectorRpcError[], available?: boolean): HostedPhase {
  const saw = (reason: ConnectorErrorReason) => errors.some(error => error.reason === reason)

  if (saw('NEEDS_NOUS_AUTH')) {
    return 'signedOut'
  }

  if (saw('CONNECTORS_UNAVAILABLE')) {
    return 'unavailable'
  }

  if (errors.length > 0) {
    return 'failed'
  }

  if (pending) {
    return 'loading'
  }

  return available === false ? 'unavailable' : 'ready'
}

/** Why a tool list is not on screen. `unavailable` is the catch-all, so a reason
 *  the backend adds later lands on the state with a Retry rather than on a
 *  state that claims something specific. */
const TOOLS_STATUS: Partial<Record<ConnectorErrorReason, ToolsEditorStatus>> = {
  CONNECTOR_NOT_FOUND: 'gone',
  NEEDS_NOUS_AUTH: 'signedOut'
}

export interface ConnectorToolsView {
  /** The 24 h cache's own age, in milliseconds, for the freshness cue. */
  freshness: ToolsFreshness | null
  /** Refresh: revalidate upstream and write the answer over the cached list. */
  refresh: () => void
  refreshing: boolean
  /** Exactly what `useToolsEditor({ status })` takes; `null` means the list is
   *  on screen and the editor owns the phase. */
  status: ToolsEditorStatus | null
  tools: ToolInput[]
}

/** One connector's tool list. `slug` is null while no dialog is open, which
 *  parks the query instead of fetching a list nobody asked for. */
export function useConnectorTools(scope: ProfileScope, slug: null | string): ConnectorToolsView {
  const key = connectorToolsQueryKey(scope, slug ?? '')

  const tools = useQuery({
    ...READ_OPTIONS,
    enabled: slug !== null,
    queryFn: () => connectorTools(scope, slug ?? ''),
    queryKey: key,
    staleTime: TOOLS_STALE_MS
  })

  const revalidate = useMutation({
    mutationFn: () => connectorTools(scope, slug ?? '', true),
    // Write through rather than invalidate: the refreshed answer IS the new
    // cache entry, and a second round trip would make Refresh cost two.
    onSuccess: data => queryClient.setQueryData(key, data)
  })

  const error = tools.error === null ? null : asConnectorError(tools.error)

  return {
    freshness: tools.data ? toolsFreshness(tools.data) : null,
    refresh: () => revalidate.mutate(),
    refreshing: revalidate.isPending,
    status: toolsStatus(slug, tools.isPending, error),
    tools: tools.data?.tools ?? []
  }
}

function toolsStatus(slug: null | string, pending: boolean, error: ConnectorRpcError | null): ToolsEditorStatus | null {
  if (error) {
    return (error.reason && TOOLS_STATUS[error.reason]) ?? 'unavailable'
  }

  // A parked query is `pending` forever; with no connector open there is nothing
  // to report, and the dialog that opens one will ask again.
  return slug !== null && pending ? 'loading' : null
}
