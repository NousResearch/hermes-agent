import type {
  ComposioAccountRow,
  ComposioAccountStatus,
  ComposioCatalogResult,
  ComposioConnectResult,
  ComposioProviderRow,
  ComposioProviderState,
  ComposioProviderStatus,
  ComposioRemoveResult,
  ComposioStatusResult,
  ComposioToolRow,
  ComposioToolsResult,
  ComposioToolsStatus
} from '@hermes/shared/gateway-events'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { hermesConfigKey } from '@/app/hooks/use-config-record'
import { SearchField } from '@/components/ui/search-field'
import type { ProfileScope } from '@/hermes'
import { profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { requestGatewayForAgent } from '@/store/gateway'

import { ConnectorDialog } from './connector-dialog'
import { ConnectorRowCard } from './connector-row-card'
import { ToolRow } from './tool-row'
import type { ConnectorCardModel, ConnectorState, ConnectorStateWord, ToolRowModel } from './types'

const COMPOSIO_MCP_NAME = 'composio'
const COMPOSIO_MCP_URL = 'https://connect.composio.dev/mcp'
const PROVIDER_QUERY_ROOT = ['composio-providers'] as const
const REFRESH_MS = 10_000
const RECONNECT_ACCOUNT_STATES = new Set<ComposioAccountStatus>(['expired', 'revoked', 'failed', 'inactive'])

type ProviderDefinition = ComposioProviderRow
type ProviderAccount = ComposioAccountRow
type ProviderStatus = ComposioProviderState
type ProviderState = ComposioProviderStatus
type AccountState = ComposioAccountStatus
type ProviderCatalogResult = ComposioCatalogResult
type ProviderStatusResult = ComposioStatusResult
type ProviderConnectResult = ComposioConnectResult
type ProviderRemoveResult = ComposioRemoveResult

type ProviderTool = ComposioToolRow

type ProviderToolsResult = ComposioToolsResult

interface McpServerAddResult {
  ok: boolean
}

export function useComposioProviderCatalog(profile: ProfileScope) {
  return useQuery({
    queryKey: catalogKey(profile),
    queryFn: () => rpc<ProviderCatalogResult>(profile, 'composio.catalog'),
    staleTime: 5 * 60_000
  })
}

interface ProviderRowsProps {
  actionsDisabled?: boolean
  catalog: ReturnType<typeof useComposioProviderCatalog>
  excludedToolkits?: readonly string[]
  filterQuery: string
  mcpServers: Record<string, Record<string, unknown>>
  profile: ProfileScope
  refreshMcpConfig: () => void
}

function rpc<T>(profile: ProfileScope, method: string, params: Record<string, unknown> = {}): Promise<T> {
  const scope = capabilityScoped(profile)

  return requestGatewayForAgent<T>(
    scope.connectionId ?? null,
    scope.profile ?? 'default',
    method,
    { ...params, profile: scope.profile ?? 'default' },
    60_000,
    undefined,
    { spawnPriority: 'foreground' }
  )
}

function rpcWithoutProfile<T>(profile: ProfileScope, method: string, params: Record<string, unknown> = {}): Promise<T> {
  const scope = capabilityScoped(profile)

  return requestGatewayForAgent<T>(
    scope.connectionId ?? null,
    scope.profile ?? 'default',
    method,
    params,
    60_000,
    undefined,
    { spawnPriority: 'foreground' }
  )
}

const catalogKey = (profile: ProfileScope) => [...PROVIDER_QUERY_ROOT, profileScopeKey(profile), 'catalog'] as const
const statusKey = (profile: ProfileScope) => [...PROVIDER_QUERY_ROOT, profileScopeKey(profile), 'status'] as const

const toolsKey = (profile: ProfileScope, provider: string, accountId: string | null) =>
  [...PROVIDER_QUERY_ROOT, profileScopeKey(profile), 'tools', provider, accountId] as const

function providerCard(provider: ProviderDefinition, status?: ProviderStatus): ConnectorCardModel {
  const phase = status?.status ?? 'disconnected'

  const state: ConnectorState =
    phase === 'connected'
      ? 'connected'
      : phase === 'pending'
        ? 'connecting'
        : phase === 'expired' || phase === 'revoked' || phase === 'inactive'
          ? 'expired'
          : phase === 'failed' || phase === 'requires_mcp_auth' || phase === 'unavailable'
            ? 'broken'
            : 'available'

  const words: Record<ConnectorState, ConnectorStateWord> = {
    available: 'available',
    broken: 'couldNotConnect',
    connected: 'connected',
    connecting: 'connecting',
    expired: 'accessExpired',
    off: 'offForYou',
    unknown: 'connectionUnknown'
  }

  const account = status?.accounts?.find(row => row.status === 'active') ?? status?.accounts?.[0]

  return {
    accountSource: 'composio',
    description: providerDescription(provider),
    inCatalog: true,
    name: provider.label,
    residency: 'hosted',
    slug: provider.toolkit,
    state,
    stateWord: words[state],
    verb: state === 'available' ? 'connect' : state === 'expired' || state === 'broken' ? 'reconnect' : undefined,
    ways: {
      hosted: {
        accountLabel: account ? accountLabel(provider, account, (status?.accounts ?? []).indexOf(account)) : undefined,
        accounts: [],
        connected: state === 'connected',
        retiredAccounts: [],
        state
      },
      local: null
    }
  }
}

function providerDescription(provider: ProviderDefinition): string {
  const known: Record<string, string> = {
    gmail: 'Search, read, and send email.',
    googlecalendar: 'Find availability and manage calendar events.',
    googledrive: 'Search and organize files in Drive.',
    slack: 'Find messages and work with channels.',
    github: 'Browse repositories, issues, and pull requests.',
    notion: 'Find and update pages and databases.',
    linear: 'Search and manage issues and projects.',
    jira: 'Find and update work items and projects.',
    outlook: 'Work with Outlook mail and calendar.',
    resend: 'Manage email delivery and contacts.',
    zoho_inventory: 'Manage products, orders, and inventory.'
  }

  return known[provider.toolkit] ?? `Connect ${provider.label} to browse its available tools.`
}

function accountLabel(provider: ProviderDefinition, account: ProviderAccount, index: number): string {
  return account.alias || account.display_name || `${provider.label} account ${index + 1}`
}

function providerToolRows(tools: readonly ProviderTool[]): ToolRowModel[] {
  return tools.map(tool => ({
    categories: [],
    deprecated: false,
    description: tool.description ?? '',
    facet: 'unclassified',
    hints: [],
    lockedBy: null,
    name: tool.tool_slug,
    on: true,
    slug: tool.tool_slug
  }))
}

function queryMatches(provider: ProviderDefinition, query: string): boolean {
  const needle = query.trim().toLocaleLowerCase()

  if (!needle) {
    return true
  }

  return [provider.label, provider.provider, provider.toolkit].some(value => value.toLocaleLowerCase().includes(needle))
}

function statusCopy(status: ProviderState | AccountState | undefined, t: ReturnType<typeof useI18n>['t']): string {
  switch (status) {
    case 'active':

    case 'connected':
      return t.connectors.connected

    case 'disconnected':
      return t.connectors.notConnected

    case 'expired':
      return t.connectorsPage.accounts.status.expired

    case 'failed':
      return t.connectorsPage.accounts.status.failed

    case 'inactive':
      return t.connectorsPage.accounts.status.inactive

    case 'pending':
      return t.connectorsPage.accounts.status.pending

    case 'requires_mcp_auth':
      return t.connectors.needsAuth

    case 'revoked':
      return t.connectorsPage.accounts.status.revoked

    default:
      return t.connectors.unavailable
  }
}

function readableExternalUrl(value: string): string {
  const url = new URL(value)

  if (
    url.protocol !== 'https:' ||
    url.hostname !== 'connect.composio.dev' ||
    (url.port !== '' && url.port !== '443') ||
    url.username !== '' ||
    url.password !== ''
  ) {
    throw new Error('The provider did not return a trusted sign-in URL.')
  }

  return url.toString()
}

function providerIsWaiting(state: ProviderStatus | undefined, watching: boolean, stopped: boolean): boolean {
  return watching || (!stopped && (state?.status === 'pending' || state?.accounts?.some(account => account.status === 'pending') === true))
}

export function composioProvidersMatch(providers: readonly ProviderDefinition[], query: string): boolean {
  return providers.some(provider => queryMatches(provider, query))
}

export function composioProvidersForDirectory(
  providers: readonly ProviderDefinition[],
  cards: readonly ConnectorCardModel[]
): ProviderDefinition[] {
  const nativeProviders = new Set(
    cards.filter(card => card.ways.hosted !== null).map(card => card.slug.toLocaleLowerCase())
  )

  return providers.filter(
    provider => !nativeProviders.has(provider.provider.toLocaleLowerCase()) && !nativeProviders.has(provider.toolkit.toLocaleLowerCase())
  )
}

export function ComposioProviderRows({
  actionsDisabled = false,
  catalog,
  excludedToolkits = [],
  filterQuery,
  mcpServers,
  profile,
  refreshMcpConfig
}: ProviderRowsProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const accountCopy = copy.accounts
  const queryClient = useQueryClient()
  const [busyProvider, setBusyProvider] = useState<string | null>(null)
  const [nativeAuthProvider, setNativeAuthProvider] = useState<string | null>(null)
  const [watching, setWatching] = useState<ReadonlySet<string>>(() => new Set())
  const [stopped, setStopped] = useState<ReadonlySet<string>>(() => new Set())
  const [errorProvider, setErrorProvider] = useState<string | null>(null)
  const [errorText, setErrorText] = useState<string | null>(null)
  const [removing, setRemoving] = useState<{ account: ProviderAccount; provider: ProviderDefinition } | null>(null)
  const [selectedProviderKey, setSelectedProviderKey] = useState<string | null>(null)
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null)
  const cancelledNativeAuth = useRef(false)
  const activeOperation = useRef<symbol | null>(null)
  const profileEpoch = useRef(0)
  const profileKey = profileScopeKey(profile)

  // This effect invalidates stale async work owned by the previous profile.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    profileEpoch.current += 1
    cancelledNativeAuth.current = true
    activeOperation.current = null
    setBusyProvider(null)
    setNativeAuthProvider(null)
    setWatching(new Set())
    setStopped(new Set())
    setErrorProvider(null)
    setErrorText(null)
    setRemoving(null)
    setSelectedProviderKey(null)
    setSelectedAccountId(null)

    return () => {
      profileEpoch.current += 1
      cancelledNativeAuth.current = true
    }
  }, [profileKey])

  const watchedOrPending = useCallback(
    (providers: readonly ProviderStatus[] | undefined) => {
      if (!providers) {
        return false
      }

      return providers.some(
        provider =>
          !stopped.has(provider.provider) &&
          (watching.has(provider.provider) ||
            provider.status === 'pending' ||
            (provider.accounts ?? []).some(account => account.status === 'pending'))
      )
    },
    [stopped, watching]
  )

  const status = useQuery({
    enabled: catalog.isSuccess,
    queryKey: statusKey(profile),
    queryFn: () => rpc<ProviderStatusResult>(profile, 'composio.status'),
    refetchInterval: query => (watchedOrPending(query.state.data?.providers) ? REFRESH_MS : false),
    refetchOnWindowFocus: true,
    staleTime: 10_000
  })

  const visibleProviders = useMemo(
    () => (catalog.data?.providers ?? []).filter(provider => !excludedToolkits.includes(provider.toolkit) && queryMatches(provider, filterQuery)),
    [catalog.data?.providers, excludedToolkits, filterQuery]
  )

  const statusByProvider = useMemo(
    () => new Map((status.data?.providers ?? []).map(provider => [provider.provider, provider])),
    [status.data?.providers]
  )

  const selectedProvider = visibleProviders.find(provider => provider.provider === selectedProviderKey)

  const selectedProviderStatus = selectedProvider
    ? statusByProvider.get(selectedProvider.provider) ?? {
        accounts: [],
        provider: selectedProvider.provider,
        status: 'unavailable' as const,
        toolkit: selectedProvider.toolkit
      }
    : undefined

  const activeAccounts = (selectedProviderStatus?.accounts ?? []).filter(account => account.status === 'active')
  const selectedAccount = activeAccounts.find(account => account.id === selectedAccountId) ?? activeAccounts[0]

  const providerTools = useQuery({
    enabled: selectedProvider !== undefined && selectedAccount !== undefined,
    queryKey: toolsKey(profile, selectedProvider?.provider ?? '', selectedAccount?.id ?? null),
    queryFn: () => rpc<ProviderToolsResult>(profile, 'composio.tools', {
      account_id: selectedAccount!.id,
      provider: selectedProvider!.provider
    })
  })

  const refreshStatus = async (provider?: string) => {
    if (provider) {
      setStopped(previous => new Set([...previous].filter(item => item !== provider)))
      setWatching(previous => new Set([...previous, provider]))
    } else {
      setStopped(new Set())
    }

    await status.refetch()
  }

  const ensureComposioServer = async (epoch: number) => {
    const existing = mcpServers[COMPOSIO_MCP_NAME]

    if (
      existing &&
      (existing.url !== COMPOSIO_MCP_URL || existing.auth !== 'oauth' || existing.trust !== 'untrusted')
    ) {
      throw new Error('Review the existing Composio server configuration before using provider accounts.')
    }

    if (!existing) {
      // Add only after a provider-button click. Trust is explicit so Composio's
      // metadata calls stay read-only until this account authorizes a provider.
      const scope = capabilityScoped(profile)

      const added = await rpc<McpServerAddResult>(profile, 'mcp.servers.add', {
        config: { trust: 'untrusted' },
        name: COMPOSIO_MCP_NAME,
        preset: COMPOSIO_MCP_NAME,
        profile: scope.profile ?? 'default'
      })

      if (!added.ok) {
        throw new Error('Composio setup could not be saved.')
      }

      await queryClient.invalidateQueries({ queryKey: hermesConfigKey(profile) })
      refreshMcpConfig()
    }

    return epoch === profileEpoch.current
  }

  const reloadComposioMcp = async (epoch: number) => {
    const reload = await rpcWithoutProfile<{ status: string }>(profile, 'reload.mcp', { confirm: true })

    if (reload.status !== 'reloaded') {
      throw new Error('Composio MCP server did not reload.')
    }

    return epoch === profileEpoch.current
  }

  const authorizeComposioMcp = async (epoch: number) => {
    cancelledNativeAuth.current = false
    await completeMcpDesktopOAuth({
      cancelled: () => cancelledNativeAuth.current,
      profile,
      serverName: COMPOSIO_MCP_NAME
    })

    if (epoch !== profileEpoch.current) {
      return false
    }

    return reloadComposioMcp(epoch)
  }

  const startProviderConnect = async (provider: ProviderDefinition) => {
    if (activeOperation.current !== null) {
      return
    }

    const operation = Symbol(provider.provider)
    const epoch = profileEpoch.current
    activeOperation.current = operation
    setBusyProvider(provider.provider)
    setErrorProvider(null)
    setErrorText(null)

    try {
      if (!(await ensureComposioServer(epoch)) || epoch !== profileEpoch.current) {
        return
      }

      // Connecting the configured MCP server is a user-authorized step. Status
      // reads stay passive, while an explicit provider click loads newly added
      // and already configured servers before any provider RPC reaches them.
      if (!(await reloadComposioMcp(epoch)) || epoch !== profileEpoch.current) {
        return
      }

      const start = () => rpc<ProviderConnectResult>(profile, 'composio.connect', { provider: provider.provider })
      let result = await start()

      if (epoch !== profileEpoch.current) {
        return
      }

      if (result.status === 'requires_mcp_auth') {
        setNativeAuthProvider(provider.provider)

        try {
          if (!(await authorizeComposioMcp(epoch))) {
            return
          }
        } finally {
          if (epoch === profileEpoch.current) {
            setNativeAuthProvider(null)
          }
        }

        result = await start()

        if (epoch !== profileEpoch.current) {
          return
        }

        if (result.status === 'requires_mcp_auth') {
          throw new Error('Composio still needs sign-in. Refresh and try again.')
        }
      }

      if (result.auth_url) {
        await window.hermesDesktop.openExternal(readableExternalUrl(result.auth_url))
      }

      if (result.status === 'pending') {
        setStopped(previous => new Set([...previous].filter(item => item !== provider.provider)))
        setWatching(previous => new Set([...previous, provider.provider]))
      }

      await queryClient.invalidateQueries({ queryKey: statusKey(profile) })
      await status.refetch()
    } catch (error) {
      if (epoch === profileEpoch.current && !(error instanceof McpOAuthCancelled)) {
        setErrorProvider(provider.provider)
        setErrorText(t.connectors.connectErrorFor(provider.label))
      }
    } finally {
      if (activeOperation.current === operation) {
        activeOperation.current = null
        setBusyProvider(null)
        setNativeAuthProvider(null)
      }
    }
  }

  const stopWaiting = (provider: string) => {
    setStopped(previous => new Set([...previous, provider]))
    setWatching(previous => new Set([...previous].filter(item => item !== provider)))
  }

  const cancelNativeAuthorization = () => {
    cancelledNativeAuth.current = true
    setNativeAuthProvider(null)
  }

  const removeAccount = async () => {
    if (!removing) {
      return
    }

    if (activeOperation.current !== null) {
      return
    }

    const { account, provider } = removing
    const operation = Symbol(provider.provider)
    const epoch = profileEpoch.current
    activeOperation.current = operation

    setBusyProvider(provider.provider)
    setErrorProvider(null)
    setErrorText(null)

    try {
      const result = await rpc<ProviderRemoveResult>(profile, 'composio.remove', {
        account_id: account.id,
        provider: provider.provider
      })

      if (epoch !== profileEpoch.current) {
        return
      }

      if (result.status !== 'removed') {
        setErrorProvider(provider.provider)
        setErrorText(result.status === 'requires_mcp_auth' ? t.connectors.needsAuth : t.connectors.connectErrorFor(provider.label))

        return
      }

      setRemoving(null)
      await queryClient.invalidateQueries({ queryKey: statusKey(profile) })
      await status.refetch()
    } catch {
      if (epoch === profileEpoch.current) {
        setErrorProvider(provider.provider)
        setErrorText(t.connectors.connectErrorFor(provider.label))
      }
    } finally {
      if (activeOperation.current === operation) {
        activeOperation.current = null
        setBusyProvider(null)
      }
    }
  }

  if (catalog.isPending) {
    return (
      <section aria-label={copy.group.available} className="grid gap-2" data-testid="composio-provider-rows">
        <h3 className="text-xs font-semibold text-(--ui-text-primary)">{copy.group.available}</h3>
        <p className="text-xs text-(--ui-text-tertiary)" role="status">{copy.page.loading}</p>
      </section>
    )
  }

  if (catalog.isError) {
    return (
      <section aria-label={copy.group.available} className="grid gap-2" data-testid="composio-provider-rows">
        <h3 className="text-xs font-semibold text-(--ui-text-primary)">{copy.group.available}</h3>
        <p className="text-xs text-(--ui-text-tertiary)" role="alert">
          {t.connectors.unavailable}
        </p>
        <button className="w-fit text-xs text-(--ui-text-secondary) underline" onClick={() => void catalog.refetch()} type="button">
          {t.connectors.retry}
        </button>
      </section>
    )
  }

  return (
    <section aria-label={copy.group.available} className="grid gap-2" data-testid="composio-provider-rows">
      <header className="flex items-center gap-2">
        <h3 className="flex-1 text-xs font-semibold text-(--ui-text-primary)">{copy.group.available}</h3>
        <button className="text-xs text-(--ui-text-secondary) underline" onClick={() => void refreshStatus()} type="button">
          {t.connectors.refresh}
        </button>
      </header>

      {visibleProviders.length === 0 ? (
        filterQuery.trim() ? null : <p className="text-xs text-(--ui-text-tertiary)">{t.connectors.unavailable}</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {visibleProviders.map(provider => {
            const state = statusByProvider.get(provider.provider)
            const accounts = state?.accounts ?? []
            const isBusy = busyProvider === provider.provider
            const card = providerCard(provider, state)

            return (
              <div data-provider={provider.provider} key={provider.provider}>
                <ConnectorRowCard
                  busy={isBusy}
                  card={card}
                  onOpen={() => {
                    setSelectedProviderKey(provider.provider)
                    setSelectedAccountId(null)
                  }}
                  onVerb={() => void startProviderConnect(provider)}
                />
              </div>
            )
          })}
        </div>
      )}

      {selectedProvider ? (
        <ConnectorDialog
          accounts={
            <>
              <ProviderAccountRows
                accounts={selectedProviderStatus?.accounts ?? []}
                actionsDisabled={actionsDisabled}
                busy={busyProvider === selectedProvider.provider}
                onRemove={account => setRemoving({ account, provider: selectedProvider })}
                onSelect={setSelectedAccountId}
                onStart={() => void startProviderConnect(selectedProvider)}
                provider={selectedProvider}
                selectedAccountId={selectedAccount?.id ?? null}
              />
              {removing?.provider.provider === selectedProvider.provider ? (
                <RemoveProviderAccount
                  account={removing.account}
                  accountIndex={(selectedProviderStatus?.accounts ?? []).indexOf(removing.account)}
                  busy={busyProvider === selectedProvider.provider}
                  onCancel={() => setRemoving(null)}
                  onRemove={() => void removeAccount()}
                  provider={selectedProvider}
                />
              ) : null}
            </>
          }
          card={providerCard(selectedProvider, selectedProviderStatus)}
          connectElement={
            <ProviderActions
              actionsDisabled={actionsDisabled}
              busy={busyProvider === selectedProvider.provider}
              error={errorProvider === selectedProvider.provider ? errorText : null}
              nativeAuth={nativeAuthProvider === selectedProvider.provider}
              onAdd={() => void startProviderConnect(selectedProvider)}
              onCancel={cancelNativeAuthorization}
              onRefresh={() => void refreshStatus(selectedProvider.provider)}
              onStop={() => stopWaiting(selectedProvider.provider)}
              provider={selectedProvider}
              status={selectedProviderStatus!}
              stopped={stopped.has(selectedProvider.provider)}
              watching={watching.has(selectedProvider.provider)}
            />
          }
          onOpenChange={open => { if (!open) { setSelectedProviderKey(null); setRemoving(null) } }}
          open
          tools={
            <ProviderTools
              account={selectedAccount}
              loading={providerTools.isPending}
              queryError={providerTools.isError}
              selectedProvider={selectedProvider}
              status={providerTools.data?.status}
              tools={providerTools.data?.tools ?? []}
            />
          }
        />
      ) : null}

    </section>
  )
}

function ProviderActions({
  actionsDisabled,
  busy,
  error,
  nativeAuth,
  onAdd,
  onCancel,
  onRefresh,
  onStop,
  provider,
  status,
  stopped,
  watching
}: {
  actionsDisabled: boolean
  busy: boolean
  error: string | null
  nativeAuth: boolean
  onAdd: () => void
  onCancel: () => void
  onRefresh: () => void
  onStop: () => void
  provider: ProviderDefinition
  status: ProviderStatus
  stopped: boolean
  watching: boolean
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.accounts
  const pending = providerIsWaiting(status, watching, stopped)

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <button className="rounded-md bg-(--ui-fill-secondary) px-2.5 py-1.5 text-xs text-(--ui-text-primary) disabled:opacity-50" disabled={actionsDisabled || busy || nativeAuth} onClick={onAdd} type="button">
          {busy ? t.connectorsPage.card.state.connecting : (status.accounts?.length ?? 0) > 0 ? copy.addAnother : copy.add}
        </button>
        {nativeAuth ? <button className="text-xs text-(--ui-text-secondary) underline" onClick={onCancel} type="button">{t.connectors.cancel}</button> : null}
        {pending ? <button className="text-xs text-(--ui-text-secondary) underline" onClick={onStop} type="button">{t.connectorsPage.card.verb.stopWaiting}</button> : null}
        {stopped ? <button className="text-xs text-(--ui-text-secondary) underline" onClick={onRefresh} type="button">{t.connectors.refresh}</button> : null}
      </div>
      {status.accounts?.some(account => RECONNECT_ACCOUNT_STATES.has(account.status)) ? (
        <p className="text-[0.7rem] text-(--ui-text-tertiary)">{copy.reconnectNote}</p>
      ) : null}
      {error ? <p className="text-xs text-(--ui-text-danger)" role="alert">{error || t.connectors.connectErrorFor(provider.label)}</p> : null}
    </div>
  )
}

function ProviderAccountRows({
  accounts,
  actionsDisabled,
  busy,
  onRemove,
  onSelect,
  onStart,
  provider,
  selectedAccountId
}: {
  actionsDisabled: boolean
  accounts: readonly ProviderAccount[]
  busy: boolean
  onRemove: (account: ProviderAccount) => void
  onSelect: (accountId: string | null) => void
  onStart: () => void
  provider: ProviderDefinition
  selectedAccountId: string | null
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.accounts

  return (
    <section aria-label={`${provider.label} ${copy.heading}`} className="grid max-h-[34vh] shrink-0 gap-1 overflow-y-auto border-b border-(--ui-stroke-tertiary) px-3.5 py-2">
      {accounts.map((account, index) => (
        <div className="flex min-w-0 items-center gap-2 border-t border-(--ui-stroke-tertiary) py-2 first:border-t-0" key={account.id}>
          {account.status === 'active' ? (
            <button aria-pressed={selectedAccountId === account.id} className="min-w-0 flex-1 truncate text-left text-xs font-medium text-(--ui-text-primary)" onClick={() => onSelect(account.id)} type="button">
              {accountLabel(provider, account, index)}{account.is_default ? ' · Default' : ''}
            </button>
          ) : <span className="min-w-0 flex-1 truncate text-xs text-(--ui-text-secondary)">{accountLabel(provider, account, index)}</span>}
          <span className="shrink-0 text-[0.68rem] text-(--ui-text-tertiary)">{statusCopy(account.status, t)}</span>
          {RECONNECT_ACCOUNT_STATES.has(account.status) ? <button className="shrink-0 text-xs text-(--ui-text-secondary) underline" disabled={actionsDisabled || busy} onClick={onStart} type="button">{t.connectorsPage.card.verb.reconnect}</button> : null}
          {account.status !== 'pending' ? <button className="shrink-0 text-xs text-(--ui-text-secondary) underline" disabled={actionsDisabled || busy} onClick={() => onRemove(account)} type="button">{copy.remove}</button> : null}
        </div>
      ))}
      {accounts.length === 0 ? <p className="text-xs text-(--ui-text-tertiary)">{t.connectors.notConnected}</p> : null}
    </section>
  )
}

function ProviderTools({
  account,
  loading,
  queryError,
  selectedProvider,
  status,
  tools
}: {
  account?: ProviderAccount
  loading: boolean
  queryError: boolean
  selectedProvider: ProviderDefinition
  status?: ComposioToolsStatus
  tools: readonly ProviderTool[]
}) {
  const { t } = useI18n()
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<string | null>(null)
  const rows = useMemo(() => providerToolRows(tools), [tools])
  const filtered = rows.filter(row => `${row.name} ${row.description}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))

  return (
    <section aria-label={`${selectedProvider.label} tools`} className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3.5 py-2">
      <SearchField containerClassName="flex-1" onChange={setQuery} placeholder={t.connectorsPage.tools.searchCountPlaceholder(rows.length)} value={query} />
        <span className="text-[0.7rem] text-(--ui-text-tertiary)">{rows.length}</span>
      </div>
      {!account ? <p className="px-3.5 py-4 text-xs text-(--ui-text-tertiary)">{t.connectorsPage.accounts.add}</p> : null}
      {account && loading ? <p className="px-3.5 py-4 text-xs text-(--ui-text-tertiary)" role="status">{t.connectorsPage.tools.loading}</p> : null}
      {account && queryError ? <p className="px-3.5 py-4 text-xs text-(--ui-text-tertiary)" role="alert">{t.connectors.unavailable}</p> : null}
      {account && !loading && !queryError && status !== undefined && status !== 'ready' ? (
        <p className="px-3.5 py-4 text-xs text-(--ui-text-tertiary)" role="status">
          {status === 'requires_mcp_auth' ? t.connectors.needsAuth : t.connectors.unavailable}
        </p>
      ) : null}
      {account && !loading && !queryError ? (
        <div aria-label={t.connectorsPage.tools.toolList(selectedProvider.label)} className="min-h-0 flex-1 overflow-y-auto px-3.5 py-2" role="list">
          {status === 'ready' && filtered.length === 0 ? <p className="py-4 text-xs text-(--ui-text-tertiary)">{t.connectorsPage.tools.noMatch}</p> : null}
          {status === 'ready' ? filtered.map(tool => <ToolRow expanded={expanded === tool.slug} key={tool.slug} on={true} onExpand={() => setExpanded(expanded === tool.slug ? null : tool.slug)} onToggle={() => undefined} preview readOnly tool={tool} />) : null}
        </div>
      ) : null}
    </section>
  )
}

function RemoveProviderAccount({
  account,
  accountIndex,
  busy,
  onCancel,
  onRemove,
  provider
}: {
  account: ProviderAccount
  accountIndex: number
  busy: boolean
  onCancel: () => void
  onRemove: () => void
  provider: ProviderDefinition
}) {
  const { t } = useI18n()
  const name = accountLabel(provider, account, accountIndex)
  const copy = t.connectorsPage.accounts

  return (
    <div aria-label={copy.removeTitle(provider.label, name)} className="grid gap-2 border-b border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-3.5 py-3" role="alertdialog">
      <p className="text-xs font-medium text-(--ui-text-primary)">{copy.removeTitle(provider.label, name)}</p>
      <p className="text-xs text-(--ui-text-tertiary)">{copy.removeBody}</p>
      <div className="flex justify-end gap-2">
        <button className="text-xs text-(--ui-text-secondary)" onClick={onCancel} type="button">{t.connectors.cancel}</button>
        <button className="text-xs text-(--ui-text-danger)" disabled={busy} onClick={onRemove} type="button">{copy.remove}</button>
      </div>
    </div>
  )
}
