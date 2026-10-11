import type {
  ComposioAccountRow,
  ComposioAccountStatus,
  ComposioCatalogResult,
  ComposioConnectResult,
  ComposioProviderRow,
  ComposioProviderState,
  ComposioProviderStatus,
  ComposioRemoveResult,
  ComposioStatusResult
} from '@hermes/shared/gateway-events'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { hermesConfigKey } from '@/app/hooks/use-config-record'
import type { ProfileScope } from '@/hermes'
import { profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { completeMcpDesktopOAuth, McpOAuthCancelled } from '@/lib/mcp-dashboard-oauth'
import { requestGatewayForAgent } from '@/store/gateway'

const COMPOSIO_MCP_NAME = 'composio'
const COMPOSIO_MCP_URL = 'https://connect.composio.dev/mcp'
const PROVIDER_QUERY_ROOT = ['composio-providers'] as const
const REFRESH_MS = 2_000
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

  if (url.protocol !== 'https:') {
    throw new Error('The provider did not return a secure sign-in URL.')
  }

  return url.toString()
}

export function composioProvidersMatch(providers: readonly ProviderDefinition[], query: string): boolean {
  return providers.some(provider => queryMatches(provider, query))
}

export function ComposioProviderRows({
  actionsDisabled = false,
  catalog,
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
    () => (catalog.data?.providers ?? []).filter(provider => queryMatches(provider, filterQuery)),
    [catalog.data?.providers, filterQuery]
  )

  const statusByProvider = useMemo(
    () => new Map((status.data?.providers ?? []).map(provider => [provider.provider, provider])),
    [status.data?.providers]
  )

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
            const isNativeAuth = nativeAuthProvider === provider.provider
            const providerWatching = watching.has(provider.provider)

            return (
              <article
                className="grid gap-2 rounded-lg border border-(--ui-stroke-tertiary) p-3"
                data-provider={provider.provider}
                key={provider.provider}
              >
                <div className="flex min-w-0 items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <h4 className="truncate text-sm font-medium text-(--ui-text-primary)">{provider.label}</h4>
                    <p className="text-xs text-(--ui-text-tertiary)">{statusCopy(state?.status, t)}</p>
                  </div>
                  {isNativeAuth ? (
                    <button className="text-xs text-(--ui-text-secondary) underline" onClick={cancelNativeAuthorization} type="button">
                      {t.connectors.cancel}
                    </button>
                  ) : null}
                </div>

                {accounts.length > 0 ? (
                  <ul className="grid gap-1.5">
                    {accounts.map(account => (
                      <li className="flex min-w-0 items-center gap-2 text-xs" key={account.id}>
                        <span className="min-w-0 flex-1 truncate text-(--ui-text-secondary)">
                          {account.alias || account.account_type || provider.label}
                        </span>
                        <span className="shrink-0 text-(--ui-text-tertiary)">{statusCopy(account.status, t)}</span>
                        {account.status !== 'pending' && (
                          <>
                            {RECONNECT_ACCOUNT_STATES.has(account.status) ? (
                              <button
                                className="shrink-0 text-(--ui-text-secondary) underline"
                                disabled={isBusy}
                                onClick={() => void startProviderConnect(provider)}
                                type="button"
                              >
                                {copy.card.verb.reconnect}
                              </button>
                            ) : null}
                            <button
                              className="shrink-0 text-(--ui-text-secondary) underline"
                              disabled={isBusy}
                              onClick={() => setRemoving({ account, provider })}
                              type="button"
                            >
                              {accountCopy.remove}
                            </button>
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                ) : null}

                {accounts.some(account => account.status === 'expired' || account.status === 'revoked') && (
                  <p className="text-[0.7rem] text-(--ui-text-tertiary)">{accountCopy.reconnectNote}</p>
                )}

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    className="w-fit rounded-md bg-(--ui-fill-secondary) px-2.5 py-1.5 text-xs text-(--ui-text-primary) disabled:opacity-50"
                    disabled={actionsDisabled || isBusy || isNativeAuth}
                    onClick={() => void startProviderConnect(provider)}
                    type="button"
                  >
                    {isBusy ? copy.card.state.connecting : accounts.length > 0 ? accountCopy.addAnother : accountCopy.add}
                  </button>

                  {providerWatching ||
                  ((state?.status === 'pending' || accounts.some(account => account.status === 'pending')) &&
                    !stopped.has(provider.provider)) ? (
                    <button
                      className="text-xs text-(--ui-text-secondary) underline"
                      onClick={() => stopWaiting(provider.provider)}
                      type="button"
                    >
                      {copy.card.verb.stopWaiting}
                    </button>
                  ) : null}

                  {stopped.has(provider.provider) ? (
                    <button
                      className="text-xs text-(--ui-text-secondary) underline"
                      onClick={() => void refreshStatus(provider.provider)}
                      type="button"
                    >
                      {t.connectors.refresh}
                    </button>
                  ) : null}
                </div>

                {errorProvider === provider.provider ? (
                  <p className="text-xs text-(--ui-text-danger)" role="alert">
                    {errorText ?? t.connectors.connectErrorFor(provider.label)}
                  </p>
                ) : null}

                {removing?.provider.provider === provider.provider ? (
                  <div aria-label={accountCopy.removeTitle(provider.label, removing.account.alias ?? removing.account.account_type ?? provider.label)} className="grid gap-2 rounded-md border border-(--ui-stroke-tertiary) p-2" role="alertdialog">
                    <p className="text-xs font-medium text-(--ui-text-primary)">
                      {accountCopy.removeTitle(provider.label, removing.account.alias ?? removing.account.account_type ?? provider.label)}
                    </p>
                    <p className="text-xs text-(--ui-text-tertiary)">{accountCopy.removeBody}</p>
                    <div className="flex justify-end gap-2">
                      <button className="text-xs text-(--ui-text-secondary)" onClick={() => setRemoving(null)} type="button">
                        {t.connectors.cancel}
                      </button>
                      <button className="text-xs text-(--ui-text-danger)" disabled={isBusy} onClick={() => void removeAccount()} type="button">
                        {accountCopy.remove}
                      </button>
                    </div>
                  </div>
                ) : null}
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
