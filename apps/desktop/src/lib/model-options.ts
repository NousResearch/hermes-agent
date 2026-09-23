import type { ModelCapabilities, ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'
import type { QueryClient } from '@tanstack/react-query'

import { getGlobalModelOptions, type HermesGateway } from '@/hermes'

type CatalogProviderIdentity = Pick<ModelOptionProvider, 'aliases' | 'name' | 'slug'>

/** True when `currentProvider` is this catalog row — slug, display name, or
 *  a custom-provider alias (`custom:<key>` vs the bare config key, #87035). */
export function catalogProviderMatches(provider: CatalogProviderIdentity, currentProvider: string): boolean {
  if (!currentProvider) {
    return false
  }

  return (
    provider.slug === currentProvider ||
    provider.name === currentProvider ||
    (provider.aliases?.includes(currentProvider) ?? false)
  )
}

/** A provider the user configured whose model catalog has not been discovered
 *  yet: the backend emits a user-defined row with an empty model list. Pickers
 *  keep it visible so a configured provider is never silently absent (#49656). */
export function isUndiscoveredConfiguredProvider(provider: ModelOptionProvider): boolean {
  return provider.is_user_defined === true && (provider.models?.length ?? 0) === 0
}

/** The catalog's option support for the current pick, or undefined while the
 *  catalog is loading / doesn't say. Callers treat undefined as "assume
 *  reasoning" so controls never flicker away during the fetch. */
export function currentModelCapabilities(
  options: ModelOptionsResult | null | undefined,
  provider: string,
  model: string
): ModelCapabilities | undefined {
  return options?.providers?.find(row => catalogProviderMatches(row, provider))?.capabilities?.[model]
}

// A picked (provider, model) pair is never retargeted from catalog membership.
// Picker rows are hints (discovered / curated / capped lists); a custom endpoint
// or a newer release legitimately serves ids the row lacks, and the backend
// soft-accepts them. Diffing the pick against the catalog silently swapped
// `deepseek-v4.1-flash` for the row's `-0731` sibling. The only authority on a
// pick's validity is the gateway's switch result.

interface ModelOptionsRequest {
  /** When false, include ambient/unconfigured providers (onboarding/setup
   *  surfaces). Chat pickers default to true so only explicitly configured
   *  providers are listed (#56974). */
  explicitOnly?: boolean
  gateway?: HermesGateway
  /** Owner-routed RPC. When set, catalog reads hit this dispatcher instead of
   *  `gateway.request` — a tile's model menu must not query the ambient
   *  chrome socket (#93892). */
  request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  /** Profile for the REST recovery path. Must match the catalog owner so a
   *  secondary tile does not fall back to the launch profile's models. */
  profile?: null | string
  refresh?: boolean
  sessionId?: null | string
}

export function modelOptionsQueryKey(
  profile: null | string | undefined,
  sessionId?: null | string,
  ownerConnectionId?: null | string
) {
  const profileKey = (profile ?? '').trim() || 'default'
  const ownerKey = (ownerConnectionId ?? '').trim()

  return ['model-options', profileKey, sessionId || 'global', ...(ownerKey ? ['owner', ownerKey] : [])] as const
}

function hasSelectableModels(options: ModelOptionsResult | null | undefined): boolean {
  return options?.providers?.some(provider => (provider.models?.length ?? 0) > 0) ?? false
}

function restModelOptions(
  explicitOnly: boolean,
  refresh: boolean,
  profile?: null | string
): Promise<ModelOptionsResult> {
  const opts = { explicitOnly, ...(refresh ? { refresh: true } : {}) }
  const profileKey = (profile ?? '').trim()

  return profileKey ? getGlobalModelOptions(opts, profileKey) : getGlobalModelOptions(opts)
}

export async function requestModelOptions({
  explicitOnly = true,
  gateway,
  profile,
  refresh = false,
  request,
  sessionId
}: ModelOptionsRequest): Promise<ModelOptionsResult> {
  const dispatch = request ?? (gateway ? gateway.request.bind(gateway) : null)

  if (dispatch) {
    const params: Record<string, unknown> = {}

    if (sessionId) {
      params.session_id = sessionId
    }

    if (refresh) {
      params.refresh = true
    }

    if (explicitOnly) {
      params.explicit_only = true
    }

    const profileKey = (profile ?? '').trim()

    if (profileKey) {
      params.profile = profileKey
    }

    let gatewayError: unknown
    let gatewayOptions: ModelOptionsResult | undefined

    try {
      gatewayOptions = await dispatch<ModelOptionsResult>('model.options', params)
    } catch (error) {
      gatewayError = error
    }

    if (gatewayOptions && hasSelectableModels(gatewayOptions)) {
      return gatewayOptions
    }

    // An owner-routed dispatcher can name a different registry connection than
    // the ambient REST client. Never recover that request through ambient REST:
    // profile names are not unique across sources, so doing so can cache B's
    // catalog under A's tile. Ambient gateway requests retain the compatibility
    // recovery used by older backends with incomplete model.options responses.
    if (!request) {
      try {
        const restOptions = await restModelOptions(explicitOnly, refresh, profile)

        if (hasSelectableModels(restOptions)) {
          return {
            ...restOptions,
            ...(gatewayOptions?.provider ? { provider: gatewayOptions.provider } : {}),
            ...(gatewayOptions?.model ? { model: gatewayOptions.model } : {})
          }
        }
      } catch {
        // Preserve the gateway result (or its original error) when the recovery
        // path is unavailable.
      }
    }

    if (gatewayOptions) {
      return gatewayOptions
    }

    throw gatewayError
  }

  return restModelOptions(explicitOnly, refresh, profile)
}

/** Refresh the model catalog for one picker scope. Cancels an in-flight fetch for the
 *  same key first (`revert: false` keeps the current state), so a slow response from
 *  the open-time refetch can never land after the refreshed catalog and overwrite it. */
export async function refreshModelOptions(
  queryClient: QueryClient,
  opts: {
    gateway?: HermesGateway
    ownerConnectionId?: null | string
    profile?: null | string
    request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
    sessionId?: null | string
  }
): Promise<void> {
  const queryKey = modelOptionsQueryKey(opts.profile, opts.sessionId, opts.ownerConnectionId)

  // `revert` is the SECOND argument (`cancelQueries(filters, cancelOptions)` —
  // react-query defaults it to `{ revert: true }`); passing it inside the
  // filters object would type-error AND silently revert.
  await queryClient.cancelQueries({ queryKey }, { revert: false })

  try {
    const next = await requestModelOptions({
      gateway: opts.gateway,
      profile: opts.profile,
      refresh: true,
      request: opts.request,
      sessionId: opts.sessionId
    })

    queryClient.setQueryData<ModelOptionsResult>(queryKey, next)
  } catch {
    // Network/backend hiccup — re-fetch just this scope, not every model-options query.
    void queryClient.invalidateQueries({ queryKey })
  }
}
