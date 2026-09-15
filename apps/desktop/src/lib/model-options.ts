import { isRecord } from '@hermes/shared'
import type {
  JsonValue,
  ModelOptionProvider,
  ModelOptionsResult,
  RpcMethods,
  SavedKeyModelCapabilities,
  SavedKeyModelPricing
} from '@hermes/shared'

import { getGlobalModelOptions, type HermesGateway } from '@/hermes'
import type { GatewayRequest } from '@/lib/gateway-rpc'

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

// `model.options` types the per-model capability and pricing maps as free JSON, so this
// module is the one place that reads them back into the contract's row types.
const jsonRow = (value: JsonValue | null | undefined, key: string): JsonValue | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const row = value[key]

  return isRecord(row) ? row : undefined
}

/** A placeholder catalog row for a pick the catalog has not returned yet. */
export function optimisticProvider(slug: string, model: string, name = slug): ModelOptionProvider {
  return {
    aliases: null,
    api_url: null,
    auth_type: null,
    authenticated: null,
    capabilities: null,
    featured_models: null,
    free_tier: null,
    free_tier_pending: null,
    free_tier_row: null,
    is_current: false,
    is_user_defined: false,
    key_env: null,
    models: model ? [model] : [],
    name,
    native_catalog_empty: null,
    pricing: null,
    pricing_pending: null,
    slug,
    source: '',
    total_models: model ? 1 : 0,
    unavailable_models: null,
    warning: null
  }
}

/** One model's capability row off a catalog provider. */
export function providerCapabilities(
  provider: ModelOptionProvider | null | undefined,
  model: string
): SavedKeyModelCapabilities | undefined {
  // SAFETY: the gateway fills `capabilities` with `SavedKeyModelCapabilities` rows keyed by model id.
  return jsonRow(provider?.capabilities, model) as SavedKeyModelCapabilities | undefined
}

/** One model's pricing row off a catalog provider. */
export function providerPricing(
  provider: ModelOptionProvider | null | undefined,
  model: string
): SavedKeyModelPricing | undefined {
  // SAFETY: the gateway fills `pricing` with `SavedKeyModelPricing` rows keyed by model id.
  return jsonRow(provider?.pricing, model) as SavedKeyModelPricing | undefined
}

/** The catalog's option support for the current pick, or undefined while the
 *  catalog is loading / doesn't say. Callers treat undefined as "assume
 *  reasoning" so controls never flicker away during the fetch. */
export function currentModelCapabilities(
  options: ModelOptionsResult | null | undefined,
  provider: string,
  model: string
): SavedKeyModelCapabilities | undefined {
  return providerCapabilities(
    options?.providers?.find(row => catalogProviderMatches(row, provider)),
    model
  )
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
  request?: GatewayRequest
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
    const params: RpcMethods['model.options']['params'] = {}

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
      gatewayOptions = await dispatch('model.options', params)
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
