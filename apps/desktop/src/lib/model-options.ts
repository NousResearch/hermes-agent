import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'
import type { ModelOptionProvider } from '@/types/hermes'

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

function findCatalogProvider(
  providers: ModelOptionProvider[] | undefined,
  provider: string
): ModelOptionProvider | undefined {
  if (!providers?.length || !provider) {
    return undefined
  }

  return providers.find(row => catalogProviderMatches(row, provider))
}

function catalogHasModel(providers: ModelOptionProvider[] | undefined, model: string): boolean {
  return Boolean(model) && (providers?.some(provider => (provider.models ?? []).includes(model)) ?? false)
}

/**
 * True only when a persisted **manual** composer pick has been removed from the
 * catalog (its provider still ships models, but no longer this one) — so a new
 * chat would keep 404'ing the dead model. Deliberately conservative to never
 * clobber a still-valid pick: an unknown/absent provider, an empty model list
 * (re-auth / unconfigured), or a not-yet-loaded catalog all return false.
 */
export function manualPickRemoved(
  providers: ModelOptionProvider[] | undefined,
  provider: string,
  model: string
): boolean {
  if (!providers?.length || !provider || !model) {
    return false
  }

  const row = findCatalogProvider(providers, provider)

  if (!row) {
    return false
  }

  const models = row.models ?? []

  // Empty list means the provider is present but unconfigured / awaiting
  // re-auth, not that the model was dropped — leave the pick alone.
  if (models.length === 0) {
    return false
  }

  return !models.includes(model)
}

const MOA_PROVIDER_SLUG = 'moa'

/** True when THIS provider still lists `model`. Identity is the (provider,
 *  model) pair: the same id on OpenRouter does not count as the custom /
 *  first-party pick still being offered. */
export function selectionInCatalog(
  providers: ModelOptionProvider[] | undefined,
  model: string,
  provider?: string
): boolean {
  if (!providers?.length || !model || !provider) {
    return false
  }

  const row = findCatalogProvider(providers, provider)

  return Boolean(row && (row.models ?? []).includes(model))
}

/** First real (non-MoA) catalog row that still has models. */
export function firstSelectableCatalogModel(
  providers: ModelOptionProvider[] | undefined
): { model: string; provider: string } | null {
  if (!providers?.length) {
    return null
  }

  for (const provider of providers) {
    if (provider.slug === MOA_PROVIDER_SLUG) {
      continue
    }

    const model = provider.models?.[0]

    if (model) {
      return { model, provider: provider.slug }
    }
  }

  return null
}

/**
 * After Refresh Models replaces the catalog: keep the current **pair** when
 * that provider still lists the model. Never rewrite the provider just because
 * another catalog row (OpenRouter, …) exposes the same model id.
 *
 * Conservative like `manualPickRemoved` when the current provider is absent or
 * unconfigured (empty models) — except a fully gone model (group / credential
 * swap that dropped the id everywhere) still falls back to the first
 * selectable row. Returns null when the catalog is empty/unloaded so we never
 * wipe a selection on a failed or still-hydrating refresh.
 */
export function reconcileSelectionAfterCatalogRefresh(
  currentModel: string,
  providers: ModelOptionProvider[] | undefined,
  currentProvider?: string
): { model: string; provider: string } | null {
  const next = firstSelectableCatalogModel(providers)

  if (!next) {
    return null
  }

  if (!currentModel) {
    return next
  }

  if (currentProvider) {
    if (selectionInCatalog(providers, currentModel, currentProvider)) {
      return null
    }

    const row = findCatalogProvider(providers, currentProvider)

    // Present but empty: re-auth / unconfigured, not "model dropped".
    if (row && (row.models ?? []).length === 0) {
      return null
    }

    // This provider still ships models and dropped this one — fall back.
    if (row) {
      return next
    }

    // Provider missing from the refreshed catalog. Another provider listing
    // the same id is NOT a reason to switch (the OpenRouter collision).
    if (catalogHasModel(providers, currentModel)) {
      return null
    }

    return next
  }

  // No provider on the current pair: keep when the id is still offered
  // anywhere, otherwise fall back. Callers that know the provider must pass it
  // so a shared id cannot retarget the pick.
  if (catalogHasModel(providers, currentModel)) {
    return null
  }

  return next
}

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

function hasSelectableModels(options: ModelOptionsResponse | null | undefined): boolean {
  return options?.providers?.some(provider => (provider.models?.length ?? 0) > 0) ?? false
}

function restModelOptions(
  explicitOnly: boolean,
  refresh: boolean,
  profile?: null | string
): Promise<ModelOptionsResponse> {
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
}: ModelOptionsRequest): Promise<ModelOptionsResponse> {
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
    let gatewayOptions: ModelOptionsResponse | undefined

    try {
      gatewayOptions = await dispatch<ModelOptionsResponse>('model.options', params)
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
