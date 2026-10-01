import type { ModelCapabilities, ModelOptionProvider, ModelOptionsResult } from '@hermes/shared'

import { getGlobalModelOptions, type HermesGateway, type ProfileScope } from '@/hermes'

type CatalogProviderIdentity = Partial<Pick<ModelOptionProvider, 'aliases' | 'name'>> &
  Pick<ModelOptionProvider, 'slug'>

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

/** The catalog row for `currentProvider`, matched the same way as
 *  `catalogProviderMatches` (so a saved `custom:<key>` finds its row). */
export function findCatalogProvider<T extends CatalogProviderIdentity>(
  providers: readonly T[],
  currentProvider: string
): T | undefined {
  return providers.find(row => catalogProviderMatches(row, currentProvider))
}

/** The catalog's option support for the current pick, or undefined while the
 *  catalog is loading / doesn't say. Callers treat undefined as "assume
 *  reasoning" so controls never flicker away during the fetch. */
export function currentModelCapabilities(
  options: ModelOptionsResult | null | undefined,
  provider: string,
  model: string
): ModelCapabilities | undefined {
  return findCatalogProvider(options?.providers ?? [], provider)?.capabilities?.[model]
}

// A picked (provider, model) pair is never retargeted from catalog membership.
// Picker rows are hints (discovered / curated / capped lists); a custom endpoint
// or a newer release legitimately serves ids the row lacks, and the backend
// soft-accepts them. Diffing the pick against the catalog silently swapped
// `deepseek-v4.1-flash` for the row's `-0731` sibling. The only authority on a
// pick's validity is the gateway's switch result.

/** The single, deliberate exception to the sticky-pick rule above: the virtual
 *  `moa` provider. Its catalog row vanishes entirely once no MoA preset is
 *  enabled (`hermes_cli/inventory.py` filters it out of explicit-only
 *  catalogs), so a persisted manual pick pointing at it leaves the composer
 *  pill reading `Model · moa: default` forever (#90244). For this one provider
 *  — and only with a populated catalog in hand — row absence is authoritative:
 *  the pick reseeds from the profile default. Every other provider keeps the
 *  sticky behavior; an unloaded/empty catalog never clobbers anything. */
export function moaPickRemoved(
  options: { providers?: ModelOptionProvider[] | null } | null | undefined,
  provider: string,
  model: string
): boolean {
  if (!model.trim() || provider.trim().toLowerCase() !== 'moa') {
    return false
  }

  const providers = options?.providers

  if (!providers || providers.length === 0) {
    return false
  }

  const row = providers.find(p => (p.slug || p.name || '').toLowerCase() === 'moa')

  return !(row?.models ?? []).includes(model)
}

/** A bare provider slug is the pre-migration spelling of a custom entry. The
 *  catalog aliases `custom:<key>` with the bare config key (#87035), so a pick
 *  still carrying `nvidia` and a profile default of `custom:nvidia` name the
 *  SAME endpoint — the pick's spelling is simply stale, not a distinct choice.
 *  Shipping the bare slug resolves the NATIVE provider instead of the custom
 *  entry, silently dropping the entry's `extra_body` (e.g.
 *  `thinking: {type: adaptive}`) that the user configured (#81922).
 *
 *  Only a bare slug can be superseded: a pick that already names a provider
 *  class (`custom:<other>`, `moa`, `openai-codex`) is a different endpoint and
 *  keeps the sticky behavior. The bare slug must be the default's own key, so
 *  an unrelated manual pick (`anthropic` while the default is `custom:nvidia`)
 *  is never clobbered. */
export function customDefaultSupersedesPick(pickProvider: string, defaultProvider: string): boolean {
  const pick = (pickProvider || '').trim().toLowerCase()
  const fallback = (defaultProvider || '').trim().toLowerCase()

  if (!pick || pick === fallback || !fallback.startsWith('custom:')) {
    return false
  }

  const key = fallback.slice('custom:'.length).trim()

  return key.length > 0 && pick === key
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
  /** Captured REST owner for compatibility recovery. Required when `request`
   * names a non-ambient gateway; profile alone is not a unique owner. */
  scope?: ProfileScope
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

function restModelOptions(explicitOnly: boolean, refresh: boolean, scope?: ProfileScope): Promise<ModelOptionsResult> {
  const opts = { explicitOnly, ...(refresh ? { refresh: true } : {}) }

  return scope === undefined ? getGlobalModelOptions(opts) : getGlobalModelOptions(opts, scope)
}

export async function requestModelOptions({
  explicitOnly = true,
  gateway,
  profile,
  refresh = false,
  request,
  scope,
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

    // Owner-routed dispatchers may recover through REST only when the caller
    // supplies the same captured owner. Without it, profile names are not
    // unique across sources and ambient recovery would cross gateways.
    if (!request || scope !== undefined) {
      try {
        const restScope = scope ?? ((profile ?? '').trim() || undefined)
        const restOptions = await restModelOptions(explicitOnly, refresh, restScope)

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

  const restScope = scope ?? ((profile ?? '').trim() || undefined)

  return restModelOptions(explicitOnly, refresh, restScope)
}
