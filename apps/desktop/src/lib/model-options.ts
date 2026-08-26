import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'
import type { ModelOptionProvider } from '@/types/hermes'

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

  const row = providers.find(p => p.slug === provider || p.name === provider)

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

export function modelOptionsQueryKey(profile: null | string | undefined, sessionId?: null | string) {
  const profileKey = (profile ?? '').trim() || 'default'

  return ['model-options', profileKey, sessionId || 'global'] as const
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

    // A connected Desktop gateway can occasionally return only the current
    // provider/model (or an empty provider list) while its authenticated REST
    // catalog is already populated. Recover through the same profile-scoped
    // endpoint Settings uses, but keep the live session selection authoritative.
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

    if (gatewayOptions) {
      return gatewayOptions
    }

    throw gatewayError
  }

  return restModelOptions(explicitOnly, refresh, profile)
}
