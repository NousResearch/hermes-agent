import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'
import type { ModelOptionProvider } from '@/types/hermes'

/**
 * True only when a persisted **manual** composer pick has been removed from the
 * catalog — either the model was dropped from a still-present provider, or the
 * provider itself was renamed/removed in config (so the cached slug no longer
 * matches any catalog entry). A deauthed/re-auth provider still appears with
 * models: [], so that case correctly returns false. An empty model list,
 * a not-yet-loaded catalog (undefined/empty providers), or no pick at all
 * also return false to never clobber a still-valid or ambiguous pick.
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
    // The provider itself is no longer in the catalog — it was renamed
    // or removed in config (e.g. a native provider replaced by a custom
    // provider entry with a different slug). A deauthed / re-auth provider
    // still appears in the catalog with models: [], so this branch only
    // fires on genuine removal/rename, not temporary unavailability.
    // Failing to invalidate here leaves the stale pick overriding the
    // config default (see #81922).
    return true
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

export async function requestModelOptions({
  explicitOnly = true,
  gateway,
  refresh = false,
  sessionId
}: ModelOptionsRequest): Promise<ModelOptionsResponse> {
  if (gateway) {
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
      gatewayOptions = await gateway.request<ModelOptionsResponse>('model.options', params)
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
      const restOptions = await getGlobalModelOptions({ explicitOnly, ...(refresh ? { refresh: true } : {}) })

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

  return getGlobalModelOptions({ explicitOnly, ...(refresh ? { refresh: true } : {}) })
}
