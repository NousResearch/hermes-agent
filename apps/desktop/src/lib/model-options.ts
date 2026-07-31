import { type QueryClient } from '@tanstack/react-query'

import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'
import type { MoaConfigResponse, ModelOptionProvider } from '@/types/hermes'

export const MOA_MENU_CONFIG_QUERY_KEY = ['moa-menu-config'] as const

export const moaMenuConfigQueryKey = (profile: string) => [...MOA_MENU_CONFIG_QUERY_KEY, profile] as const

/** TanStack's default structural merge assigns `__proto__` through a plain
 * object and can turn an own preset key into the cache object's prototype.
 * Keep this JSON payload atomic for both component refetches and manual writes. */
export function setMoaMenuConfigQueryData(
  queryClient: QueryClient,
  profile: string,
  config: MoaConfigResponse | null
): void {
  queryClient.setQueryDefaults(MOA_MENU_CONFIG_QUERY_KEY, { structuralSharing: false })
  queryClient.setQueryData(moaMenuConfigQueryKey(profile), config)
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
  refresh?: boolean
  sessionId?: null | string
}

export function modelOptionsQueryKey(profile: null | string | undefined, sessionId?: null | string) {
  const profileKey = (profile ?? '').trim() || 'default'

  return ['model-options', profileKey, sessionId || 'global'] as const
}

export function requestModelOptions({
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

    return gateway.request<ModelOptionsResponse>('model.options', params)
  }

  return getGlobalModelOptions({ explicitOnly, ...(refresh ? { refresh: true } : {}) })
}
