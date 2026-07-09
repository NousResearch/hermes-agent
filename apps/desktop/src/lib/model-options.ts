import { getGlobalModelOptions, type HermesGateway, type ModelOptionsResponse } from '@/hermes'

interface ModelOptionsRequest {
  /** When false, include ambient/unconfigured providers (onboarding/setup
   *  surfaces). Chat pickers default to true so only explicitly configured
   *  providers are listed (#56974). */
  explicitOnly?: boolean
  gateway?: HermesGateway
  refresh?: boolean
  sessionId?: null | string
}

type ProviderCatalog = NonNullable<ModelOptionsResponse['providers']>

function mergeProviderCatalog(globalProviders: ProviderCatalog = [], scopedProviders: ProviderCatalog = []): ProviderCatalog {
  const scopedBySlug = new Map(scopedProviders.map(provider => [provider.slug, provider]))
  const merged = globalProviders.map(provider => scopedBySlug.get(provider.slug) ?? provider)
  const seen = new Set(globalProviders.map(provider => provider.slug))

  for (const provider of scopedProviders) {
    if (!seen.has(provider.slug)) {
      merged.push(provider)
    }
  }

  return merged
}

export function mergeModelOptions(
  globalOptions: ModelOptionsResponse,
  scopedOptions: ModelOptionsResponse
): ModelOptionsResponse {
  return {
    ...globalOptions,
    ...scopedOptions,
    providers: mergeProviderCatalog(globalOptions.providers, scopedOptions.providers)
  }
}

export async function requestModelOptions({
  explicitOnly = true,
  gateway,
  refresh = false,
  sessionId
}: ModelOptionsRequest): Promise<ModelOptionsResponse> {
  if (gateway) {
    const params: Record<string, unknown> = {}

    if (refresh) {
      params.refresh = true
    }

    if (explicitOnly) {
      params.explicit_only = true
    }

    if (!sessionId) {
      return gateway.request<ModelOptionsResponse>('model.options', params)
    }

    const [globalOptions, scopedOptions] = await Promise.all([
      gateway.request<ModelOptionsResponse>('model.options', params),
      gateway.request<ModelOptionsResponse>('model.options', {
        ...params,
        session_id: sessionId
      })
    ])

    return mergeModelOptions(globalOptions, scopedOptions)
  }

  return getGlobalModelOptions({ explicitOnly, ...(refresh ? { refresh: true } : {}) })
}
