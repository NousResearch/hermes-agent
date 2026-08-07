import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { getGlobalModelOptions } from '@/hermes'
import type { ModelOptionProvider } from '@/types/hermes'

import { isConfigurableProvider } from './provider-grouping'

export interface ProviderModelCatalog {
  providers: ModelOptionProvider[]
  isPending: boolean
  isError: boolean
}

/**
 * Loads the backend model catalog and returns only providers that expose at
 * least one model, so the Provider Manager can list them on the left and show
 * their models on the right. Mirrors the filtering in
 * components/model-visibility-dialog.tsx but widens to unconfigured providers
 * (includeUnconfigured) so the manager can surface every model the backend
 * knows about, not just the currently-selected provider's.
 */
export function useProviderModelCatalog(): ProviderModelCatalog {
  const query = useQuery({
    queryKey: ['provider-model-manager', 'catalog'],
    queryFn: () => getGlobalModelOptions({ includeUnconfigured: true, explicitOnly: false })
  })

  const providers = useMemo(
    () =>
      (query.data?.providers ?? []).filter(
        // Keep any provider that exposes models. Also keep custom (user-defined)
        // providers even when they have no models yet — the manager is where the
        // user adds models, and a freshly-created custom provider starts with an
        // empty model list (all models disabled by default). Finally keep
        // inline-configurable built-in providers (api_key flow, not yet set up)
        // so the manager can surface an "Unconfigured" section and let the user
        // paste a key to activate them.
        provider =>
          (provider.models ?? []).length > 0 ||
          provider.is_user_defined === true ||
          isConfigurableProvider(provider)
      ),
    [query.data]
  )

  return {
    providers,
    isPending: query.isPending,
    isError: query.isError
  }
}
