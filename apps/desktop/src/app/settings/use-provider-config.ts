import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useMemo } from 'react'

import {
  deleteCustomEndpoint,
  discoverProviderModels,
  getCustomEndpoints,
  getGlobalModelOptions,
  saveCustomEndpoint,
  setCustomEndpointEnabled,
  setEnvVar,
  testCustomProviderConnection
} from '@/hermes'
import type { CustomEndpoint, CustomEndpointUpdate } from '@/types/hermes'

import {
  normalizeProviderName,
  type CustomProviderEntry,
  type CustomProviderModel
} from '@/lib/custom-provider-config'
import {
  $visibleModels,
  emptyProviderSentinelKey,
  modelVisibilityKey,
  setVisibleModels
} from '@/store/model-visibility'

const CATALOG_KEY = ['provider-model-manager', 'catalog'] as const
const ENDPOINTS_KEY = ['provider-config', 'endpoints'] as const

export interface UseProviderConfig {
  /** Custom providers from the canonical REST endpoints (for the Add/Edit form + delete). */
  customProviders: CustomProviderEntry[]
  isLoading: boolean
  isError: boolean
  /** True while a save/delete/enable mutation is in flight. */
  isSaving: boolean
  /** Create or update a custom provider via the REST endpoints API (key → .env). */
  saveCustomProvider: (entry: CustomProviderEntry) => Promise<void>
  /** Remove a custom provider by id (or `custom:<id>` slug). */
  deleteCustomProvider: (idOrSlug: string) => Promise<void>
  /** Toggle activation for a custom provider via `providers.<id>.enabled`. */
  setEnabled: (slug: string, enabled: boolean) => Promise<void>
  /** Query a custom provider's /models endpoint and merge discovered models
   *  into its config (all added as inactive). */
  discoverModels: (slug: string) => Promise<string[]>
  /** Manually add a single model to a custom provider (added as active). */
  addModel: (slug: string, model: CustomProviderModel) => Promise<void>
  /** Test a custom provider's connectivity (latency/error inline). */
  testProviderConnection: (slug: string) => Promise<{ ok: boolean; latencyMs?: number; error?: string }>
  /** Force the backend to re-probe all configured providers and refresh the
   *  catalog query. Used by the "Update list" button on built-in providers. */
  refreshCatalog: () => Promise<void>
  /** Persist API key (+ optional base URL override) for a built-in provider
   *  via setEnvVar. `keyEnv` is the env var name (e.g. "OPENAI_API_KEY"). */
  saveBuiltInCredentials: (keyEnv: string, apiKey: string, baseUrl?: string, slug?: string) => Promise<void>
}

/** Strip a leading `custom:` prefix from a catalog slug to get the endpoint id. */
function endpointIdFromSlug(slug: string): string {
  return slug.replace(/^custom:/, '')
}

/** Map a canonical REST endpoint into the form's CustomProviderEntry shape. */
function endpointToEntry(ep: CustomEndpoint): CustomProviderEntry {
  return {
    name: ep.id,
    base_url: ep.base_url,
    api_mode: undefined,
    models: (ep.models ?? []).map(id => ({ id }))
  }
}

/**
 * Reads/writes custom providers through the canonical REST endpoints API
 * (`/api/providers/custom-endpoints`), which persists to the keyed `providers:`
 * schema and stores API keys in `.env` behind `key_env` (never in config.yaml).
 * Enablement is the authoritative `providers.<id>.enabled` flag. Every mutation
 * re-probes the backend catalog and invalidates the Provider Manager queries so
 * the UI reflects the change immediately.
 */
export function useProviderConfig(): UseProviderConfig {
  const qc = useQueryClient()

  const endpointsQuery = useQuery({
    queryKey: ENDPOINTS_KEY,
    queryFn: getCustomEndpoints
  })

  const endpoints = useMemo<CustomEndpoint[]>(
    () => endpointsQuery.data?.endpoints ?? [],
    [endpointsQuery.data]
  )

  const customProviders = useMemo(
    () => endpoints.map(endpointToEntry),
    [endpoints]
  )

  const invalidate = useCallback(async () => {
    await getGlobalModelOptions({
      includeUnconfigured: true,
      explicitOnly: false,
      refresh: true
    })
    qc.invalidateQueries({ queryKey: CATALOG_KEY })
    qc.invalidateQueries({ queryKey: ENDPOINTS_KEY })
  }, [qc])

  const persist = useMutation({
    mutationFn: async (update: CustomEndpointUpdate) => {
      await saveCustomEndpoint(update)
      await invalidate()
    }
  })

  const findEndpoint = useCallback(
    (slug: string): CustomEndpoint | undefined => {
      const id = endpointIdFromSlug(slug)
      const norm = normalizeProviderName(id)
      return endpoints.find(ep => normalizeProviderName(ep.id) === norm)
    },
    [endpoints]
  )

  const saveCustomProvider = useCallback(
    async (entry: CustomProviderEntry) => {
      const id = normalizeProviderName(entry.name)
      const isNew = !endpoints.some(ep => normalizeProviderName(ep.id) === id)

      const update: CustomEndpointUpdate = {
        id,
        name: entry.name.trim(),
        base_url: entry.base_url.trim(),
        // Empty string means "leave the existing secret alone" → send undefined
        // so the backend does not clear a key the form never shows back.
        api_key: entry.api_key && entry.api_key !== '' ? entry.api_key : undefined,
        api_mode: entry.api_mode,
        discover_models: true,
        models: (entry.models ?? []).map(m => m.id),
        model: entry.models?.[0]?.id ?? ''
      }
      await persist.mutateAsync(update)

      // A brand-new provider starts with every model hidden (the store's default
      // for an uncustomized provider is "all visible", so we write the hide-all
      // sentinel explicitly). Editing leaves current visibility untouched.
      if (isNew) {
        const slug = `custom:${id}`
        const next = new Set($visibleModels.get() ?? [])
        next.add(emptyProviderSentinelKey(slug))
        setVisibleModels(next)
      }
    },
    [endpoints, persist]
  )

  const deleteCustomProvider = useCallback(
    async (idOrSlug: string) => {
      const id = endpointIdFromSlug(idOrSlug)
      await deleteCustomEndpoint(id)
      await invalidate()
    },
    [invalidate]
  )

  const setEnabled = useCallback(
    async (slug: string, enabled: boolean) => {
      const id = endpointIdFromSlug(slug)
      await setCustomEndpointEnabled(id, enabled)
      await invalidate()
    },
    [invalidate]
  )

  const discoverModels = useCallback(
    async (slug: string): Promise<string[]> => {
      const ep = findEndpoint(slug)
      if (!ep) {
        throw new Error(`Unknown custom provider: ${slug}`)
      }

      const { models } = await discoverProviderModels({ baseUrl: ep.base_url })

      const merged = new Set<string>(ep.models ?? [])
      const added: string[] = []
      for (const d of models) {
        if (!merged.has(d.id)) {
          merged.add(d.id)
          added.push(d.id)
        }
      }
      const mergedList = [...merged]

      await persist.mutateAsync({
        id: ep.id,
        name: ep.name,
        base_url: ep.base_url,
        discover_models: true,
        models: mergedList,
        model: ep.model || mergedList[0] || ''
      })

      // Discovered models are inactive. If the provider was default-visible (no
      // explicit keys and no hide-all sentinel), write the sentinel so the whole
      // provider starts hidden; otherwise leave existing visibility untouched.
      const stored = $visibleModels.get() ?? new Set<string>()
      const prefix = `${slug}::`
      const hasExplicit = [...stored].some(k => k.startsWith(prefix) && !k.endsWith('::'))
      const hasSentinel = stored.has(emptyProviderSentinelKey(slug))
      if (!hasExplicit && !hasSentinel) {
        const next = new Set(stored)
        next.add(emptyProviderSentinelKey(slug))
        setVisibleModels(next)
      }

      return added
    },
    [findEndpoint, persist]
  )

  const addModel = useCallback(
    async (slug: string, model: CustomProviderModel) => {
      const ep = findEndpoint(slug)
      if (!ep) {
        throw new Error(`Unknown custom provider: ${slug}`)
      }

      const merged = new Set<string>(ep.models ?? [])
      merged.add(model.id)
      const mergedList = [...merged]

      await persist.mutateAsync({
        id: ep.id,
        name: ep.name,
        base_url: ep.base_url,
        discover_models: true,
        models: mergedList,
        model: ep.model || model.id
      })

      // Manual add is active: drop the hide-all sentinel and mark this model
      // visible so it shows up immediately.
      const stored = $visibleModels.get() ?? new Set<string>()
      const next = new Set(stored)
      next.delete(emptyProviderSentinelKey(slug))
      next.add(modelVisibilityKey(slug, model.id))
      setVisibleModels(next)
    },
    [findEndpoint, persist]
  )

  const testProviderConnection = useCallback(
    async (slug: string): Promise<{ ok: boolean; latencyMs?: number; error?: string }> => {
      const ep = findEndpoint(slug)
      if (!ep) {
        throw new Error(`Unknown custom provider: ${slug}`)
      }
      return testCustomProviderConnection({ baseUrl: ep.base_url })
    },
    [findEndpoint]
  )

  const refreshCatalog = useCallback(async () => {
    await getGlobalModelOptions({
      includeUnconfigured: true,
      explicitOnly: false,
      refresh: true
    })
    qc.invalidateQueries({ queryKey: CATALOG_KEY })
  }, [qc])

  const saveBuiltInCredentials = useCallback(
    async (keyEnv: string, apiKey: string, baseUrl?: string, slug?: string) => {
      if (apiKey) {
        await setEnvVar(keyEnv, apiKey)
      }
      if (baseUrl && slug) {
        const baseUrlEnv = `${slug.toUpperCase().replace(/-/g, '_')}_BASE_URL`
        await setEnvVar(baseUrlEnv, baseUrl)
      }
      qc.invalidateQueries({ queryKey: CATALOG_KEY })
      qc.invalidateQueries({ queryKey: ENDPOINTS_KEY })
    },
    [qc]
  )

  return {
    customProviders,
    isLoading: endpointsQuery.isPending,
    isError: endpointsQuery.isError,
    isSaving: persist.isPending,
    saveCustomProvider,
    deleteCustomProvider,
    setEnabled,
    discoverModels,
    addModel,
    testProviderConnection,
    refreshCatalog,
    saveBuiltInCredentials
  }
}
