import type { ModelOptionProvider } from '@hermes/shared'
import { useState } from 'react'

import { findCatalogProvider } from '@/lib/model-options'
import { customModelSlug } from '@/store/custom-models'

export interface DelegationModelProviderValue {
  model: string
  provider: string
  /** Explicitly leave direct-endpoint routing. Other request settings stay intact. */
  resetDirectEndpoint: boolean
}

export interface SavedRoute {
  model: string
  provider: string
  baseUrl: string
}

export const INHERIT_VALUE = '__inherit__'
export const MODEL_ONLY_VALUE = '__model_only__'
export const DIRECT_VALUE = '__direct__'
export const CUSTOM_PROVIDER_VALUE = '__custom_provider__'
const PROVIDER_PREFIX = 'provider:'
// Provider item values have a separate namespace from editor actions.
export const providerItem = (provider: string) => `${PROVIDER_PREFIX}${provider}`

// Match tools/delegate_tool_config.py: native SDK routes ignore a direct URL.
const NATIVE_SDK_PROVIDERS = new Set(['bedrock', 'vertex', 'google', 'google-genai'])

export function directEndpointActive({ provider, baseUrl }: SavedRoute): boolean {
  return Boolean(baseUrl.trim()) && !NATIVE_SDK_PROVIDERS.has(provider.trim().toLowerCase())
}

function savedMode(saved: SavedRoute): string {
  if (directEndpointActive(saved)) {
    return DIRECT_VALUE
  }

  if (saved.provider) {
    return providerItem(saved.provider)
  }

  return saved.model ? MODEL_ONLY_VALUE : INHERIT_VALUE
}

/** A provider the catalog or a typed slug names, and a model unless the route inherits it. */
function routeComplete(
  draft: { mode: string; provider: string; useParentModel: boolean },
  nextProvider: string,
  nextModel: string,
  providers: readonly ModelOptionProvider[]
): boolean {
  const providerless = [INHERIT_VALUE, MODEL_ONLY_VALUE, DIRECT_VALUE].includes(draft.mode)
  const listed = findCatalogProvider(providers, draft.provider)
  const providerKnown = Boolean(listed || customModelSlug(nextProvider))
  const modelInherited = draft.mode === INHERIT_VALUE || draft.useParentModel
  // A catalog row may list ids a user could never type (MoA preset names carry spaces).
  const modelKnown = Boolean(listed?.models?.includes(nextModel) || customModelSlug(nextModel))

  return (providerless || providerKnown) && (modelInherited || modelKnown)
}

/** The route an Apply would write, and whether Apply is allowed. */
function draftOutcome(
  saved: SavedRoute,
  draft: { mode: string; model: string; provider: string; useParentModel: boolean },
  providers: readonly ModelOptionProvider[]
) {
  const initialMode = savedMode(saved)
  const inherit = draft.mode === INHERIT_VALUE
  const modelOnly = draft.mode === MODEL_ONLY_VALUE
  const direct = draft.mode === DIRECT_VALUE
  const nextProvider = inherit || modelOnly ? '' : draft.provider
  const nextModel = inherit || draft.useParentModel ? '' : draft.model
  const hadDirect = directEndpointActive(saved)
  const resetDirectEndpoint = !direct && (hadDirect || draft.mode !== initialMode)
  const clearsEndpoint = Boolean(saved.baseUrl.trim()) && resetDirectEndpoint
  const changed = nextProvider !== saved.provider || nextModel !== saved.model || clearsEndpoint
  const edited = draft.mode !== initialMode || draft.model !== saved.model || draft.provider !== saved.provider

  return {
    canApply: changed && routeComplete(draft, nextProvider, nextModel, providers),
    pending: edited || changed,
    value: { model: nextModel, provider: nextProvider, resetDirectEndpoint }
  }
}

/** One local draft for the subagent route; nothing is written until the caller applies `value`. */
export function useDelegationRouteDraft(saved: SavedRoute, providers: readonly ModelOptionProvider[]) {
  const initialMode = savedMode(saved)
  const parentModelInitially = Boolean(saved.provider && !saved.model)
  const [mode, setMode] = useState(initialMode)
  const [model, setModel] = useState(saved.model)
  const [provider, setProvider] = useState(saved.provider)
  const [useParentModel, setUseParentModel] = useState(parentModelInitially)

  const reset = () => {
    setMode(initialMode)
    setModel(saved.model)
    setProvider(saved.provider)
    setUseParentModel(parentModelInitially)
  }

  const chooseMode = (value: string) => {
    setMode(value)
    setUseParentModel(false)

    if (value === INHERIT_VALUE) {
      setModel('')
    } else if (value === DIRECT_VALUE) {
      // The endpoint keeps its own saved route; a provider picked meanwhile must not ride along.
      setProvider(saved.provider)
      setModel(saved.model)
    } else if (value === CUSTOM_PROVIDER_VALUE) {
      setProvider('')
      setModel('')
    } else if (value.startsWith(PROVIDER_PREFIX)) {
      const selected = value.slice(PROVIDER_PREFIX.length)
      setProvider(selected)

      // Never reinterpret a model from the previous route as a completed switch.
      if (selected !== provider) {
        setModel('')
      }
    }
  }

  return {
    mode,
    model,
    provider,
    useParentModel,
    hasDirectEndpoint: directEndpointActive(saved),
    setModel,
    setProvider,
    setUseParentModel,
    chooseMode,
    reset,
    ...draftOutcome(saved, { mode, model, provider, useParentModel }, providers)
  }
}

export type DelegationRouteDraft = ReturnType<typeof useDelegationRouteDraft>
