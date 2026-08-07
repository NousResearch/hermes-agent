import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import {
  $visibleModels,
  effectiveVisibleKeys,
  emptyProviderSentinelKey,
  isProviderSentinel,
  modelVisibilityKey,
  resolveVisibleKeys,
  setVisibleModels,
  toggleModelVisibility
} from '@/store/model-visibility'
import type { ModelOptionProvider } from '@/types/hermes'

export interface ProviderModelVisibility {
  /** Whether `model` is currently active for the selected provider. */
  isVisible: (model: string) => boolean
  /** Toggle `model` active/inactive for the selected provider. */
  toggle: (model: string) => void
  /** Bulk-set visibility for many models at once (Activate all / Deactivate
   *  all). Writes a single store update instead of N toggles. */
  setMany: (models: string[], visible: boolean) => void
  /** True when every model of the selected provider is hidden (sentinel set). */
  allHidden: boolean
  /** Count of currently-active models for the selected provider. */
  visibleCount: number
}

/**
 * Binds the persisted `$visibleModels` store to a single selected provider,
 * exposing per-model active state and a toggle. Reuses the load-bearing
 * `effectiveVisibleKeys` / `toggleModelVisibility` logic from
 * store/model-visibility.ts — this hook is a thin, provider-scoped view over
 * it and must not fork that logic.
 *
 * `enabled` reflects provider-level activation (from the backend `enabled`
 * flag). When false, the provider is deactivated: every model reads as hidden
 * and toggling is a no-op, so the model list can show a "disabled" state
 * without mutating `$visibleModels`.
 */
export function useProviderModelVisibility(
  providerSlug: string | null,
  providers: readonly ModelOptionProvider[],
  enabled = true
): ProviderModelVisibility {
  const stored = useStore($visibleModels)

  const visible = useMemo(() => effectiveVisibleKeys(stored, providers), [stored, providers])

  const isVisible = useCallback(
    (model: string) => {
      if (!enabled || !providerSlug) {
        return false
      }

      return visible.has(modelVisibilityKey(providerSlug, model))
    },
    [visible, providerSlug, enabled]
  )

  const toggle = useCallback(
    (model: string) => {
      if (!enabled || !providerSlug) {
        return
      }

      setVisibleModels(toggleModelVisibility($visibleModels.get(), providers, providerSlug, model))
    },
    [providers, providerSlug, enabled]
  )

  const setMany = useCallback(
    (models: string[], makeVisible: boolean) => {
      if (!enabled || !providerSlug) {
        return
      }

      // Start from the resolved working set (preserves other providers'
      // sentinels), then apply the bulk change for THIS provider only.
      const next = new Set(resolveVisibleKeys($visibleModels.get(), providers))
      const sentinel = emptyProviderSentinelKey(providerSlug)

      if (makeVisible) {
        next.delete(sentinel)
        for (const model of models) {
          next.add(modelVisibilityKey(providerSlug, model))
        }
      } else {
        for (const model of models) {
          next.delete(modelVisibilityKey(providerSlug, model))
        }
        // If every model is now hidden, record the explicit hide-all sentinel.
        const remaining = [...next].some(k => k.startsWith(`${providerSlug}::`) && !isProviderSentinel(k))
        if (!remaining) {
          next.add(sentinel)
        }
      }

      setVisibleModels(next)
    },
    [providers, providerSlug, enabled]
  )

  const allHidden = !enabled || (providerSlug ? (stored?.has(emptyProviderSentinelKey(providerSlug)) ?? false) : false)

  const visibleCount = enabled && providerSlug
    ? [...visible].filter(key => key.startsWith(`${providerSlug}::`)).length
    : 0

  return { isVisible, toggle, setMany, allHidden, visibleCount }
}
