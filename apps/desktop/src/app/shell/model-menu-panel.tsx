import type { ModelOptionsResult } from '@hermes/shared'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DropdownMenuItem, dropdownMenuRow } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, reconcileSelectionAfterCatalogRefresh, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import { DEFAULT_REASONING_EFFORT } from '@/lib/reasoning-effort'
import { cn } from '@/lib/utils'

import { ModelCatalogMenu } from './model-catalog-menu'
import { type ModelMenuHostProps, useModelMenuController } from './use-model-menu-controller'

export { ModelMenuCloseContext } from './model-catalog-menu'
export type { ModelSelection } from './use-model-menu-controller'

/**
 * The composer's model menu: `ModelCatalogMenu` (the shared renderer) plus the
 * controller that gives a selection its meaning HERE (`useModelMenuController`).
 */
export function ModelMenuPanel(props: ModelMenuHostProps) {
  const { gateway, ownerConnectionId, profile = 'default', requestGateway } = props
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const [refreshing, setRefreshing] = useState(false)
  const queryClient = useQueryClient()
  // Bind to THIS surface's SessionView (primary or tile) so each pane's menu
  // shows/switches its own model — not the primary-only globals.
  const view = useSessionView()
  const activeSessionId = useStore(view.$runtimeId)
  const currentFastMode = useStore(view.$fast)
  const currentModel = useStore(view.$model)
  const currentProvider = useStore(view.$provider)
  const currentReasoningEffort = useStore(view.$reasoningEffort)
  const modelPresets = useStore($modelPresets)
  const defaultEffort = useStore($defaultReasoningEffort) || DEFAULT_REASONING_EFFORT
  const visibleModels = useStore($visibleModels)
  const touchesPrimary = view.kind === 'primary'

  // Subscribe to the SAME query the menu runs (identical key ⇒ React Query
  // dedupes, no second fetch). It must be a live subscription, not a cache
  // peek: with no model in the session store yet, currentPickerSelection falls
  // back to the catalog's reported current, and a non-reactive read would
  // never repaint that fallback once the catalog resolved.
  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, activeSessionId),
    queryFn: (): Promise<ModelOptionsResponse> =>
      requestModelOptions({ gateway, profile, request: requestGateway, sessionId: activeSessionId })
  })

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  // Explicit "Refresh Models": re-fetch the catalog with refresh:true so the
  // backend busts its 1h provider-model disk cache and re-pulls each provider's
  // live list. Fixes live-only models (e.g. OpenCode Zen free tier) vanishing
  // when the cache expires and falls back to the curated static list.
  const refreshModels = async () => {
    if (refreshing) {
      return
    }

    setRefreshing(true)

    try {
      const queryKey = modelOptionsQueryKey(profile, activeSessionId, ownerConnectionId)

      const next = await requestModelOptions({
        gateway,
        profile,
        refresh: true,
        request: requestGateway,
        sessionId: activeSessionId
      })

      queryClient.setQueryData<ModelOptionsResponse>(queryKey, next)

      // Group / credential swaps can return a catalog that no longer contains
      // the session's current model. The store + currentPickerSelection would
      // otherwise keep painting the stale id (it is not in the new list).
      const switchTo = reconcileSelectionAfterCatalogRefresh(optionsModel, next.providers)

      if (switchTo) {
        await onSelectModel({ ...switchTo, sessionId: activeSessionId || null })
      }
    } catch {
      // Network/backend hiccup — fall back to a plain invalidate so the next
      // open re-fetches (still cached, but no worse than before).
      void queryClient.invalidateQueries({ queryKey: ['model-options'] })
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <ModelCatalogMenu
      controller={controller}
      footer={
        <DropdownMenuItem
          className={cn(dropdownMenuRow, 'text-(--ui-text-tertiary)')}
          disabled={refreshing}
          onSelect={event => {
            event.preventDefault()
            void refreshModels()
          }}
        >
          <Codicon className={cn(refreshing && 'animate-spin')} name="sync" size="0.75rem" />
          {copy.refreshModels}
        </DropdownMenuItem>
      }
      gateway={gateway}
      includeMoa
      ownerConnectionId={ownerConnectionId}
      profile={profile}
      request={requestGateway}
      sessionId={activeSessionId}
    />
  )
}
