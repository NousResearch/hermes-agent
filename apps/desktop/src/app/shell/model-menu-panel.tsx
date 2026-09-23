import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DropdownMenuItem, dropdownMenuRow } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { refreshModelOptions } from '@/lib/model-options'
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
  const { activeSessionId, controller } = useModelMenuController(props)

  // Explicit "Refresh Models": re-fetch the catalog with refresh:true so the
  // backend busts its 1h provider-model disk cache and re-pulls each provider's
  // live list. Fixes live-only models (e.g. OpenCode Zen free tier) vanishing
  // when the cache expires and falls back to the curated static list.
  // The shared helper cancels any in-flight fetch for this scope first, so the
  // refreshed catalog can never be overwritten by a slower open-time refetch.
  const refreshModels = async () => {
    if (refreshing) {
      return
    }

    setRefreshing(true)

    try {
      // The refreshed catalog is a hint list, never a reason to move the pick:
      // a custom slug the row lacks is still what the user selected.
      await refreshModelOptions(queryClient, {
        gateway,
        ownerConnectionId,
        profile,
        request: requestGateway,
        sessionId: activeSessionId
      })
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
