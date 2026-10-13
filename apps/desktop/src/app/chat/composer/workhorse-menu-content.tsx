import { DEFAULT_REASONING_EFFORT } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'

import { useSessionView } from '@/app/chat/session-view'
import { ModelCatalogMenu, type ModelMenuController } from '@/app/shell/model-catalog-menu'
import { ModelOptionsContent } from '@/app/shell/model-edit-submenu'
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  dropdownMenuRow,
  dropdownMenuSectionLabel,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { currentModelCapabilities, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection, formatModelPillLabel } from '@/lib/model-status-label'
import { reasoningEffortLabel } from '@/lib/reasoning-effort'

import { type useWorkhorseConfig, workhorseRouteReset } from './workhorse-config'

/** Read the same owner/session catalog as the parent, but never its mutation controller. */
export function WorkhorseMenuContent({
  profile,
  connectionId,
  delegation,
  commit,
  request
}: ReturnType<typeof useWorkhorseConfig>) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.workhorse
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const parentModel = useStore(view.$model)
  const parentProvider = useStore(view.$provider)
  const parentEffort = useStore(view.$reasoningEffort)

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, connectionId),
    queryFn: () => requestModelOptions({ profile, request, sessionId })
  })

  const parent = currentPickerSelection({ model: parentModel, provider: parentProvider }, modelOptions.data)
  const model = String(delegation.model ?? '').trim()
  const provider = String(delegation.provider ?? '').trim()

  const rawEffort = String(delegation.reasoning_effort ?? '')
    .trim()
    .toLowerCase()

  const effort = rawEffort === 'false' || rawEffort === 'disabled' ? 'none' : rawEffort
  const capabilities = currentModelCapabilities(modelOptions.data, provider || parent.provider, model || parent.model)

  const select = (nextModel: string, nextProvider: string) =>
    commit('model', { model: nextModel, provider: nextProvider, ...workhorseRouteReset })

  const setEffort = (next: string) => commit('effort', { reasoning_effort: next })

  const controller: ModelMenuController = {
    allowSpeed: false,
    // Model and effort inheritance are independent: selecting a row must not
    // copy the parent's remembered preset or the catalog's default effort.
    applyPreset: () => undefined,
    current: { model, provider, effort: effort || parentEffort, fast: false },
    presetFor: () => ({}),
    select,
    // Like detached task overrides, editing a different row targets that child
    // model. Both writes use the existing owner queue, never session options or
    // the shared parent preset store. The footer edits effort independently.
    setOptions: (patch, row) => {
      if (patch.effort !== undefined) {
        if (!row.isActive) {
          select(row.model, row.provider)
        }

        setEffort(patch.effort)
      }
    }
  }

  return (
    <ModelCatalogMenu
      controller={controller}
      footer={
        <>
          <DropdownMenuItem className={dropdownMenuRow} onSelect={() => select('', '')}>
            {copy.resetModel}
          </DropdownMenuItem>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger className={dropdownMenuRow} data-testid="workhorse-reasoning-menu">
              {t.shell.modelOptions.effort}
              <span className="ml-auto text-(--ui-text-tertiary)">
                {effort ? reasoningEffortLabel(effort) : copy.effortInherit}
              </span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="w-52 p-0" sideOffset={4}>
              <DropdownMenuItem className={dropdownMenuRow} onSelect={() => setEffort('')}>
                {copy.resetEffort}
              </DropdownMenuItem>
              <ModelOptionsContent
                canDisableReasoning={capabilities?.can_disable_reasoning ?? undefined}
                defaultEffort={parentEffort && parentEffort !== 'none' ? parentEffort : DEFAULT_REASONING_EFFORT}
                effort={effort || parentEffort}
                fastControl={{ kind: 'none' }}
                isActive
                model={model || parent.model}
                onSelectModel={() => undefined}
                onSetOptions={patch => {
                  if (patch.effort !== undefined) {
                    setEffort(patch.effort)
                  }
                }}
                provider={provider || parent.provider}
                reasoning={capabilities?.reasoning !== false}
              />
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        </>
      }
      header={
        <DropdownMenuLabel className={dropdownMenuSectionLabel}>
          {copy.label} · {model ? formatModelPillLabel(model) : copy.inherit}
        </DropdownMenuLabel>
      }
      includeMoa
      ownerConnectionId={connectionId ?? undefined}
      profile={profile}
      request={request}
      sessionId={sessionId}
      visibilitySelection={parent}
    />
  )
}
