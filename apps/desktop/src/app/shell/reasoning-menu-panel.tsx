import { useStore } from '@nanostores/react'

import { useSessionView } from '@/app/chat/session-view'
import { DropdownMenuItem, dropdownMenuRow } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { catalogProviderMatches, currentModelCapabilities } from '@/lib/model-options'
import { collapseModelFamilies } from '@/store/model-visibility'

import { ModelOptionsContent, resolveFastControl } from './model-edit-submenu'
import { type ModelMenuHostProps, useModelMenuController } from './use-model-menu-controller'

/**
 * The composer's reasoning menu: the same options rows the catalog shows per
 * model row (Thinking / Fast / Effort), opened directly for the ACTIVE model
 * from its own pill. Same controller as the model menu, so an edit here and an
 * edit in the catalog submenu are one code path — session write, preset,
 * optimistic store, rollback.
 */
export function ReasoningMenuPanel(props: ModelMenuHostProps) {
  const { t } = useI18n()
  const view = useSessionView()
  const currentModel = useStore(view.$model)
  const { controller, defaultEffort, modelOptions } = useModelMenuController(props)
  const { model, provider } = controller.current
  const caps = currentModelCapabilities(modelOptions.data, provider, model)

  if (!model) {
    return (
      <DropdownMenuItem className={dropdownMenuRow} disabled>
        {t.shell.modelOptions.noOptions}
      </DropdownMenuItem>
    )
  }

  // The catalog's provider row is what gates fast; the live model id (which
  // may be the `-fast` sibling) decides which way the variant toggle points.
  const providerRow = modelOptions.data?.providers?.find(p => catalogProviderMatches(p, provider))
  const providerModels = providerRow?.models ?? []

  const family = collapseModelFamilies(providerModels, providerRow?.capabilities).find(item =>
    item.memberIds.includes(model)
  )

  const row = { isActive: true, model, presetModel: family?.id ?? model, provider }

  return (
    <ModelOptionsContent
      canDisableReasoning={caps?.can_disable_reasoning ?? undefined}
      defaultEffort={defaultEffort}
      effort={controller.current.effort}
      effortWire={controller.current.effortWire}
      fastControl={resolveFastControl(
        currentModel || model,
        providerModels,
        caps?.fast ?? false,
        controller.current.fast
      )}
      isActive
      model={model}
      modelDefaultEffort={caps?.default_reasoning_effort}
      onSelectModel={nextModel => controller.select(nextModel, provider)}
      onSetOptions={patch => controller.setOptions(patch, row)}
      provider={provider}
      reasoning={caps?.reasoning ?? true}
      reasoningBudget={caps?.reasoning_budget ?? undefined}
      reasoningControl={caps?.reasoning_control ?? undefined}
      reasoningEfforts={caps?.reasoning_efforts ?? undefined}
    />
  )
}
