import type { ModelOptionProvider } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { findCatalogProvider } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { $customModels, withCustomModels } from '@/store/custom-models'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE } from './constants'
import { ModelSelect } from './model-select'

export interface DelegationModelProviderValue {
  model: string
  provider: string
}

export const INHERIT_VALUE = '__inherit__'
export const MODEL_ONLY_VALUE = '__model_only__'

interface DelegationModelProviderFieldProps {
  model: string
  provider: string
  onChange: (next: DelegationModelProviderValue) => void
}

/**
 * Guided picker for delegation.model + delegation.provider in Settings → Advanced.
 *
 * Replaces the two bare text fields with a paired provider and model picker
 * sourced from `getGlobalModelOptions()` and `$customModels`, matching the
 * Main Model and Fallback Models dropdown patterns.
 *
 * Special states:
 * - "Inherit from main agent": writes provider="" and model=""
 * - "Custom model (use parent credentials)": writes provider="" and model="<typed-model>"
 *   which is supported by the runtime resolver tools/delegate_tool.py.
 * - Standard provider selection: writes provider="<slug>" and model="<model>"
 */
export function DelegationModelProviderField({
  model,
  provider,
  onChange
}: DelegationModelProviderFieldProps) {
  const { t } = useI18n()
  const m = t.settings.model
  const c = t.settings.config

  const modelOptions = useQuery({
    queryKey: ['model-options', 'global'],
    queryFn: () => getGlobalModelOptions()
  })

  const customModels = useStore($customModels)

  const providers = useMemo(
    () => withCustomModels((modelOptions.data?.providers ?? []).filter(p => p.slug), customModels),
    [modelOptions.data?.providers, customModels]
  )

  // Determine current dropdown value
  const derivePickerValue = (p: string, mdl: string): string => {
    if (!p && !mdl) {
      return INHERIT_VALUE
    }

    if (!p && mdl) {
      return MODEL_ONLY_VALUE
    }

    return p
  }

  const [pickerValue, setPickerValue] = useState<string>(() => derivePickerValue(provider, model))
  const [draftModel, setDraftModel] = useState<string>(model)

  const lastCommittedRef = useRef<string>(JSON.stringify({ model, provider }))

  // Resync if external config changes (e.g. profile switch)
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see FallbackModelsField)
  useEffect(() => {
    const pair = JSON.stringify({ model, provider })

    if (pair !== lastCommittedRef.current) {
      lastCommittedRef.current = pair
      setPickerValue(derivePickerValue(provider, model))
      setDraftModel(model)
    }
  }, [model, provider])

  const commit = (nextModel: string, nextProvider: string) => {
    const serialized = JSON.stringify({ model: nextModel, provider: nextProvider })

    if (serialized === lastCommittedRef.current) {
      return
    }

    lastCommittedRef.current = serialized
    onChange({ model: nextModel, provider: nextProvider })
  }

  const handleProviderChange = (selected: string) => {
    setPickerValue(selected)

    if (selected === INHERIT_VALUE) {
      setDraftModel('')
      commit('', '')
    } else if (selected === MODEL_ONLY_VALUE) {
      // Model-only override: provider is empty, model is preserved or empty draft
      commit(draftModel, '')
    } else {
      // Provider changed to a concrete catalog provider.
      // If the current model belongs to this new provider, keep it; otherwise reset model.
      const provRow = findCatalogProvider(providers, selected)
      const provModels = provRow?.models ?? []
      const nextMdl = provModels.includes(draftModel) ? draftModel : ''
      setDraftModel(nextMdl)
      commit(nextMdl, selected)
    }
  }

  const handleModelChange = (nextModel: string) => {
    setDraftModel(nextModel)

    if (pickerValue === INHERIT_VALUE) {
      commit('', '')
    } else if (pickerValue === MODEL_ONLY_VALUE) {
      commit(nextModel, '')
    } else {
      commit(nextModel, pickerValue)
    }
  }

  const selectedProviderRow: ModelOptionProvider | undefined = useMemo(
    () => (pickerValue !== INHERIT_VALUE && pickerValue !== MODEL_ONLY_VALUE ? findCatalogProvider(providers, pickerValue) : undefined),
    [providers, pickerValue]
  )

  const providerModels = selectedProviderRow?.models ?? []

  // If a persisted provider isn't in catalog, keep it visible in options
  const displayProviders = useMemo(() => {
    if (
      pickerValue &&
      pickerValue !== INHERIT_VALUE &&
      pickerValue !== MODEL_ONLY_VALUE &&
      !findCatalogProvider(providers, pickerValue)
    ) {
      return [{ name: pickerValue, slug: pickerValue, models: [] }, ...providers]
    }

    return providers
  }, [providers, pickerValue])

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Select onValueChange={handleProviderChange} value={pickerValue || EMPTY_SELECT_VALUE}>
        <SelectTrigger aria-label={c.delegationProviderSelectLabel ?? 'Subagent Provider'} className={cn('min-w-48 flex-1', CONTROL_TEXT)}>
          <SelectValue placeholder={m.provider} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={INHERIT_VALUE}>
            {c.delegationInherit ?? 'Inherit from main agent'}
          </SelectItem>
          <SelectItem value={MODEL_ONLY_VALUE}>
            {c.delegationModelOnly ?? 'Custom model (use parent credentials)'}
          </SelectItem>
          {displayProviders.map(p => (
            <SelectItem key={p.slug} value={p.slug}>
              {p.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {pickerValue !== INHERIT_VALUE && (
        <ModelSelect
          aria-label={c.delegationModelSelectLabel ?? 'Subagent Model'}
          className="min-w-56 flex-1"
          models={providerModels}
          onValueChange={handleModelChange}
          provider={selectedProviderRow}
          providerSlug={pickerValue === MODEL_ONLY_VALUE ? '' : pickerValue}
          value={draftModel}
        />
      )}
    </div>
  )
}
