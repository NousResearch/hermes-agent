import { useQuery } from '@tanstack/react-query'
import { useRef, useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE } from './constants'

export interface DelegationModelProviderValue {
  model: string
  provider: string
}

const INHERIT_VALUE = '__inherit__'

/**
 * Guided picker for delegation.model + delegation.provider.
 *
 * Replaces the two bare free-text inputs in Settings → Advanced with a paired
 * provider + model picker sourced from `getGlobalModelOptions()`.  Mirrors the
 * `FallbackModelsField` pattern (provider → model cascade, out-of-catalog
 * free-text fallback) but for a single delegation pair.
 *
 * "Inherit from main agent" is surfaced as a first-class dropdown option that
 * writes `""` to both config keys — the documented default behaviour.
 */
export function DelegationModelProviderField({
  model,
  provider,
  parentModelLabel,
  onChange
}: {
  model: string
  provider: string
  /** Human-readable label for the parent model, shown when inheriting. */
  parentModelLabel?: string
  onChange: (next: DelegationModelProviderValue) => void
}) {
  const { t } = useI18n()
  const m = t.settings.model

  const modelOptions = useQuery({
    queryKey: ['model-options', 'global'],
    queryFn: () => getGlobalModelOptions()
  })

  const providers = (modelOptions.data?.providers ?? []).filter(p => p.slug)

  const isInherit = !model && !provider

  // Track the last committed pair so we can ignore the autosave echo.
  const lastCommittedRef = useRef<string>(JSON.stringify({ model, provider }))

  const commit = (nextModel: string, nextProvider: string) => {
    const pair = JSON.stringify({ model: nextModel, provider: nextProvider })

    if (pair === lastCommittedRef.current) {
      return
    }

    lastCommittedRef.current = pair
    onChange({ model: nextModel, provider: nextProvider })
  }

  const selectedProviderRow = providers.find(p => p.slug === provider)
  const catalog = selectedProviderRow?.models ?? []

  // Keep an out-of-catalog model selectable.
  const modelItems = model && !catalog.includes(model) ? [model, ...catalog] : catalog

  // When provider has no catalog (e.g. user-defined endpoint), fall back to
  // free-text model input.
  const modelFreeText = provider !== '' && catalog.length === 0

  return (
    <div className="grid w-full gap-1.5">
      <Select
        onValueChange={value => {
          if (value === INHERIT_VALUE) {
            commit('', '')
          } else {
            commit('', value)
          }
        }}
        value={isInherit ? INHERIT_VALUE : provider || EMPTY_SELECT_VALUE}
      >
        <SelectTrigger className={cn('w-full', CONTROL_TEXT)}>
          <SelectValue placeholder={m.provider} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={INHERIT_VALUE}>
            {t.settings.config.delegationInherit ?? 'Inherit from main agent'}
          </SelectItem>
          {providers.map(p => (
            <SelectItem key={p.slug} value={p.slug}>
              {p.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {isInherit ? (
        parentModelLabel ? (
          <p className="text-xs text-muted-foreground">
            {t.settings.config.delegationCurrentlyInheriting?.replace('{model}', parentModelLabel) ??
              `Currently inheriting: ${parentModelLabel}`}
          </p>
        ) : null
      ) : modelFreeText ? (
        <Input
          className={CONTROL_TEXT}
          onChange={e => commit(e.target.value, provider)}
          placeholder={t.settings.config.delegationCustomModelId ?? 'Custom model ID…'}
          value={model}
        />
      ) : (
        <Select
          onValueChange={nextModel => commit(nextModel, provider)}
          value={model || EMPTY_SELECT_VALUE}
        >
          <SelectTrigger className={cn('w-full', CONTROL_TEXT)}>
            <SelectValue placeholder={m.model} />
          </SelectTrigger>
          <SelectContent>
            {modelItems.map(item => (
              <SelectItem key={item} value={item}>
                {item}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
