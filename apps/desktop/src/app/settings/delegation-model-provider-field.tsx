import { useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import type { ModelOptionProvider } from '@/hermes'
import { cn } from '@/lib/utils'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE } from './constants'
import type { DelegationModelsCopy } from './delegation-models-copy'
import type { DelegationModelProviderValue } from './delegation-models-state'

const INHERIT_VALUE = '__inherit__'
const DIRECT_VALUE = '__direct__'
const CUSTOM_VALUE = '__custom_model__'

/**
 * Adapted from webtecnica's #67523 guided provider/model picker (fd6e822f).
 * Catalog ownership is now profile-scoped in the parent. Intermediate pairs
 * are local drafts, never autosaves. Blank provider + explicit model remains
 * representable, including without a catalog (#67347's model-only reservation).
 */
export function DelegationModelProviderField({
  allowInherit = true,
  copy,
  directEndpoint = false,
  labelPrefix = '',
  model,
  onChange,
  parentProvider = '',
  provider,
  providers
}: {
  allowInherit?: boolean
  copy: DelegationModelsCopy
  directEndpoint?: boolean
  labelPrefix?: string
  model: string
  onChange: (next: DelegationModelProviderValue, providerChanged: boolean) => void
  parentProvider?: string
  provider: string
  providers: readonly ModelOptionProvider[]
}) {
  const [manualModel, setManualModel] = useState(false)
  const selectedProviderRow = providers.find(p => p.slug === (provider || parentProvider))
  const catalog = directEndpoint ? [] : (selectedProviderRow?.models ?? [])
  const modelItems = model && !catalog.includes(model) ? [model, ...catalog] : catalog
  const modelFreeText = manualModel || catalog.length === 0 || (!!model && !catalog.includes(model))
  const knownProvider = providers.some(p => p.slug === provider)

  return (
    <div className="grid w-full gap-2 sm:grid-cols-2">
      {providers.length === 0 ? (
        <Input
          aria-label={`${labelPrefix}${copy.provider}`}
          className={CONTROL_TEXT}
          onChange={event => onChange({ model: '', provider: event.target.value }, true)}
          placeholder={allowInherit ? copy.mainProvider : copy.provider}
          value={provider}
        />
      ) : (
        <Select
          onValueChange={value => {
            if (value === DIRECT_VALUE) {
              return
            }

            setManualModel(false)
            onChange({ model: '', provider: value === INHERIT_VALUE ? '' : value }, true)
          }}
          value={directEndpoint ? DIRECT_VALUE : provider || (allowInherit ? INHERIT_VALUE : EMPTY_SELECT_VALUE)}
        >
          <SelectTrigger aria-label={`${labelPrefix}${copy.provider}`} className={cn('w-full', CONTROL_TEXT)}>
            <SelectValue placeholder={copy.provider} />
          </SelectTrigger>
          <SelectContent>
            {directEndpoint && <SelectItem value={DIRECT_VALUE}>{copy.direct}</SelectItem>}
            {allowInherit && <SelectItem value={INHERIT_VALUE}>{copy.mainProvider}</SelectItem>}
            {provider && !knownProvider && <SelectItem value={provider}>{provider}</SelectItem>}
            {providers.filter(p => p.slug).map(p => (
              <SelectItem key={p.slug} value={p.slug}>
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {modelFreeText ? (
        <Input
          aria-label={`${labelPrefix}${copy.model}`}
          className={CONTROL_TEXT}
          onChange={event => onChange({ model: event.target.value, provider }, false)}
          placeholder={allowInherit && !provider && !directEndpoint ? copy.mainModel : copy.customModel}
          value={model}
        />
      ) : (
        <Select
          onValueChange={nextModel => {
            if (nextModel === CUSTOM_VALUE) {
              setManualModel(true)
            } else {
              onChange({ model: nextModel === INHERIT_VALUE ? '' : nextModel, provider }, false)
            }
          }}
          value={model || (allowInherit && !provider ? INHERIT_VALUE : EMPTY_SELECT_VALUE)}
        >
          <SelectTrigger aria-label={`${labelPrefix}${copy.model}`} className={cn('w-full', CONTROL_TEXT)}>
            <SelectValue placeholder={copy.model} />
          </SelectTrigger>
          <SelectContent>
            {allowInherit && !provider && <SelectItem value={INHERIT_VALUE}>{copy.mainModel}</SelectItem>}
            {modelItems.map(item => (
              <SelectItem key={item} value={item}>
                {item}
              </SelectItem>
            ))}
            <SelectItem value={CUSTOM_VALUE}>{copy.customModel}</SelectItem>
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
