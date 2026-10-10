import type { ModelOptionProvider } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { getGlobalModelOptions, type ProfileScope, profileScopeKey } from '@/hermes'
import { useI18n } from '@/i18n'
import { findCatalogProvider } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { $customModels, customModelSlug, withCustomModels } from '@/store/custom-models'

import { CONTROL_TEXT } from './constants'
import { ModelSelect } from './model-select'

export interface DelegationModelProviderValue {
  model: string
  provider: string
  /** Explicitly leave direct-endpoint routing. Other request settings stay intact. */
  resetDirectEndpoint: boolean
}

export const INHERIT_VALUE = '__inherit__'
export const MODEL_ONLY_VALUE = '__model_only__'
const DIRECT_VALUE = '__direct__'
const CUSTOM_PROVIDER_VALUE = '__custom_provider__'
// Provider item values have a separate namespace from editor actions.
const providerItem = (provider: string) => `provider:${provider}`

interface DelegationModelProviderFieldProps {
  model: string
  provider: string
  baseUrl?: string
  scope?: ProfileScope
  onChange: (next: DelegationModelProviderValue) => void
}

/** A route draft belongs to the same immutable owner as the config that seeded it. */
export function DelegationModelProviderField(props: DelegationModelProviderFieldProps) {
  const identity = JSON.stringify([profileScopeKey(props.scope), props.model, props.provider, props.baseUrl ?? ''])

  return <DelegationRouteEditor key={identity} {...props} />
}

function DelegationRouteEditor({ model, provider, baseUrl = '', scope, onChange }: DelegationModelProviderFieldProps) {
  const { t } = useI18n()
  const c = t.settings.config
  // Match tools/delegate_tool_config.py: native SDK routes ignore a direct URL.
  const nativeSdk = ['bedrock', 'vertex', 'google', 'google-genai'].includes(provider.trim().toLowerCase())
  const hasDirectEndpoint = Boolean(baseUrl.trim()) && !nativeSdk

  const initialMode = hasDirectEndpoint
    ? DIRECT_VALUE
    : provider
      ? providerItem(provider)
      : model
        ? MODEL_ONLY_VALUE
        : INHERIT_VALUE

  const [mode, setMode] = useState(initialMode)
  const [draftModel, setDraftModel] = useState(model)
  const [draftProvider, setDraftProvider] = useState(provider)
  const [useParentModel, setUseParentModel] = useState(Boolean(provider && !model))

  const modelOptions = useQuery({
    queryKey: ['model-options', 'delegation', profileScopeKey(scope)],
    queryFn: () => getGlobalModelOptions(undefined, scope),
    retry: false
  })

  const customModels = useStore($customModels)

  // Custom ids enrich only providers actually returned for this owner. They
  // cannot introduce another profile's endpoints/provider rows.
  const providers = useMemo(
    () =>
      withCustomModels(
        (modelOptions.data?.providers ?? []).filter(p => p.slug),
        customModels
      ),
    [modelOptions.data?.providers, customModels]
  )

  const selectedProvider: ModelOptionProvider | undefined = findCatalogProvider(providers, draftProvider)

  const displayProviders =
    draftProvider && !providers.some(p => p.slug === draftProvider)
      ? [
          {
            name: selectedProvider?.name ?? draftProvider,
            slug: draftProvider,
            models: selectedProvider?.models ?? []
          },
          ...providers
        ]
      : providers

  const inherit = mode === INHERIT_VALUE
  const direct = mode === DIRECT_VALUE
  const modelOnly = mode === MODEL_ONLY_VALUE
  const manualProvider = mode === CUSTOM_PROVIDER_VALUE
  const nextProvider = inherit || modelOnly ? '' : draftProvider
  const nextModel = inherit || useParentModel ? '' : draftModel
  const resetDirectEndpoint = !direct && (hasDirectEndpoint || mode !== initialMode)
  const changed = nextProvider !== provider || nextModel !== model || (Boolean(baseUrl.trim()) && resetDirectEndpoint)
  const validProvider = inherit || modelOnly || direct || Boolean(selectedProvider || customModelSlug(nextProvider))
  const validModel = inherit || useParentModel || Boolean(customModelSlug(nextModel))
  const pending = mode !== initialMode || draftModel !== model || draftProvider !== provider || changed

  const reset = () => {
    setMode(initialMode)
    setDraftModel(model)
    setDraftProvider(provider)
    setUseParentModel(Boolean(provider && !model))
  }

  const chooseProvider = (value: string) => {
    setMode(value)
    setUseParentModel(false)

    if (value === INHERIT_VALUE) {
      setDraftModel('')
    } else if (value === CUSTOM_PROVIDER_VALUE) {
      setDraftProvider('')
      setDraftModel('')
    } else if (value.startsWith('provider:')) {
      const selected = value.slice('provider:'.length)
      setDraftProvider(selected)

      // Never reinterpret a model from the previous route as a completed switch.
      if (selected !== draftProvider) {
        setDraftModel('')
      }
    }
  }

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Select onValueChange={chooseProvider} value={mode}>
          <SelectTrigger aria-label={c.delegationProviderSelectLabel} className={cn('min-w-48 flex-1', CONTROL_TEXT)}>
            <SelectValue placeholder={t.settings.model.provider} />
          </SelectTrigger>
          <SelectContent>
            {hasDirectEndpoint && <SelectItem value={DIRECT_VALUE}>{c.delegationDirect}</SelectItem>}
            <SelectItem value={INHERIT_VALUE}>{c.delegationInherit}</SelectItem>
            <SelectItem value={MODEL_ONLY_VALUE}>{c.delegationModelOnly}</SelectItem>
            {displayProviders.map(p => (
              <SelectItem key={p.slug} value={providerItem(p.slug)}>
                {p.name}
              </SelectItem>
            ))}
            <SelectItem value={CUSTOM_PROVIDER_VALUE}>{c.delegationCustomProvider}</SelectItem>
          </SelectContent>
        </Select>
        {!inherit && !useParentModel && (
          <ModelSelect
            aria-label={c.delegationModelSelectLabel}
            className="min-w-56 flex-1"
            models={direct || modelOnly || manualProvider ? [] : (selectedProvider?.models ?? [])}
            onValueChange={setDraftModel}
            provider={direct || modelOnly || manualProvider ? undefined : selectedProvider}
            providerSlug={modelOnly || direct || manualProvider ? '' : draftProvider}
            value={draftModel}
          />
        )}
      </div>
      {manualProvider && (
        <Input
          aria-label={c.delegationCustomProviderLabel}
          onChange={event => setDraftProvider(event.target.value)}
          placeholder="custom:provider"
          value={draftProvider}
        />
      )}
      {!inherit && !modelOnly && (
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Checkbox checked={useParentModel} onCheckedChange={checked => setUseParentModel(checked === true)} />
          {c.delegationParentModel}
        </label>
      )}
      {useParentModel && <p className="text-xs text-muted-foreground">{c.delegationParentModelWarning}</p>}
      {hasDirectEndpoint && (
        <p className="text-xs text-muted-foreground">{direct ? c.delegationDirectActive : c.delegationDirectClear}</p>
      )}
      {modelOptions.isError && (
        <div className="flex flex-wrap items-center gap-2" role="status">
          <p className="text-xs text-muted-foreground">{c.delegationCatalogFailed}</p>
          <Button
            disabled={modelOptions.isFetching}
            onClick={() => void modelOptions.refetch()}
            size="xs"
            variant="secondary"
          >
            {t.common.retry}
          </Button>
        </div>
      )}
      {pending && (
        <div className="flex items-center gap-2">
          <Button
            disabled={!changed || !validProvider || !validModel}
            onClick={() => onChange({ model: nextModel, provider: nextProvider, resetDirectEndpoint })}
            size="xs"
          >
            {t.common.apply}
          </Button>
          <Button onClick={reset} size="xs" variant="text">
            {t.common.cancel}
          </Button>
          <span className="text-xs text-muted-foreground">{c.delegationDraftHint}</span>
        </div>
      )}
    </div>
  )
}
