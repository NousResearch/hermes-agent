import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { useI18n } from '@/i18n'
import { requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
import { $customModels, addCustomModel, isCustomModel, mergeCustomModels, removeCustomModel } from '@/store/custom-models'
import type { ModelOptionProvider, ModelPricing } from '@/types/hermes'

import type { HermesGateway } from '../hermes'
import { cn } from '../lib/utils'
import { startManualOnboarding } from '../store/onboarding'

import { InlineNotice } from './notifications'
import { Button } from './ui/button'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Input } from './ui/input'
import { Skeleton } from './ui/skeleton'

interface ModelPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  gw?: HermesGateway
  sessionId?: string | null
  currentModel: string
  currentProvider: string
  onSelect: (selection: { provider: string; model: string }) => void
  /**
   * Optional class to apply to DialogContent. Use to override z-index when
   * stacking the picker on top of another fixed overlay (e.g. the desktop
   * onboarding overlay, which sits at z-1300; the default Dialog z-130 ends
   * up rendering underneath and blocks pointer events).
   */
  contentClassName?: string
}

export function ModelPickerDialog({
  open,
  onOpenChange,
  gw,
  sessionId,
  currentModel,
  currentProvider,
  onSelect,
  contentClassName
}: ModelPickerDialogProps) {
  const { t } = useI18n()
  const copy = t.modelPicker
  // Own the search term so we can filter manually. cmdk's built-in
  // shouldFilter reorders items by its fuzzy-match score (≈alphabetical with
  // an empty query), which destroys the backend's curated order. We disable
  // it and do a plain substring filter that preserves array order — matching
  // the `hermes model` CLI picker, which shows the curated list verbatim.
  const [search, setSearch] = useState('')
  const [showCustomInput, setShowCustomInput] = useState(false)
  const [customProvider, setCustomProvider] = useState('')
  const [customModelDraft, setCustomModelDraft] = useState('')
  const customModels = useStore($customModels)

  const modelOptions = useQuery({
    queryKey: ['model-options', sessionId || 'global'],
    queryFn: () => requestModelOptions({ gateway: gw, sessionId }),
    enabled: open
  })

  const providers = modelOptions.data?.providers ?? []

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    !!sessionId,
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  const loading = modelOptions.isPending && !modelOptions.data

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const selectModel = (provider: ModelOptionProvider, model: string) => {
    onSelect({ provider: provider.slug, model })
    onOpenChange(false)
  }

  // Open the full onboarding provider selector to add/switch a provider.
  // Reuses the entire onboarding flow (OAuth rows, API-key form, device-code,
  // model-confirm) instead of duplicating provider UI here. Closes the picker
  // so the onboarding overlay (z-1300) isn't rendered underneath it.
  const addProvider = () => {
    startManualOnboarding()
    onOpenChange(false)
  }

  const resetCustomInput = () => {
    setShowCustomInput(false)
    setCustomProvider('')
    setCustomModelDraft('')
  }

  const openCustomInput = () => {
    setCustomProvider(optionsProvider || currentProvider || providers[0]?.slug || '')
    setShowCustomInput(true)
  }

  // Persist the typed model against its provider, then select it immediately —
  // going through the same onSelect({ provider, model }) contract as backend
  // rows so custom picks apply exactly like curated ones.
  const submitCustomModel = () => {
    const slug = customProvider.trim()
    const model = customModelDraft.trim()

    if (!slug || !model) {
      return
    }

    addCustomModel(slug, model)
    onSelect({ provider: slug, model })
    resetCustomInput()
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className={cn('max-h-[85vh] max-w-2xl gap-0 overflow-hidden p-0', contentClassName)}>
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription className="font-mono text-xs leading-relaxed">
            {copy.current} {optionsModel || currentModel || copy.unknown}
            {optionsProvider || currentProvider ? ` · ${optionsProvider || currentProvider}` : ''}
          </DialogDescription>
        </DialogHeader>

        {showCustomInput ? (
          <CustomModelForm
            draft={customModelDraft}
            onCancel={resetCustomInput}
            onDraftChange={setCustomModelDraft}
            onProviderChange={setCustomProvider}
            onSubmit={submitCustomModel}
            provider={customProvider}
            providers={providers}
          />
        ) : (
          <Command className="rounded-none bg-card" shouldFilter={false}>
            <CommandInput autoFocus onValueChange={setSearch} placeholder={copy.search} value={search} />
            <CommandList className="max-h-96">
              {!loading && !error && <CommandEmpty>{copy.noModels}</CommandEmpty>}
              <ModelResults
                currentModel={optionsModel || currentModel}
                currentProvider={optionsProvider || currentProvider}
                customModels={customModels}
                error={error}
                loading={loading}
                onRemoveCustom={removeCustomModel}
                onSelectModel={selectModel}
                providers={providers}
                search={search}
              />
            </CommandList>
          </Command>
        )}

        <DialogFooter className="flex-row items-center justify-end gap-2 bg-card p-3">
          {!showCustomInput && (
            <Button className="mr-auto" onClick={openCustomInput} variant="ghost">
              {copy.customModel.add}
            </Button>
          )}
          <Button onClick={addProvider} variant="ghost">
            {copy.addProvider}
          </Button>
          <Button onClick={() => onOpenChange(false)} variant="outline">
            {t.common.cancel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function CustomModelForm({
  provider,
  providers,
  draft,
  onProviderChange,
  onDraftChange,
  onSubmit,
  onCancel
}: {
  provider: string
  providers: ModelOptionProvider[]
  draft: string
  onProviderChange: (slug: string) => void
  onDraftChange: (value: string) => void
  onSubmit: () => void
  onCancel: () => void
}) {
  const { t } = useI18n()
  const copy = t.modelPicker.customModel

  return (
    <div className="space-y-3 border-b border-border bg-card px-4 py-3">
      <p className="text-sm font-medium">{copy.heading}</p>
      {providers.length > 0 && (
        <div className="space-y-1.5">
          <span className="text-xs text-muted-foreground">{copy.provider}</span>
          <div className="flex flex-wrap gap-1.5">
            {providers.map(p => (
              <Button
                className="text-xs"
                key={p.slug}
                onClick={() => onProviderChange(p.slug)}
                size="sm"
                variant={provider === p.slug ? 'default' : 'outline'}
              >
                {p.name}
              </Button>
            ))}
          </div>
        </div>
      )}
      <div className="space-y-1.5">
        <span className="text-xs text-muted-foreground">{copy.modelId}</span>
        <div className="flex gap-2">
          <Input
            autoFocus
            className="font-mono"
            disabled={!provider}
            onChange={event => onDraftChange(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                onSubmit()
              }
            }}
            placeholder={copy.placeholder}
            value={draft}
          />
          <Button disabled={!provider || !draft.trim()} onClick={onSubmit} size="sm">
            {copy.addButton}
          </Button>
          <Button onClick={onCancel} size="sm" variant="outline">
            {t.common.cancel}
          </Button>
        </div>
      </div>
    </div>
  )
}

function ModelResults({
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  customModels,
  onRemoveCustom,
  onSelectModel,
  search
}: {
  loading: boolean
  error: string | null
  providers: ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  customModels: Record<string, string[]>
  onRemoveCustom: (provider: string, model: string) => void
  onSelectModel: (provider: ModelOptionProvider, model: string) => void
  search: string
}) {
  const { t } = useI18n()
  const copy = t.modelPicker

  if (loading) {
    return <LoadingResults />
  }

  if (error) {
    return (
      <div className="px-3 py-3">
        <InlineNotice kind="error" title={copy.loadFailed}>
          {error}
        </InlineNotice>
      </div>
    )
  }

  if (providers.length === 0) {
    return <div className="px-4 py-6 text-sm text-muted-foreground">{copy.noAuthenticatedProviders}</div>
  }

  const q = normalize(search)

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q ||
    model.toLowerCase().includes(q) ||
    provider.name.toLowerCase().includes(q) ||
    provider.slug.toLowerCase().includes(q)

  // A provider is selectable when it has curated models OR the user added a
  // custom model to it — so a custom entry survives a picker reopen even when
  // the provider's backend catalog is empty. Switching to a NOT-yet-configured
  // provider still goes through the "Add provider" footer button.
  const configured = providers.filter(
    p => (p.models ?? []).length > 0 || (customModels[p.slug] ?? []).length > 0
  )

  return (
    <>
      {configured.map(provider => {
        // Preserve the backend's curated order, then append custom entries —
        // filter in place, no re-sort.
        const models = mergeCustomModels(provider.slug, provider.models, customModels).filter(m =>
          matches(provider, m)
        )

        if (models.length === 0) {
          return null
        }

        const unavailable = new Set(provider.unavailable_models ?? [])

        return (
          <CommandGroup heading={<ProviderHeading provider={provider} />} key={provider.slug}>
            {provider.warning && (
              <div className="px-2 pb-2">
                <InlineNotice className="px-2.5 py-1.5 text-xs" kind="warning">
                  {provider.warning}
                </InlineNotice>
              </div>
            )}
            {models.map(model => {
              const isCurrent = model === currentModel && provider.slug === currentProvider
              const custom = isCustomModel(provider.slug, model, customModels)
              const price = provider.pricing?.[model]
              const locked = unavailable.has(model)

              return (
                <CommandItem
                  className={cn(
                    'flex items-center gap-2 pl-6 font-mono',
                    isCurrent &&
                      'bg-primary text-primary-foreground data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground',
                    locked && 'cursor-not-allowed opacity-45'
                  )}
                  disabled={locked}
                  key={`${provider.slug}:${model}`}
                  onSelect={() => {
                    if (!locked) {
                      onSelectModel(provider, model)
                    }
                  }}
                  value={`${provider.slug}:${model}`}
                >
                  <span className="min-w-0 flex-1 truncate">{model}</span>
                  {custom && (
                    <>
                      <span className="shrink-0 rounded-sm bg-muted px-1 py-0.5 text-[0.62rem] uppercase tracking-wide text-muted-foreground">
                        {copy.customModel.badge}
                      </span>
                      <button
                        aria-label={copy.customModel.remove}
                        className="shrink-0 rounded-sm px-1 text-xs text-muted-foreground hover:text-foreground"
                        onClick={event => {
                          event.stopPropagation()
                          onRemoveCustom(provider.slug, model)
                        }}
                        onPointerDown={event => event.stopPropagation()}
                        title={copy.customModel.remove}
                        type="button"
                      >
                        ✕
                      </button>
                    </>
                  )}
                  {locked && (
                    <span className="shrink-0 text-[0.62rem] uppercase tracking-wide opacity-80">{copy.pro}</span>
                  )}
                  <ModelPrice isCurrent={isCurrent} price={price} />
                </CommandItem>
              )
            })}
            {unavailable.size > 0 && (
              <div className="px-6 pb-2 pt-1 text-[0.62rem] leading-relaxed text-muted-foreground">
                {copy.proNeedsSubscription}
              </div>
            )}
          </CommandGroup>
        )
      })}
    </>
  )
}

// Compact In/Out $/Mtok price tag, mirroring the CLI picker's price columns.
// Renders nothing when pricing is unavailable for the model.
function ModelPrice({ price, isCurrent }: { price?: ModelPricing; isCurrent: boolean }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  if (!price || (!price.input && !price.output)) {
    return null
  }

  if (price.free) {
    return (
      <span
        className={cn(
          'shrink-0 rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide',
          isCurrent ? 'bg-primary-foreground/20' : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
        )}
      >
        {copy.free}
      </span>
    )
  }

  return (
    <span
      className={cn(
        'shrink-0 text-[0.66rem] tabular-nums',
        isCurrent ? 'text-primary-foreground/80' : 'text-muted-foreground'
      )}
      title={copy.priceTitle}
    >
      {price.input || '?'} / {price.output || '?'}
    </span>
  )
}

function LoadingResults() {
  return (
    <CommandGroup heading={<Skeleton className="h-3 w-32" />}>
      {Array.from({ length: 4 }, (_, rowIndex) => (
        <div className="rounded-sm py-1.5 pl-6 pr-2" key={rowIndex}>
          <Skeleton className={cn('h-5', rowIndex % 3 === 0 ? 'w-3/5' : rowIndex % 3 === 1 ? 'w-4/5' : 'w-1/2')} />
        </div>
      ))}
    </CommandGroup>
  )
}

function ProviderHeading({ provider }: { provider: ModelOptionProvider }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  // free_tier is only set for Nous. true → "Free tier", false → "Pro".
  const tierBadge =
    provider.free_tier === true ? (
      <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
        {copy.freeTier}
      </span>
    ) : provider.free_tier === false ? (
      <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
        {copy.pro}
      </span>
    ) : null

  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate">{provider.name}</span>
      <span className="font-mono text-xs font-normal normal-case tracking-normal text-muted-foreground">
        {provider.slug} · {provider.total_models ?? provider.models?.length ?? 0}
      </span>
      {tierBadge}
    </span>
  )
}
