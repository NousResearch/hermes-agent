import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import type { ModelOptionProvider, ModelOptionsResponse, ModelPricing } from '@/types/hermes'

import type { HermesGateway } from '../hermes'
import { getGlobalModelOptions } from '../hermes'
import { cn } from '../lib/utils'
import { startManualOnboarding } from '../store/onboarding'

import { InlineNotice } from './notifications'
import { Button } from './ui/button'
import { Checkbox } from './ui/checkbox'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { Skeleton } from './ui/skeleton'

interface ModelPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  gw?: HermesGateway
  sessionId?: string | null
  currentModel: string
  currentProvider: string
  onSelect: (selection: { provider: string; model: string; persistGlobal: boolean }) => void
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
  const [persistGlobal, setPersistGlobal] = useState(!sessionId)
  // Own the search term so we can filter manually. cmdk's built-in
  // shouldFilter reorders items by its fuzzy-match score (≈alphabetical with
  // an empty query), which destroys the backend's curated order. We disable
  // it and do a plain substring filter that preserves array order — matching
  // the `hermes model` CLI picker, which shows the curated list verbatim.
  const [search, setSearch] = useState('')

  const modelOptions = useQuery({
    queryKey: ['model-options', sessionId || 'global'],
    queryFn: () => {
      if (gw && sessionId) {
        return gw.request<ModelOptionsResponse>('model.options', {
          session_id: sessionId
        })
      }

      return getGlobalModelOptions()
    },
    enabled: open
  })

  const providers = modelOptions.data?.providers ?? []
  const optionsModel = String(modelOptions.data?.model ?? currentModel ?? '')
  const optionsProvider = String(modelOptions.data?.provider ?? currentProvider ?? '')
  const loading = modelOptions.isPending && !modelOptions.data

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const selectModel = (provider: ModelOptionProvider, model: string) => {
    onSelect({
      provider: provider.slug,
      model,
      persistGlobal: persistGlobal || !sessionId
    })
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

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className={cn('max-h-[85vh] max-w-2xl gap-0 overflow-hidden p-0', contentClassName)}>
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>Switch model</DialogTitle>
          <DialogDescription className="font-mono text-xs leading-relaxed">
            current: {optionsModel || currentModel || '(unknown)'}
            {optionsProvider || currentProvider ? ` · ${optionsProvider || currentProvider}` : ''}
          </DialogDescription>
        </DialogHeader>

        <Command className="rounded-none bg-card" shouldFilter={false}>
          <CommandInput
            autoFocus
            onValueChange={setSearch}
            placeholder="Filter providers and models..."
            value={search}
          />
          <CommandList className="max-h-96">
            {!loading && !error && <CommandEmpty>No models found.</CommandEmpty>}
            <ModelResults
              currentModel={optionsModel || currentModel}
              currentProvider={optionsProvider || currentProvider}
              error={error}
              loading={loading}
              onSelectModel={selectModel}
              providers={providers}
              search={search}
            />
          </CommandList>
        </Command>

        <DialogFooter className="flex-row items-center justify-between gap-3 border-t border-border bg-card p-3 sm:justify-between">
          <label className="flex cursor-pointer select-none items-center gap-2 text-xs text-muted-foreground">
            <Checkbox
              checked={persistGlobal || !sessionId}
              disabled={!sessionId}
              onCheckedChange={checked => setPersistGlobal(checked === true)}
            />
            {sessionId ? 'Persist globally (otherwise this session only)' : 'Persist globally'}
          </label>

          <div className="flex items-center gap-2">
            <Button onClick={addProvider} variant="ghost">
              Add provider
            </Button>
            <Button onClick={() => onOpenChange(false)} variant="outline">
              Cancel
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ModelResults({
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  onSelectModel,
  search
}: {
  loading: boolean
  error: string | null
  providers: ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  onSelectModel: (provider: ModelOptionProvider, model: string) => void
  search: string
}) {
  if (loading) {
    return <LoadingResults />
  }

  if (error) {
    return (
      <div className="px-3 py-3">
        <InlineNotice kind="error" title="Could not load models">
          {error}
        </InlineNotice>
      </div>
    )
  }

  if (providers.length === 0) {
    return <div className="px-4 py-6 text-sm text-muted-foreground">No authenticated providers.</div>
  }

  const q = search.trim().toLowerCase()

  const matches = (provider: ModelOptionProvider, model: string) =>
    !q ||
    model.toLowerCase().includes(q) ||
    provider.name.toLowerCase().includes(q) ||
    provider.slug.toLowerCase().includes(q)

  // Only configured providers (those with curated models) are selectable
  // here. Switching to a NOT-yet-configured provider goes through the
  // "Add provider" footer button, which opens the full onboarding selector.
  const configured = providers.filter(p => (p.models ?? []).length > 0)

  return (
    <>
      {configured.map(provider => {
        // Preserve the backend's curated order — filter in place, no re-sort.
        const models = (provider.models ?? []).filter(m => matches(provider, m))

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
                  {locked && <span className="shrink-0 text-[0.62rem] uppercase tracking-wide opacity-80">Pro</span>}
                  <ModelPrice isCurrent={isCurrent} price={price} />
                </CommandItem>
              )
            })}
            {unavailable.size > 0 && (
              <div className="px-6 pb-2 pt-1 text-[0.62rem] leading-relaxed text-muted-foreground">
                Pro models need a paid Nous subscription.
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
        Free
      </span>
    )
  }

  return (
    <span
      className={cn(
        'shrink-0 text-[0.66rem] tabular-nums',
        isCurrent ? 'text-primary-foreground/80' : 'text-muted-foreground'
      )}
      title="Input / Output price per million tokens"
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
  // free_tier is only set for Nous. true → "Free tier", false → "Pro".
  const tierBadge =
    provider.free_tier === true ? (
      <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
        Free tier
      </span>
    ) : provider.free_tier === false ? (
      <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
        Pro
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
