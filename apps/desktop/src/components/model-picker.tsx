import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { getLocalModelsStatus } from '@/hermes'
import { useI18n } from '@/i18n'
import { modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { modelSearchText } from '@/lib/model-search-text'
import { currentPickerSelection } from '@/lib/model-status-label'
import { normalize } from '@/lib/text'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $localModelsEnabled } from '@/store/local-models-flag'
import { $localRuntimeJobs, runningModelDownloads, watchLocalRuntimeJobs } from '@/store/local-runtime-jobs'
import type { LocalModelLoadProgress, ModelOptionProvider, ModelPricing } from '@/types/hermes'

import type { HermesGateway } from '../hermes'
import { cn } from '../lib/utils'
import { startManualOnboarding } from '../store/onboarding'

import { InlineNotice } from './notifications'
import { Button } from './ui/button'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { HighlightMatches } from './ui/highlight-matches'
import { Skeleton } from './ui/skeleton'

interface ModelPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  gw?: HermesGateway
  sessionId?: string | null
  currentModel: string
  currentProvider: string
  onSelect: (selection: { provider: string; model: string }) => void
  ownerConnectionId?: string
  profile?: string
  request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  /**
   * Optional class for DialogContent. Use it to lift the picker onto a higher
   * rung of the overlay ladder when it opens over another fixed overlay (the
   * desktop onboarding overlay, say) — on the default modal rung it renders
   * underneath and blocks pointer events.
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
  ownerConnectionId,
  profile = 'default',
  request,
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

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, ownerConnectionId),
    queryFn: () => requestModelOptions({ gateway: gw, profile, request, sessionId }),
    enabled: open
  })

  // Live load state for the managed local server: which model is loading
  // into memory right now, with a REAL percent (per-tensor callback relayed
  // over the router's SSE stream). Polled only while the picker is open —
  // 2s idle cadence is enough for a bar under a ~40s load. Errors read as
  // "nothing loading" (remote-only installs have no local-models routes).
  // Every local-models read here sits behind the --local launch flag (strict:
  // the llamacpp provider group hides even with staged models on disk).
  const localModelsEnabled = $localModelsEnabled.get()

  const localStatus = useQuery({
    queryKey: ['local-models-loading', profile],
    queryFn: () => getLocalModelsStatus(),
    enabled: open && localModelsEnabled,
    refetchInterval: 2_000,
    retry: false
  })

  const loadingModels: Record<string, LocalModelLoadProgress> = localStatus.data?.loading ?? {}

  // Models on their way into the local library right now (downloads +
  // quickstart runs), rendered as grayed progress rows. The jobs store
  // republishes every ~700ms with fresh byte counts while anything runs —
  // and this dialog stays MOUNTED app-wide when closed — so subscribe only
  // to download identity (changes when a download starts/ends, and never
  // while closed); each row selects its own percent scalar (#72163 class).
  const downloadsKey = useStoreSelector($localRuntimeJobs, jobs =>
    open && localModelsEnabled
      ? runningModelDownloads(jobs)
          .map(job => `${job.job_id}\u0000${job.target}`)
          .join('\u0001')
      : ''
  )

  const downloads = useMemo(
    () =>
      downloadsKey === ''
        ? []
        : downloadsKey.split('\u0001').map(pair => {
            const [jobId, target] = pair.split('\u0000')

            return { jobId, target }
          }),
    [downloadsKey]
  )

  // Rediscover in-flight work on open: the poller idles when nothing was
  // running, and a download can start from any surface.
  useEffect(() => {
    if (open && localModelsEnabled) {
      watchLocalRuntimeJobs()
    }
  }, [open, localModelsEnabled])

  // A finished download turns into a real selectable model — refetch the
  // options so the placeholder row is replaced while the picker is open.
  const refetchOptions = modelOptions.refetch

  useEffect(() => {
    if (!open) {
      return
    }

    let prevActive = runningModelDownloads($localRuntimeJobs.get()).length > 0

    return $localRuntimeJobs.listen(next => {
      const active = runningModelDownloads(next).length > 0

      if (prevActive && !active) {
        void refetchOptions()
      }

      prevActive = active
    })
  }, [open, refetchOptions])

  const providers = modelOptions.data?.providers ?? []

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
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
  // so the onboarding overlay isn't rendered underneath it.
  const addProvider = () => {
    startManualOnboarding()
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className={cn('max-h-[85vh] max-w-2xl', contentClassName)}
      >
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription className="font-mono text-xs leading-relaxed">
            {copy.current} {optionsModel || currentModel || copy.unknown}
            {optionsProvider || currentProvider ? ` · ${optionsProvider || currentProvider}` : ''}
          </DialogDescription>
        </DialogHeader>

        <Command className="rounded-none bg-card" shouldFilter={false}>
          <CommandInput autoFocus onValueChange={setSearch} placeholder={copy.search} value={search} />
          <CommandList className="max-h-96">
            {!loading && !error && <CommandEmpty>{copy.noModels}</CommandEmpty>}
            <ModelResults
              currentModel={optionsModel || currentModel}
              currentProvider={optionsProvider || currentProvider}
              downloads={downloads}
              error={error}
              loading={loading}
              loadingModels={loadingModels}
              onSelectModel={selectModel}
              providers={providers}
              search={search}
            />
          </CommandList>
        </Command>

        <DialogFooter className="flex-row items-center justify-end gap-2 bg-card p-3">
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

function ModelResults({
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  downloads,
  loadingModels,
  onSelectModel,
  search
}: {
  loading: boolean
  error: string | null
  providers: ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  downloads: { jobId: string; target: string }[]
  loadingModels: Record<string, LocalModelLoadProgress>
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
    modelSearchText(model).toLowerCase().includes(q) ||
    provider.name.toLowerCase().includes(q) ||
    provider.slug.toLowerCase().includes(q)

  // Only configured providers (those with curated models) are selectable
  // here. Switching to a NOT-yet-configured provider goes through the
  // "Add provider" footer button, which opens the full onboarding selector.
  // The local provider sits behind the --local launch flag (strict: staged
  // models on disk don't show without it). Module-level read — a launch flag
  // can't change mid-session.
  const localModelsShown = $localModelsEnabled.get()

  const configured = providers.filter(
    p => (p.models ?? []).length > 0 && (localModelsShown || p.slug !== LOCAL_PROVIDER_SLUG)
  )

  // In-flight local downloads render as disabled progress rows: inside the
  // Local group when it exists, else as their own group (first download —
  // nothing staged yet, so the backend reports no Local provider at all).
  const visibleDownloads = downloads.filter(job => !q || (job.target || '').toLowerCase().includes(q))
  const hasLocalGroup = configured.some(p => p.slug === LOCAL_PROVIDER_SLUG)

  return (
    <>
      {configured.map(provider => {
        // Preserve the backend's curated order — filter in place, no re-sort.
        const models = (provider.models ?? []).filter(m => matches(provider, m))
        const groupDownloads = provider.slug === LOCAL_PROVIDER_SLUG ? visibleDownloads : []

        if (models.length === 0 && groupDownloads.length === 0) {
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
              // Managed local model loading into memory right now: show the
              // real load percent inline (keyed by exact model id — remote
              // providers never match).
              const loadProgress = loadingModels[model]

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
                  <span className="min-w-0 flex-1 truncate">
                    <HighlightMatches query={search} text={model} />
                  </span>
                  {loadProgress && (
                    <span className="flex shrink-0 items-center gap-1.5" title={copy.loadingIntoMemory}>
                      <span className="h-1 w-16 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
                        <span
                          className="block h-full rounded-full bg-primary transition-[width] duration-500"
                          style={{ width: `${Math.max(2, loadProgress.percent)}%` }}
                        />
                      </span>
                      <span className="text-[0.62rem] tabular-nums text-muted-foreground">{loadProgress.percent}%</span>
                    </span>
                  )}
                  {locked && (
                    <span className="shrink-0 text-[0.62rem] uppercase tracking-wide opacity-80">{copy.pro}</span>
                  )}
                  <ModelPrice isCurrent={isCurrent} price={price} />
                </CommandItem>
              )
            })}
            {groupDownloads.map(job => (
              <DownloadingModelRow jobId={job.jobId} key={job.jobId} target={job.target} />
            ))}
            {unavailable.size > 0 && (
              <div className="px-6 pb-2 pt-1 text-[0.62rem] leading-relaxed text-muted-foreground">
                {copy.proNeedsSubscription}
              </div>
            )}
          </CommandGroup>
        )
      })}
      {!hasLocalGroup && visibleDownloads.length > 0 && (
        <CommandGroup heading={copy.localDownloadsHeading} key="local-downloads">
          {visibleDownloads.map(job => (
            <DownloadingModelRow jobId={job.jobId} key={job.jobId} target={job.target} />
          ))}
        </CommandGroup>
      )}
    </>
  )
}

// The backend's provider row for staged local models (inventory.py's
// _local_runtime_row). Downloads-in-flight attach to this group.
const LOCAL_PROVIDER_SLUG = 'llamacpp'

// A model still downloading: visible so the user knows it's coming (and
// where it will land), disabled so it can't be selected early, with the
// same byte progress the settings pane shows. Percent is selected here, per
// row, so the poller's 700ms byte ticks repaint this leaf only.
function DownloadingModelRow({ jobId, target }: { jobId: string; target: string }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  const percent = useStoreSelector($localRuntimeJobs, jobs => jobs.find(job => job.job_id === jobId)?.percent ?? null)

  return (
    <CommandItem className="flex items-center gap-2 pl-6 font-mono opacity-60" disabled value={`downloading:${jobId}`}>
      <span className="min-w-0 flex-1 truncate">{target}</span>
      <span className="flex shrink-0 items-center gap-1.5" title={copy.downloading}>
        <span className="h-1 w-16 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
          <span
            className="block h-full rounded-full bg-primary transition-[width] duration-500"
            style={{ width: `${Math.max(2, percent ?? 0)}%` }}
          />
        </span>
        <span className="text-[0.62rem] tabular-nums text-muted-foreground">
          {typeof percent === 'number' ? `${percent}%` : copy.downloading}
        </span>
      </span>
    </CommandItem>
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
      <span className="shrink-0 inline-flex items-center gap-1.5">
        {typeof price.discount_percent === 'number' ? (
          <span
            className={cn(
              'rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold',
              isCurrent ? 'bg-primary-foreground/20' : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
            )}
          >
            -{price.discount_percent}%
          </span>
        ) : null}
        <span
          className={cn(
            'shrink-0 rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide',
            isCurrent ? 'bg-primary-foreground/20' : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
          )}
        >
          {copy.free}
        </span>
      </span>
    )
  }

  const onSale = typeof price.discount_percent === 'number' && Boolean(price.was_input || price.was_output)

  return (
    <span
      className={cn(
        'shrink-0 inline-flex items-center gap-1.5 text-[0.66rem] tabular-nums',
        isCurrent ? 'text-primary-foreground/80' : 'text-muted-foreground'
      )}
      title={copy.priceTitle}
    >
      {onSale ? (
        <span
          className={cn(
            'rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold',
            isCurrent ? 'bg-primary-foreground/20' : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
          )}
        >
          -{price.discount_percent}%
        </span>
      ) : null}
      <span>
        {price.input || '?'} / {price.output || '?'}
      </span>
      {onSale ? (
        <span
          className={cn(
            'line-through decoration-from-font opacity-70',
            isCurrent ? 'text-primary-foreground/60' : 'text-muted-foreground/80'
          )}
        >
          {copy.wasPrice} {price.was_input || '?'} / {price.was_output || '?'}
        </span>
      ) : null}
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
