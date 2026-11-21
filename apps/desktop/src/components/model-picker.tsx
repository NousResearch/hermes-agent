import type { ModelOptionProvider, ModelPricing, PlanUsage, PlanUsageGauge } from '@hermes/shared'
import {
  gaugeTitle,
  hasTopTier,
  PAYMENT_PAUSE_REASON,
  planBlocks,
  planLimit,
  splitRecommended,
  usageTag
} from '@hermes/shared'
import { fuzzyRank, modelSearchText } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { type ReactElement, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { catalogProviderMatches, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import { accountResetMs, formatReset, modelResetMs } from '@/lib/provider-limit'
import { foldIncludes, normalize } from '@/lib/text'
import { formatPercentUsed, formatResetsIn } from '@/lib/usage-percent'
import { cn } from '@/lib/utils'
import { $customModels, addCustomModel, customModelCandidate, withCustomModels } from '@/store/custom-models'
import { $localModelsEnabled } from '@/store/local-models-flag'
import {
  type LocalModelsOwner,
  runningModelDownloads,
  useLocalModelsOwner,
  useLocalModelsStatus,
  useLocalRuntimeJobs
} from '@/store/local-runtime-jobs'
import type { LocalModelLoadProgress, LocalRuntimeJob } from '@/types/hermes'

import type { HermesGateway } from '../hermes'
import { startManualOnboarding } from '../store/onboarding'

import { InlineNotice } from './notifications'
import { ProviderStatusChip } from './provider-status-chip'
import { Badge } from './ui/badge'
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
  ownerConnectionId?: null | string
  profile?: string
  /** Desktop route profile for provider setup; `profile` may be the backend-side target. */
  setupProfile?: string
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
  setupProfile,
  contentClassName
}: ModelPickerDialogProps): ReactElement {
  const { t } = useI18n()
  const copy = t.modelPicker
  // Own the search term so we can filter manually. cmdk's built-in
  // shouldFilter reorders items by its fuzzy-match score (≈alphabetical with
  // an empty query), which destroys the backend's curated order. We disable
  // it: an empty query shows the curated list verbatim (like the `hermes
  // model` CLI picker) and a query ranks with the shared fuzzyRank.
  const [search, setSearch] = useState<string>('')
  // "Add custom model…" flips the search into slug entry: the typed id is
  // offered per provider even while it fuzzy-matches catalog rows.
  const [slugEntry, setSlugEntry] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)

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

  const owner: LocalModelsOwner = useLocalModelsOwner(profile, ownerConnectionId)
  const localStatus = useLocalModelsStatus(owner, open && localModelsEnabled, true)

  const loadingModels: Record<string, LocalModelLoadProgress> = localStatus.data?.loading ?? {}

  // Models on their way into the local library right now (downloads +
  // quickstart runs), rendered as grayed progress rows. The jobs store
  // republishes every ~700ms with fresh byte counts while anything runs —
  // and this dialog stays MOUNTED app-wide when closed — so subscribe only
  // to download identity (changes when a download starts/ends, and never
  // while closed); each row selects its own percent scalar (#72163 class).
  const downloadsKey: string = useLocalRuntimeJobs(
    owner,
    (jobs: readonly LocalRuntimeJob[]): string =>
      open && localModelsEnabled
        ? runningModelDownloads(jobs)
            .map(job => `${job.job_id}\u0000${job.target}`)
            .join('\u0001')
        : '',
    open && localModelsEnabled
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

  const customModels = useStore($customModels)
  const providers = withCustomModels(modelOptions.data?.providers ?? [], customModels)

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

  const selectCustomModel = (provider: ModelOptionProvider, model: string) => {
    addCustomModel(provider.slug, model, provider)
    selectModel(provider, model)
  }

  // Open the full onboarding provider selector to add/switch a provider.
  // Reuses the entire onboarding flow (OAuth rows, API-key form, device-code,
  // model-confirm) instead of duplicating provider UI here. Closes the picker
  // so the onboarding overlay isn't rendered underneath it.
  const addProvider = () => {
    const ownerProfile = setupProfile ?? profile

    startManualOnboarding(
      undefined,
      ownerConnectionId !== undefined ? { connectionId: ownerConnectionId, profile: ownerProfile } : ownerProfile
    )
    onOpenChange(false)
  }

  const enterSlug = () => {
    setSlugEntry(true)
    searchRef.current?.focus()
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
          <CommandInput
            autoFocus
            onValueChange={setSearch}
            placeholder={slugEntry ? copy.customModelPlaceholder : copy.search}
            ref={searchRef}
            value={search}
          />
          <CommandList className="max-h-96">
            {!loading && !error && <CommandEmpty>{copy.noModels}</CommandEmpty>}
            <ModelResults
              currentModel={optionsModel || currentModel}
              currentProvider={optionsProvider || currentProvider}
              downloads={downloads}
              error={error}
              loading={loading}
              loadingModels={loadingModels}
              offerCustom={slugEntry}
              onSelectCustomModel={selectCustomModel}
              onSelectModel={selectModel}
              owner={owner}
              providers={providers}
              search={search}
            />
          </CommandList>
        </Command>

        <DialogFooter className="flex-row items-center justify-end gap-2 bg-card p-3">
          <Button className="mr-auto" onClick={enterSlug} variant="ghost">
            {copy.addCustomModelAction}
          </Button>
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
  owner,
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  downloads,
  loadingModels,
  onSelectModel,
  onSelectCustomModel,
  offerCustom,
  search
}: {
  owner: LocalModelsOwner
  loading: boolean
  error: string | null
  providers: readonly ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  downloads: { jobId: string; target: string }[]
  loadingModels: Record<string, LocalModelLoadProgress>
  onSelectModel: (provider: ModelOptionProvider, model: string) => void
  onSelectCustomModel: (provider: ModelOptionProvider, model: string) => void
  /** Offer the typed id as a custom model even while catalog rows match. */
  offerCustom: boolean
  search: string
}): ReactElement {
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

  // Model rows rank with the same fuzzyRank + modelSearchText the web and TUI
  // pickers use, so one query orders identically on every surface. A query
  // that names the provider itself keeps its whole curated list, in order.
  const rankModels = (provider: ModelOptionProvider, models: readonly string[]) => {
    if (!q || foldIncludes(provider.name, q) || foldIncludes(provider.slug, q)) {
      return [...models]
    }

    return fuzzyRank(models, q, modelSearchText).map(r => r.item)
  }

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
  const visibleDownloads = downloads.filter(job => !q || foldIncludes(job.target || '', q))
  const hasLocalGroup = configured.some(p => p.slug === LOCAL_PROVIDER_SLUG)

  const groups = configured.map(provider => ({
    provider,
    // Empty query: the backend's curated order, verbatim (it already leads a
    // window plan with its featured picks).
    models: rankModels(provider, provider.models ?? []),
    downloads: provider.slug === LOCAL_PROVIDER_SLUG ? visibleDownloads : []
  }))

  const hasMatches = groups.some(g => g.models.length > 0 || g.downloads.length > 0)

  // A typed id nothing lists: one row per configured provider, current
  // provider first, so the slug is one Enter away and remembered afterwards.
  // While the query still matches catalog rows the section stays out of the
  // way unless the user asked for it via "Add custom model…".
  const customSlug = offerCustom || !hasMatches ? customModelCandidate(search, configured) : null

  const customProviders = customSlug
    ? [...configured].sort(
        (a, b) =>
          Number(catalogProviderMatches(b, currentProvider)) - Number(catalogProviderMatches(a, currentProvider))
      )
    : []

  return (
    <>
      {groups.map(({ provider, models, downloads: groupDownloads }) => {
        if (models.length === 0 && groupDownloads.length === 0) {
          return null
        }

        const unavailable = new Set(provider.unavailable_models ?? [])
        const accountLimited = accountResetMs(provider) !== null

        const renderModel = (model: string) => {
          const isCurrent = model === currentModel && catalogProviderMatches(provider, currentProvider)
          const price = provider.pricing?.[model]
          const locked = unavailable.has(model)
          const resetMs = modelResetMs(provider, model)
          const resetLabel = resetMs === null ? null : formatReset(resetMs)
          // Window plans: a model a full allowance stops looks unavailable; the limit line says why, once.
          const dimmed = !locked && (accountLimited || resetLabel !== null || planBlocks(provider, model))
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
              <span className={cn('min-w-0 flex-1 truncate', dimmed && !isCurrent && 'text-muted-foreground')}>
                <HighlightMatches foldSeparators query={search} text={model} />
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
              {locked && <span className="shrink-0 text-[0.62rem] uppercase tracking-wide opacity-80">{copy.pro}</span>}
              {resetLabel && (
                <Badge
                  className="font-sans tabular-nums"
                  size="xs"
                  title={t.shell.modelMenu.modelLimitedTip(resetLabel)}
                  variant="warn"
                >
                  {t.shell.modelMenu.modelResets(resetLabel)}
                </Badge>
              )}
              {provider.plan_usage?.kind === 'usage_window' ? (
                <UsageLevelTag isCurrent={isCurrent} level={usageTag(provider, model)} />
              ) : (
                <ModelPrice isCurrent={isCurrent} price={price} />
              )}
            </CommandItem>
          )
        }

        // Window plans, no query: "Recommended" (the plan's picks) then "Other models".
        // A query is its own ranking, so the list stays flat while searching.
        const split = q ? null : splitRecommended(provider, models)

        return (
          <CommandGroup heading={<ProviderHeading provider={provider} />} key={provider.slug}>
            {provider.plan_usage && <PlanUsageSummary provider={provider} usage={provider.plan_usage} />}
            {provider.warning && (
              <div className="px-2 pb-2">
                <InlineNotice className="px-2.5 py-1.5 text-xs" kind="warning">
                  {provider.warning}
                </InlineNotice>
              </div>
            )}
            {split ? (
              <>
                <ModelSectionLabel>{copy.sectionRecommended}</ModelSectionLabel>
                {split.recommended.map(renderModel)}
                {split.other.length > 0 && <ModelSectionLabel>{copy.sectionOther}</ModelSectionLabel>}
                {split.other.map(renderModel)}
              </>
            ) : (
              models.map(renderModel)
            )}
            {hasTopTier(provider, models) && (
              <div className="px-6 pb-2 pt-1 text-[0.62rem] leading-relaxed text-muted-foreground">
                {copy.largeFootnote}
              </div>
            )}
            {groupDownloads.map(job => (
              <DownloadingModelRow jobId={job.jobId} key={job.jobId} owner={owner} target={job.target} />
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
            <DownloadingModelRow jobId={job.jobId} key={job.jobId} owner={owner} target={job.target} />
          ))}
        </CommandGroup>
      )}
      {customSlug && customProviders.length > 0 && (
        <CommandGroup heading={copy.customModel} key="custom-model">
          {customProviders.map(provider => (
            <CommandItem
              className="flex items-center gap-2 pl-6 font-mono"
              key={`custom:${provider.slug}`}
              onSelect={() => onSelectCustomModel(provider, customSlug)}
              value={`custom:${provider.slug}:${customSlug}`}
            >
              <span className="min-w-0 flex-1 truncate">{customSlug}</span>
              <span className="shrink-0 text-[0.66rem] text-muted-foreground">{provider.name}</span>
            </CommandItem>
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
function DownloadingModelRow({
  owner,
  jobId,
  target
}: {
  owner: LocalModelsOwner
  jobId: string
  target: string
}): ReactElement {
  const { t } = useI18n()
  const copy = t.modelPicker
  const copyLocal = t.settings.localModels

  const percent: number | null = useLocalRuntimeJobs(
    owner,
    (jobs: readonly LocalRuntimeJob[]): number | null =>
      jobs.find((job: LocalRuntimeJob): boolean => job.job_id === jobId)?.percent ?? null,
    false
  )

  const paused: boolean = useLocalRuntimeJobs(
    owner,
    (jobs: readonly LocalRuntimeJob[]): boolean =>
      jobs.find((job: LocalRuntimeJob): boolean => job.job_id === jobId)?.status === 'paused',
    false
  )

  return (
    <CommandItem className="flex items-center gap-2 pl-6 font-mono opacity-60" disabled value={`downloading:${jobId}`}>
      <span className="min-w-0 flex-1 truncate">{target}</span>
      <span
        className="flex shrink-0 items-center gap-1.5"
        title={paused ? copyLocal.downloadPausedLabel : copy.downloading}
      >
        <span className="h-1 w-16 overflow-hidden rounded-full bg-(--ui-bg-tertiary)">
          <span
            className={cn(
              'block h-full rounded-full',
              paused ? 'bg-muted-foreground/60' : 'bg-primary transition-[width] duration-500'
            )}
            style={{ width: `${Math.max(2, percent ?? 0)}%` }}
          />
        </span>
        <span className="text-[0.62rem] tabular-nums text-muted-foreground">
          {paused ? copyLocal.downloadPausedLabel : typeof percent === 'number' ? `${percent}%` : copy.downloading}
        </span>
      </span>
    </CommandItem>
  )
}

function gaugeLabel(gauge: PlanUsageGauge, copy: { today: string; week: string; month: string; topTier: string }) {
  return gaugeTitle(gauge, { day: copy.today, month: copy.month, top_tier: copy.topTier, week: copy.week })
}

/** Bracketed `[||||      ]` allowance meter (fills as it is USED); a sliver always shows one
 * cell, not-quite-full never shows all ten. */
function AsciiBar({
  fill,
  isCurrent,
  percentLabel,
  style
}: {
  fill: number
  isCurrent?: boolean
  percentLabel?: string
  style: string
}) {
  const cells = 10
  const value = Math.min(1, Math.max(0, fill))
  let filled = Math.round(value * cells)

  if (value > 0 && filled === 0) {
    filled = 1
  }

  if (value < 1 && filled === cells) {
    filled = cells - 1
  }

  // Green / yellow / red: how much of the allowance is used.
  const color =
    style === 'red'
      ? 'text-red-600 dark:text-red-400'
      : style === 'yellow'
        ? 'text-amber-600 dark:text-amber-400'
        : 'text-emerald-600 dark:text-emerald-400'

  return (
    <span className="inline-flex items-center gap-1 font-mono" title={percentLabel}>
      <span
        aria-hidden
        className={cn(
          'whitespace-pre text-[0.62rem] leading-none tracking-[-0.08em]',
          isCurrent ? 'text-primary-foreground/85' : color
        )}
      >
        [{'|'.repeat(filled)}
        {' '.repeat(cells - filled)}]
      </span>
      {percentLabel ? (
        <span
          className={cn(
            'text-[0.62rem] tabular-nums',
            isCurrent ? 'text-primary-foreground/70' : 'text-muted-foreground'
          )}
        >
          {percentLabel}
        </span>
      ) : null}
    </span>
  )
}

/** Allowance gauge: fills as it is used, `NN% used` (same rule as the portal and CLI). */
function UsageBar({ gauge, isCurrent }: { gauge: PlanUsageGauge; isCurrent?: boolean }) {
  const { t } = useI18n()
  const used = 1 - Math.min(1, Math.max(0, gauge.left_fraction))

  return (
    <AsciiBar
      fill={used}
      isCurrent={isCurrent}
      percentLabel={t.modelPicker.percentUsed(formatPercentUsed(used))}
      style={gauge.style}
    />
  )
}

/** Window plans: "Recommended" / "Other models" title inside a provider group (not selectable). */
function ModelSectionLabel({ children }: { children: string }) {
  return (
    <div className="px-6 pb-1 pt-2 text-[0.62rem] font-medium uppercase tracking-wide text-muted-foreground">
      {children}
    </div>
  )
}

/** Window plans: Large (its own monthly limit) or Free (never runs out); other models carry no tag. */
export function UsageLevelTag({
  isCurrent = false,
  level
}: {
  isCurrent?: boolean
  level: 'free' | 'top_tier' | null
}) {
  const { t } = useI18n()

  if (!level) {
    return null
  }

  return (
    <span
      className={cn(
        'shrink-0 rounded-sm px-1 py-0.5 font-sans text-[0.62rem] font-medium',
        isCurrent
          ? 'bg-primary-foreground/20 text-primary-foreground'
          : level === 'free'
            ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
            : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
      )}
      title={level === 'top_tier' ? t.modelPicker.largeFootnote : undefined}
    >
      {t.modelPicker.usageLevels[level]}
    </span>
  )
}

/**
 * Window plans: one line above the model list once an allowance is full, saying what happens next —
 * paid (or Large) models now bill your top-up credits, pay as you go, or they wait for the reset.
 * Renders nothing while every allowance has room, for every other account, and while paused.
 */
export function PlanLimitLine({ className, provider }: { className?: string; provider: ModelOptionProvider }) {
  const { t } = useI18n()
  const copy = t.modelPicker
  const limit = planLimit(provider)

  if (!limit) {
    return null
  }

  const when = formatResetsIn(limit.gauge.resets_at_ms)
  const name = gaugeTitle(limit.gauge, copy.limitNames)
  const head = `${copy.limitReached(name)}${when ? ` · ${copy.resetsIn(when)}` : ''}`

  const then =
    limit.then === 'credits'
      ? limit.scope === 'all'
        ? copy.limitThenCredits
        : copy.limitThenCreditsLarge
      : limit.then === 'wait'
        ? limit.scope === 'all'
          ? copy.limitThenWait
          : copy.limitThenWaitLarge
        : null

  return (
    <div className={cn('text-[0.62rem] leading-snug', className)}>
      <span className={limit.then === 'wait' ? 'text-amber-700 dark:text-amber-400' : 'text-muted-foreground'}>
        {head}
      </span>
      {then ? (
        <span className={limit.then === 'credits' ? 'text-foreground' : 'text-muted-foreground'}> · {then}</span>
      ) : null}
    </div>
  )
}

function PlanUsageSummary({ provider, usage }: { provider: ModelOptionProvider; usage: PlanUsage }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  const windowGauges = usage.gauges ?? []

  // Window plans only: any other shape (an older backend's credit-grant gauge) shows nothing.
  if (usage.kind !== 'usage_window') {
    return null
  }

  const limited = planLimit(provider) !== null

  return (
    <div className="px-6 pb-2 pt-0.5 text-[0.62rem] leading-relaxed text-muted-foreground">
      <div className="flex flex-col gap-0.5">
        {limited ? (
          <PlanLimitLine provider={provider} />
        ) : (
          <div>
            {copy.included}
            {usage.paused_reason
              ? ` · ${usage.paused_reason === PAYMENT_PAUSE_REASON ? copy.pausedRenewal : copy.pausedOther}`
              : ''}
          </div>
        )}
        {windowGauges.map(g => {
          const when = formatResetsIn(g.resets_at_ms)

          return (
            <div className="flex items-center gap-2 font-mono" key={g.label}>
              <span className="w-[4.5rem] shrink-0">{gaugeLabel(g, copy)}</span>
              <UsageBar gauge={g} />
              {when ? <span className="text-muted-foreground">{copy.resetsIn(when)}</span> : null}
            </div>
          )
        })}
      </div>
    </div>
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

  // Two different facts wear the same badge: `free_tier` is a signed-in Nous
  // account on the free plan; `free_tier_row` is the no-account route's own
  // row. Either way the user is on free inference, so say so. Never match the
  // route by name — the label is copy.
  const planName = provider.plan_usage?.plan

  const tierBadge =
    provider.plan_usage?.kind === 'usage_window' && planName ? (
      <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
        {planName}
      </span>
    ) : provider.free_tier === true || provider.free_tier_row === true ? (
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
      <ProviderStatusChip provider={provider} />
    </span>
  )
}
