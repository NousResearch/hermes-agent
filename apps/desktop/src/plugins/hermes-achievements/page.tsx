import {
  Badge,
  Button,
  cn,
  Codicon,
  EmptyState,
  ErrorState,
  Loader,
  Progress,
  SearchField,
  SegmentedControl,
  useMutation,
  useQuery,
  useQueryClient
} from '@hermes/plugin-sdk'
import { useMemo, useState } from 'react'

import {
  ACHIEVEMENTS_KEY,
  achievementsRefetchInterval,
  fetchAchievements,
  rescanAchievements,
  scanFailure,
  scanIsActive
} from './api'
import { useAchievementsI18n } from './i18n'
import type { Achievement, AchievementsResponse } from './types'

type StateFilter = 'all' | 'in-progress' | 'secret' | 'unlocked'

const TIER_TONES: Record<string, string> = {
  Copper: 'text-orange-500',
  Silver: 'text-(--ui-text-secondary)',
  Gold: 'text-amber-500',
  Diamond: 'text-cyan-500',
  Olympian: 'text-violet-500'
}

function sectionId(category: string): string {
  return `achievements-${category.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

function ScanNotice({ data }: { data: AchievementsResponse }) {
  const k = useAchievementsI18n()
  const active = scanIsActive(data.scan_meta)
  const failure = scanFailure(data)
  const hasUsableSnapshot = data.achievements.length > 0 && data.scan_meta.mode !== 'pending'
  const mode = data.scan_meta.mode
  const current = data.scan_meta.sessions_scanned_so_far ?? data.scan_meta.sessions_total ?? 0
  const expected = data.scan_meta.sessions_expected_total ?? 0
  const determinate = expected > 0
  const progress = determinate ? Math.min(1, Math.max(0, current / expected)) : 0

  if (!active && !failure && !data.is_stale) {
    return null
  }

  const message = failure
    ? k.scanFailed
    : data.is_stale && hasUsableSnapshot
      ? k.stale
      : mode === 'pending'
        ? k.scanPending
        : k.scanRunning

  return (
    <section
      aria-live="polite"
      className={cn(
        'mx-4 flex shrink-0 items-start gap-2 border-y border-(--ui-stroke-tertiary) py-2 text-xs',
        failure ? 'text-destructive' : 'text-(--ui-text-secondary)'
      )}
      role={failure ? 'alert' : 'status'}
    >
      <Codicon className="mt-px shrink-0" name={failure ? 'warning' : active ? 'sync' : 'clock'} size="0.8rem" />
      <div className="min-w-0 flex-1">
        <p>{message}</p>
        {failure && <p className="mt-0.5 break-words text-[0.6875rem] opacity-85">{failure}</p>}
        {active && (
          <div className="mt-2 flex items-center gap-2">
            <Progress
              aria-label={k.scanRunning}
              animated
              className="flex-1"
              indeterminate={!determinate}
              size="sm"
              value={progress}
            />
            <span className="shrink-0 tabular-nums text-[0.6875rem] text-(--ui-text-tertiary)">
              {determinate ? k.scanProgress(current, expected) : k.scanProgressUnknown}
            </span>
          </div>
        )}
      </div>
    </section>
  )
}

function ProgressBar({ achievement }: { achievement: Achievement }) {
  const k = useAchievementsI18n()
  const value = achievement.unlocked && achievement.next_tier === null ? 1 : Math.min(1, Math.max(0, achievement.progress_pct / 100))
  const label = achievement.state === 'secret'
    ? k.hidden
    : achievement.unlocked && achievement.next_tier === null
      ? k.complete
      : k.progressValue(achievement.progress, achievement.next_threshold)

  return (
    <div className="grid gap-1.5">
      <Progress aria-label={`${achievement.name}: ${label}`} value={value} />
      <div className="flex items-center justify-between gap-2 text-[0.6875rem] text-(--ui-text-tertiary)">
        <span>{achievement.state === 'secret' ? k.hidden : achievement.next_tier ? k.nextTier(achievement.next_tier) : k.complete}</span>
        <span className="tabular-nums">{label}</span>
      </div>
    </div>
  )
}

function AchievementRow({ achievement }: { achievement: Achievement }) {
  const k = useAchievementsI18n()
  const tierTone = achievement.tier ? TIER_TONES[achievement.tier] : undefined
  const unlockedDate = achievement.unlocked_at
    ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(achievement.unlocked_at * 1000))
    : null
  const rowIcon = achievement.state === 'secret' ? 'lock' : achievement.unlocked ? 'star-full' : 'star-empty'

  return (
    <article className="grid gap-3 border-b border-(--ui-stroke-tertiary) py-4 last:border-b-0">
      <div className="flex items-start gap-3">
        <div
          aria-hidden
          className={cn(
            'grid size-9 shrink-0 place-items-center rounded-md bg-(--ui-bg-quaternary) text-(--ui-text-tertiary)',
            achievement.unlocked && 'text-(--ui-accent)'
          )}
        >
          <Codicon name={rowIcon} size="1rem" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <h3 className="text-sm font-medium text-foreground">{achievement.name}</h3>
            {achievement.unlocked && <Badge size="xs">{achievement.tier ?? k.unlockedBadge}</Badge>}
            {!achievement.unlocked && achievement.state === 'secret' && <Badge size="xs" variant="muted">{k.secret}</Badge>}
          </div>
          <p className="mt-1 text-xs leading-5 text-(--ui-text-secondary)">{achievement.description}</p>
        </div>
        {achievement.tier && (
          <span className={cn('shrink-0 text-[0.6875rem] font-semibold uppercase tracking-wide', tierTone)}>{achievement.tier}</span>
        )}
      </div>

      <ProgressBar achievement={achievement} />

      <div className="grid gap-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
        <p>{achievement.criteria}</p>
        {unlockedDate && <p>{k.unlockedOn(unlockedDate)}</p>}
        {achievement.evidence?.title && <p>{k.evidence(achievement.evidence.title)}</p>}
      </div>
    </article>
  )
}

function byCategory(achievements: Achievement[]): Array<[string, Achievement[]]> {
  const groups = new Map<string, Achievement[]>()

  for (const achievement of achievements) {
    const existing = groups.get(achievement.category)

    if (existing) {
      existing.push(achievement)
    } else {
      groups.set(achievement.category, [achievement])
    }
  }

  return [...groups.entries()]
}

export function AchievementsPage() {
  const k = useAchievementsI18n()
  const queryClient = useQueryClient()
  const [search, setSearch] = useState('')
  const [stateFilter, setStateFilter] = useState<StateFilter>('all')
  const query = useQuery({
    queryFn: fetchAchievements,
    queryKey: ACHIEVEMENTS_KEY,
    refetchInterval: achievementsRefetchInterval
  })
  const rescan = useMutation({
    mutationFn: rescanAchievements,
    onSuccess: data => queryClient.setQueryData(ACHIEVEMENTS_KEY, data)
  })
  const data = query.data
  const activeScan = scanIsActive(data?.scan_meta)
  const requestError = query.error ?? rescan.error
  const items = useMemo(() => {
    const needle = search.trim().toLowerCase()

    return (data?.achievements ?? []).filter(achievement => {
      const stateMatches =
        stateFilter === 'all' ||
        (stateFilter === 'unlocked' && achievement.unlocked) ||
        (stateFilter === 'secret' && achievement.state === 'secret') ||
        (stateFilter === 'in-progress' && !achievement.unlocked && achievement.state === 'discovered')
      const searchMatches = !needle || `${achievement.name} ${achievement.description} ${achievement.category}`.toLowerCase().includes(needle)

      return stateMatches && searchMatches
    })
  }, [data?.achievements, search, stateFilter])

  const filters: Array<{ id: StateFilter; label: string }> = [
    { id: 'all', label: k.all },
    { id: 'unlocked', label: k.unlocked },
    { id: 'in-progress', label: k.inProgress },
    { id: 'secret', label: k.secret }
  ]

  return (
    <div className="flex h-full flex-col overflow-hidden bg-(--ui-surface-background)">
      <header className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-base font-semibold text-foreground">{k.title}</h1>
          <p className="mt-0.5 text-xs text-(--ui-text-tertiary)">{k.subtitle}</p>
        </div>
        <Button disabled={rescan.isPending || activeScan} onClick={() => rescan.mutate()} size="sm" variant="outline">
          <Codicon name="refresh" size="0.8rem" />
          {rescan.isPending || activeScan ? k.scanning : k.scan}
        </Button>
      </header>

      {data && <ScanNotice data={data} />}

      {data && (
        <div className="flex shrink-0 flex-wrap items-center gap-2 px-4 py-3">
          <span className="text-xs font-medium text-foreground">{k.unlockedSummary(data.unlocked_count, data.total_count)}</span>
          <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{k.discoveredSummary(data.discovered_count)}</span>
          <span className="text-[0.6875rem] text-(--ui-text-tertiary)">{k.secretSummary(data.secret_count)}</span>
          <div className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-2">
            <div aria-label={k.stateFilter} role="group">
              <SegmentedControl onChange={setStateFilter} options={filters} value={stateFilter} />
            </div>
            {(data.achievements.length > 0 || search) && (
              <SearchField aria-label={k.searchLabel} onChange={setSearch} placeholder={k.search} value={search} />
            )}
          </div>
        </div>
      )}

      {requestError && !data ? (
        <div className="grid flex-1 place-items-center px-6">
          <ErrorState description={errorText(requestError, k.unknownError)} title={k.errorTitle}>
            <Button onClick={() => void query.refetch()} size="sm" variant="secondary">{k.retry}</Button>
          </ErrorState>
        </div>
      ) : !data ? (
        <div aria-label={k.loadingTitle} className="grid flex-1 place-items-center" role="status">
          <div className="grid justify-items-center gap-3 text-center">
            <Loader type="lemniscate-bloom" />
            <div>
              <p className="text-sm font-medium">{k.loadingTitle}</p>
              <p className="mt-1 text-xs text-(--ui-text-tertiary)">{k.loadingDescription}</p>
            </div>
          </div>
        </div>
      ) : data.achievements.length === 0 ? (
        <EmptyState className="flex-1" description={k.emptyDescription} title={k.emptyTitle} />
      ) : items.length === 0 ? (
        <EmptyState className="flex-1" description={k.noMatchDescription} title={k.noMatchTitle} />
      ) : (
        <div className="flex-1 overflow-y-auto px-4 pb-6">
          <div className="mx-auto grid w-full max-w-5xl gap-7">
            {byCategory(items).map(([category, achievements]) => {
              const headingId = sectionId(category)

              return (
                <section aria-labelledby={headingId} key={category}>
                  <div className="sticky top-0 z-10 flex items-center gap-2 bg-(--ui-surface-background) py-2">
                    <h2 className="text-xs font-semibold uppercase tracking-wide text-(--ui-text-secondary)" id={headingId}>
                      {category}
                    </h2>
                    <span className="text-[0.625rem] tabular-nums text-(--ui-text-quaternary)">{achievements.length}</span>
                  </div>
                  <div>{achievements.map(achievement => <AchievementRow achievement={achievement} key={achievement.id} />)}</div>
                </section>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
