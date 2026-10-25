/**
 * Achievements page — score header, filter tabs, and the achievement card
 * grid. Read-only: the only mutation is the Rescan button, which forces the
 * backend to re-evaluate session history.
 */

import {
  Badge,
  Button,
  cn,
  Codicon,
  EmptyState,
  ErrorState,
  host,
  relativeTime,
  Skeleton,
  useQuery
} from '@hermes/plugin-sdk'
import { useState } from 'react'

import { achievementsKey, fetchAchievements, rescanAchievements } from './api'
import { useAchievementsI18n } from './i18n'
import { type Achievement, type AchievementFilter, TIER_ORDER } from './types'

const FILTERS: AchievementFilter[] = ['all', 'unlocked', 'discovered', 'secret']

function tierIndex(tier: string | null): number {
  return tier ? TIER_ORDER.indexOf(tier as (typeof TIER_ORDER)[number]) : -1
}

function tierBadgeClass(tier: string | null): string {
  // Tier is conveyed by label + a subtle theme-safe accent, never hardcoded
  // colors. Higher tiers get a stronger visual weight via emphasis.
  const i = tierIndex(tier)

  if (i < 0) {return 'text-(--ui-text-quaternary)'}

  if (i >= 4) {return 'text-(--ui-accent) font-semibold'}

  if (i >= 3) {return 'text-(--ui-accent)'}

  if (i >= 2) {return 'text-(--ui-text-primary)'}

  return 'text-(--ui-text-secondary)'
}

function ScoreHeader({
  data,
  onRescan,
  rescinding
}: {
  data: { unlocked_count: number; discovered_count: number; secret_count: number; total_count: number; generated_at: number; is_stale: boolean }
  onRescan: () => void
  rescinding: boolean
}): React.JSX.Element {
  const k = useAchievementsI18n()
  const pct = data.total_count ? Math.round((data.unlocked_count / data.total_count) * 100) : 0

  return (
    <div className="border-b border-(--ui-stroke-secondary) px-6 py-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl font-semibold tabular-nums">
              {data.unlocked_count}/{data.total_count}
            </span>
            <span className="text-sm text-(--ui-text-secondary)">
              {k.scoreUnlocked} · {pct}%
            </span>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs text-(--ui-text-tertiary)">
            <span>{data.discovered_count} {k.discovered}</span>
            <span>{data.secret_count} {k.secret}</span>
            {data.generated_at > 0 && <span>{k.scanned(relativeTime(data.generated_at * 1000))}</span>}
            {data.is_stale && <Badge variant="warn">{k.stale}</Badge>}
          </div>
        </div>
        <Button disabled={rescinding} onClick={onRescan} size="sm" variant="secondary">
          {rescinding ? k.scanning : k.rescan}
        </Button>
      </div>
      <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-(--ui-bg-quaternary)">
        <div
          className="h-full rounded-full bg-(--ui-accent) transition-all"
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
    </div>
  )
}

function AchievementCard({ item }: { item: Achievement }): React.JSX.Element {
  const k = useAchievementsI18n()
  const [open, setOpen] = useState(false)
  const isSecret = item.state === 'secret'
  const pct = item.progress_pct ?? 0

  return (
    <div
      className={cn(
        'flex flex-col rounded-lg border p-4',
        item.unlocked
          ? 'border-(--ui-stroke-strong) bg-(--ui-bg-tertiary)'
          : 'border-(--ui-stroke-secondary) bg-(--ui-bg-secondary)',
        isSecret && 'opacity-70'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <Codicon
            className={cn('shrink-0', item.unlocked ? 'text-(--ui-accent)' : 'text-(--ui-text-tertiary)')}
            name="milestone"
          />
          <span className="truncate text-sm font-medium">{isSecret ? k.secretName : item.name}</span>
        </div>
        {item.tier ? (
          <Badge className={cn('shrink-0 text-[0.6875rem]', tierBadgeClass(item.tier))} variant="outline">
            {item.tier}
          </Badge>
        ) : item.unlocked ? (
          <Badge className="shrink-0 text-[0.6875rem] text-(--ui-accent)" variant="outline">
            Earned
          </Badge>
        ) : null}
      </div>
      <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-(--ui-text-tertiary)">
        {isSecret ? k.secretDescription : item.description}
      </p>
      <div className="mt-3">
        <div className="flex items-center justify-between text-[0.6875rem] text-(--ui-text-tertiary)">
          <span>
            {item.unlocked
              ? item.next_tier
                ? k.nextTier(item.next_tier, item.next_threshold)
                : k.maxTier
              : item.next_tier
                ? k.nextTier(item.next_tier, item.next_threshold)
                : ''}
          </span>
          {!isSecret && <span className="tabular-nums">{pct}%</span>}
        </div>
        <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-(--ui-bg-quaternary)">
          <div
            className={cn('h-full rounded-full', item.unlocked ? 'bg-(--ui-accent)' : 'bg-(--ui-text-tertiary)')}
            style={{ width: `${isSecret ? 0 : Math.min(100, pct)}%` }}
          />
        </div>
      </div>
      {item.criteria && (
        <div className="mt-3">
          <button
            className="text-[0.6875rem] text-(--ui-text-tertiary) underline decoration-dotted underline-offset-2 hover:text-(--ui-text-primary)"
            onClick={() => setOpen(o => !o)}
            type="button"
          >
            {open ? k.hideWhatCounts : k.whatCounts}
          </button>
          {open && <p className="mt-1.5 text-[0.6875rem] leading-relaxed text-(--ui-text-tertiary)">{item.criteria}</p>}
        </div>
      )}
      {item.evidence?.title && (
        <p className="mt-2 truncate text-[0.6875rem] text-(--ui-text-quaternary)">{k.evidenceFrom(item.evidence.title)}</p>
      )}
    </div>
  )
}

export function AchievementsPage(): React.JSX.Element {
  const k = useAchievementsI18n()
  const [filter, setFilter] = useState<AchievementFilter>('all')
  const [rescinding, setRescinding] = useState(false)

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: achievementsKey(),
    queryFn: fetchAchievements,
    refetchInterval: 120_000
  })

  const rescan = async (): Promise<void> => {
    setRescinding(true)

    try {
      await rescanAchievements()
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      host.notify({ kind: 'error', message: `Achievements rescan failed: ${message}` })
    } finally {
      setRescinding(false)
    }
  }

  if (isLoading) {
    return (
      <div className="grid h-full grid-cols-1 gap-4 overflow-y-auto p-6 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 9 }, (_, i) => (
          <Skeleton className="h-40 w-full rounded-lg" key={i} />
        ))}
      </div>
    )
  }

  if (isError || !data) {
    return (
      <ErrorState
        description={`${error?.message ?? 'Unknown error'} — ${k.loadFailedHint}`}
        title={k.loadFailed}
      >
        <Button onClick={() => refetch()} variant="secondary">
          {k.retry}
        </Button>
      </ErrorState>
    )
  }

  const items = data.achievements ?? []
  const shown = items.filter(a => filter === 'all' || a.state === filter)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ScoreHeader data={data} onRescan={rescan} rescinding={rescinding} />
      <div className="flex items-center gap-1 border-b border-(--ui-stroke-secondary) px-6 py-2">
        {FILTERS.map(f => {
          const count =
            f === 'all'
              ? data.total_count
              : f === 'unlocked'
                ? data.unlocked_count
                : f === 'discovered'
                  ? data.discovered_count
                  : data.secret_count

          const label =
            f === 'all' ? k.filterAll : f === 'unlocked' ? k.filterUnlocked : f === 'discovered' ? k.filterDiscovered : k.filterSecret

          return (
            <button
              className={cn(
                'rounded-md px-2.5 py-1 text-xs capitalize transition-colors',
                filter === f
                  ? 'bg-(--ui-bg-quaternary) text-(--ui-text-primary)'
                  : 'text-(--ui-text-tertiary) hover:text-(--ui-text-primary)'
              )}
              key={f}
              onClick={() => setFilter(f)}
              type="button"
            >
              {label} ({count})
            </button>
          )
        })}
      </div>
      {shown.length === 0 ? (
        <EmptyState description={k.emptyBody} title={k.emptyTitle} />
      ) : (
        <div className="grid flex-1 auto-rows-min grid-cols-1 gap-4 overflow-y-auto p-6 sm:grid-cols-2 lg:grid-cols-3">
          {shown.map(a => (
            <AchievementCard item={a} key={a.id} />
          ))}
        </div>
      )}
    </div>
  )
}
