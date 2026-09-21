import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useViewedInterval } from '@/hooks/use-viewed-interval'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { formatDuration } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import { $turnBreakdownBySession, type TurnBreakdown } from '@/store/turn-breakdown'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

interface ContextUsagePanelProps {
  breakdown: ContextBreakdown | null
  loading: boolean
  /** The session whose turn clock the breakdown section reads. */
  sessionId?: null | string
  usage: UsageStats
}

/** Presentational: the breakdown is fetched by the statusbar (see
 *  `useContextBreakdown`) because the gauge's own label needs it, so the
 *  popover opens with its numbers already in hand. `usage` is the gauge's
 *  merged figure — measured occupancy when the backend has it, the estimate
 *  otherwise — so the header and the bar can never disagree. */
export function ContextUsagePanel({ breakdown, loading, sessionId, usage }: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const turnCopy = t.shell.statusbar.turnBreakdown
  const contextMax = usage.context_max ?? 0
  const contextUsed = usage.context_used ?? 0
  const contextPercent = Math.max(0, Math.min(100, Math.round(usage.context_percent ?? 0)))

  const categories = useMemo(
    () =>
      (breakdown?.categories ?? []).map(category => ({
        ...category,
        label: copy.categories[category.id as keyof typeof copy.categories] ?? category.label
      })),
    [breakdown?.categories, copy]
  )

  const segmentTotal = categories.reduce((sum, category) => sum + category.tokens, 0) || contextUsed || 1

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-foreground">{copy.title}</p>

        <span className="text-[0.6875rem] text-muted-foreground">
          {copy.tokenSummary(
            `${usage.context_estimated ? '~' : ''}${compactNumber(contextUsed)}`,
            compactNumber(contextMax)
          )}
        </span>
      </div>

      <p className="text-[0.6875rem] text-foreground">
        {usage.context_estimated ? '~' : ''}
        {copy.percentFull(contextPercent)}
      </p>

      <ContextUsageBar categories={categories} segmentTotal={segmentTotal} />

      <ul className="flex flex-col gap-1.5">
        {categories.map(category => (
          <li className="flex items-center justify-between gap-2" key={category.id}>
            <span className="flex min-w-0 items-center gap-2">
              <span className="size-2 shrink-0 rounded-[2px]" style={{ background: category.color }} />

              <span className="truncate text-muted-foreground">{category.label}</span>
            </span>

            <span className="shrink-0 tabular-nums text-foreground">~{compactNumber(category.tokens)}</span>
          </li>
        ))}
      </ul>

      {loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.loading}</p>}

      {!loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.empty}</p>}

      <TurnBreakdownSection copy={turnCopy} sessionId={sessionId} />
    </div>
  )
}

/** Per-turn wall-clock breakdown (issue #117224). Wall time and tool time come
 *  from the per-session clock store fed by the message stream; model time is
 *  the remainder, prefixed with `~` because the windows overlap (a tool can
 *  run while the model streams) — same honesty marker the CLI uses for
 *  estimated context figures. Hidden entirely until a turn has run. */
function TurnBreakdownSection({
  copy,
  sessionId
}: {
  copy: Translations['shell']['statusbar']['turnBreakdown']
  sessionId?: null | string
}) {
  const [now, setNow] = useState(() => Date.now())
  const breakdownMap = useStore($turnBreakdownBySession)
  const breakdown: TurnBreakdown | undefined = sessionId ? breakdownMap[sessionId] : undefined
  const running = Boolean(breakdown?.startedAt && !breakdown.completedAt)

  // Tick once a second only while a turn is live, so the wall figure counts up
  // in place; a frozen turn renders a static value.
  useViewedInterval(() => setNow(Date.now()), 1000, running)

  if (!breakdown?.startedAt) {
    return null
  }

  const endedAt = breakdown.completedAt ?? now
  const wallSeconds = Math.max(0, Math.floor((endedAt - breakdown.startedAt) / 1000))
  const toolSeconds = Math.floor(breakdown.toolSeconds)
  const modelSeconds = Math.max(0, wallSeconds - toolSeconds)

  const entries: ReadonlyArray<{ label: string; value: string }> = [
    { label: copy.wall, value: formatDuration((endedAt - breakdown.startedAt)) },
    { label: copy.tools, value: `${toolSeconds}s` },
    { label: copy.model, value: `~${formatDuration(modelSeconds * 1000)}` }
  ]

  return (
    <div className="flex flex-col gap-1.5 border-t border-(--ui-stroke-secondary) pt-2" data-slot="turn-breakdown">
      <p className="font-medium text-foreground">
        {copy.title}
        {running ? ` · ${copy.running}` : ''}
      </p>

      <ul className="flex flex-col gap-1">
        {entries.map(entry => (
          <li className="flex items-center justify-between gap-2" key={entry.label}>
            <span className="text-muted-foreground">{entry.label}</span>

            <span className="tabular-nums text-foreground">{entry.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function ContextUsageBar({
  categories,
  segmentTotal
}: {
  categories: readonly ContextUsageCategory[]
  segmentTotal: number
}) {
  return (
    <div
      className={cn(
        'flex h-1.5 overflow-hidden rounded-full',
        categories.length ? 'bg-(--ui-stroke-tertiary)' : 'dither bg-(--ui-bg-elevated)'
      )}
      data-slot="context-usage-bar"
    >
      {categories.map(category => (
        <span
          className="h-full min-w-px"
          key={category.id}
          style={{
            background: category.color,
            width: `${(category.tokens / segmentTotal) * 100}%`
          }}
        />
      ))}
    </div>
  )
}
