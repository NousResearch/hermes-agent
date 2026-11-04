import { compactNumber } from '@hermes/shared'
import { useMemo } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

interface ContextUsagePanelProps {
  breakdown: ContextBreakdown | null
  loading: boolean
  usage: UsageStats
}

/** Presentational: the breakdown is fetched by the statusbar (see
 *  `useContextBreakdown`) because the gauge's own label needs it, so the
 *  popover opens with its numbers already in hand. `usage` is the gauge's
 *  merged figure — measured occupancy when the backend has it, the estimate
 *  otherwise — so the header and the bar can never disagree. */
export function ContextUsagePanel({ breakdown, loading, usage }: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const contextMax = usage.context_max ?? 0
  const contextUsed = usage.context_used ?? 0

  // A window with no occupancy reading yet (a model switch adopts the new window and zeroes the old
  // model's count) reads `—`: `gateway/session_usage.py::tui_usage` leaves the field out instead of
  // sending a measured zero, and the popover must not turn that into one.
  const contextUsedLabel =
    usage.context_used == null ? '—' : `${usage.context_estimated ? '~' : ''}${compactNumber(usage.context_used)}`

  const contextPercent =
    usage.context_percent == null ? null : Math.max(0, Math.min(100, Math.round(usage.context_percent)))

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
          {copy.tokenSummary(contextUsedLabel, compactNumber(contextMax))}
        </span>
      </div>

      <p className="flex items-center justify-between gap-2 text-[0.6875rem] text-foreground">
        <span>
          {usage.context_estimated ? '~' : ''}
          {contextPercent === null ? '—' : copy.percentFull(contextPercent)}
        </span>

        {usage.compressions !== undefined && (
          <span
            className="inline-flex items-center gap-1 text-muted-foreground"
            data-testid="context-panel-compressions"
          >
            <Codicon aria-hidden="true" name="layers" size="0.6875rem" />
            {t.shell.statusbar.compressions(usage.compressions)}
          </span>
        )}
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
    </div>
  )
}

/** The statusbar meter's trailing `[bar] N%`, plus a quiet `layers N` once the
 *  live session has compacted — the same glyph the sidebar puts on rows that
 *  came from a compression, so the two read as one idea. Zero stays out of the
 *  meter (every fresh session would carry noise); the panel shows it. */
export function ContextMeterDetail({ bar, compressions }: { bar: string; compressions?: number }) {
  if (!compressions) {
    return bar
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      {bar}
      <span className="inline-flex items-center gap-0.5 tabular-nums" data-testid="context-meter-compressions">
        <Codicon aria-hidden="true" name="layers" size="0.6875rem" />
        {compressions}
      </span>
    </span>
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
