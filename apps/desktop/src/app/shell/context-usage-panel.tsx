import { useEffect, useMemo, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

interface ContextUsagePanelProps {
  currentUsage: UsageStats
  onUsageSnapshot?: (usage: Pick<UsageStats, 'context_max' | 'context_percent' | 'context_used'>) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  sessionId: string | null
}

export function ContextUsagePanel({
  currentUsage,
  onUsageSnapshot,
  requestGateway,
  sessionId
}: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const [breakdown, setBreakdown] = useState<ContextBreakdown | null>(null)
  const [loading, setLoading] = useState(false)
  const onUsageSnapshotRef = useRef(onUsageSnapshot)
  onUsageSnapshotRef.current = onUsageSnapshot

  useEffect(() => {
    if (!sessionId) {
      setBreakdown(null)
      setLoading(false)

      return
    }

    let cancelled = false
    setLoading(true)

    void requestGateway<ContextBreakdown>('session.context_breakdown', { session_id: sessionId })
      .then(data => {
        if (!cancelled) {
          setBreakdown(data)
          onUsageSnapshotRef.current?.({
            context_max: data.context_max,
            context_percent: data.context_percent,
            context_used: data.context_used
          })
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBreakdown(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [requestGateway, sessionId])

  const contextMax = breakdown?.context_max ?? currentUsage.context_max ?? 0
  const contextUsed = breakdown?.context_used ?? currentUsage.context_used ?? 0

  const contextPercent = Math.max(
    0,
    Math.min(100, Math.round(breakdown?.context_percent ?? currentUsage.context_percent ?? 0))
  )

  const categories = useMemo(
    () =>
      (breakdown?.categories ?? []).map(category => ({
        ...category,
        label: copy.categories[category.id as keyof typeof copy.categories] ?? category.label
      })),
    [breakdown?.categories, copy]
  )

  const categoryTotal = categories.reduce((sum, category) => sum + category.tokens, 0)

  // Two ways to read the same breakdown, toggled by clicking the bar:
  //  - 'window' (default): segments are scaled against the whole context window,
  //    so the filled portion matches the "N% Full" figure above and the
  //    remaining track is visibly free space.
  //  - 'used': segments are scaled against used tokens only, filling the bar,
  //    which makes the relative split between small categories easier to read.
  // Scaling only against the used total is what made the bar look 100% full at
  // 26% usage — free space was never drawn.
  const [barScale, setBarScale] = useState<'used' | 'window'>('window')
  const [showPercent, setShowPercent] = useState(false)

  const barTotal =
    barScale === 'window'
      ? Math.max(contextMax, categoryTotal, 1)
      : categoryTotal || contextUsed || 1

  // Percentages in the list follow the same denominator as the bar, so the two
  // readings can never disagree.
  const listTotal = barTotal

  const formatPercent = (tokens: number) => {
    const pct = (tokens / listTotal) * 100

    return pct > 0 && pct < 0.1 ? '<0.1%' : `${pct.toFixed(1)}%`
  }

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-foreground">{copy.title}</p>

        <span className="text-[0.6875rem] text-muted-foreground">
          {copy.tokenSummary(`~${compactNumber(contextUsed)}`, compactNumber(contextMax))}
        </span>
      </div>

      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[0.6875rem] text-foreground">{copy.percentFull(contextPercent)}</p>

        <button
          className="text-[0.6875rem] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
          onClick={() => setShowPercent(value => !value)}
          type="button"
        >
          {showPercent ? copy.showTokens : copy.showPercentages}
        </button>
      </div>

      <button
        aria-label={barScale === 'window' ? copy.scaleToUsed : copy.scaleToWindow}
        className="cursor-pointer"
        onClick={() => setBarScale(value => (value === 'window' ? 'used' : 'window'))}
        title={barScale === 'window' ? copy.scaleToUsed : copy.scaleToWindow}
        type="button"
      >
        <ContextUsageBar categories={categories} segmentTotal={barTotal} />
      </button>

      <ul className="flex flex-col gap-1.5">
        {categories.map(category => (
          <li className="flex items-center justify-between gap-2" key={category.id}>
            <span className="flex min-w-0 items-center gap-2">
              <span className="size-2 shrink-0 rounded-[2px]" style={{ background: category.color }} />

              <span className="truncate text-muted-foreground">{category.label}</span>
            </span>

            <span className="shrink-0 tabular-nums text-foreground">
              {showPercent ? formatPercent(category.tokens) : compactNumber(category.tokens)}
            </span>
          </li>
        ))}
      </ul>

      {loading && <p className="text-[0.6875rem] text-muted-foreground">{copy.loading}</p>}

      {!loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.empty}</p>}
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
