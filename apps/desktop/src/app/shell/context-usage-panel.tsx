import { useEffect, useMemo, useState } from 'react'

import { useI18n } from '@/i18n'
import { formatK } from '@/lib/statusbar'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

interface ContextUsagePanelProps {
  currentUsage: UsageStats
  onUsageSnapshot?: (usage: Partial<UsageStats>) => void
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

          onUsageSnapshot?.({
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
  }, [onUsageSnapshot, requestGateway, sessionId])

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

  const segmentTotal = categories.reduce((sum, category) => sum + category.tokens, 0) || contextUsed || 1

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-foreground">{copy.title}</p>

        <span className="text-[0.6875rem] text-muted-foreground">
          {copy.tokenSummary(`~${formatK(contextUsed)}`, formatK(contextMax))}
        </span>
      </div>

      <p className="text-[0.6875rem] text-foreground">{copy.percentFull(contextPercent)}</p>

      <ContextUsageBar categories={categories} segmentTotal={segmentTotal} />

      <ul className="flex flex-col gap-1.5">
        {categories.map(category => (
          <li className="flex items-center justify-between gap-2" key={category.id}>
            <span className="flex min-w-0 items-center gap-2">
              <span className="size-2 shrink-0 rounded-[2px]" style={{ background: category.color }} />

              <span className="truncate text-muted-foreground">{category.label}</span>
            </span>

            <span className="shrink-0 tabular-nums text-foreground">{formatCategoryTokens(category.tokens)}</span>
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

function formatCategoryTokens(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return '0'
  }

  if (value >= 1_000) {
    return `${formatK(value)}`
  }

  return value.toLocaleString()
}
