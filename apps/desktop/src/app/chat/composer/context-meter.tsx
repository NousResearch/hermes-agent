import { useStore } from '@nanostores/react'
import { useCallback, useMemo } from 'react'

import { ContextUsagePanel } from '@/app/shell/context-usage-panel'
import { useContextBreakdown } from '@/app/shell/hooks/use-context-breakdown'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useI18n } from '@/i18n'
import { requestGatewayForProfile } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $busy, $currentUsage } from '@/store/session'
import type { UsageStats } from '@/types/hermes'

/** A window-shaped fill, not a ring: the track is the model window, the
 *  colored part is how full it is, and the colors are the same categories
 *  the detail lists. Clicking opens that list. */
export function ComposerContextMeter() {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const usage = useStore($currentUsage)
  const sessionId = useStore($activeSessionId)
  const busy = useStore($busy)
  const profile = useStore($activeGatewayProfile)

  const requestGateway = useCallback(
    <T,>(method: string, params?: Record<string, unknown>) => requestGatewayForProfile<T>(profile, method, params),
    [profile]
  )

  const { breakdown, loading } = useContextBreakdown({
    busy,
    enabled: true,
    requestGateway,
    sessionId
  })

  const gauge = useMemo<UsageStats>(
    () =>
      breakdown
        ? {
            ...usage,
            context_estimated: breakdown.context_estimated,
            context_max: breakdown.context_max,
            context_percent: breakdown.context_percent,
            context_source: breakdown.context_source,
            context_used: breakdown.context_used
          }
        : usage,
    [breakdown, usage]
  )

  const hasWindow = (gauge.context_max ?? 0) > 0
  const percent = hasWindow ? Math.max(0, Math.min(100, Math.round(gauge.context_percent ?? 0))) : null
  const categories = breakdown?.categories ?? []

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          aria-label={copy.title}
          className="inline-flex h-(--composer-control-size) shrink-0 items-center gap-1.5 rounded-full px-2 text-[0.6875rem] text-muted-foreground hover:bg-accent hover:text-foreground"
          type="button"
        >
          <ContextFill categories={categories} percent={percent} />
          <span className="tabular-nums">
            {percent == null ? '—' : `${gauge.context_estimated ? '~' : ''}${percent}%`}
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto p-0" side="top">
        <ContextUsagePanel breakdown={breakdown} loading={loading} usage={gauge} />
      </PopoverContent>
    </Popover>
  )
}

function ContextFill({
  categories,
  percent
}: {
  categories: readonly { color: string; id: string; tokens: number }[]
  percent: number | null
}) {
  const bounded = percent == null ? 0 : percent
  const total = categories.reduce((sum, category) => sum + category.tokens, 0)

  return (
    <span aria-hidden="true" className="relative h-1.5 w-14 overflow-hidden rounded-full bg-(--ui-stroke-tertiary)">
      <span className="absolute inset-y-0 left-0 flex overflow-hidden" style={{ width: `${bounded}%` }}>
        {total > 0 ? (
          categories.map(category => (
            <span
              className="h-full min-w-px"
              key={category.id}
              style={{ background: category.color, width: `${(category.tokens / total) * 100}%` }}
            />
          ))
        ) : (
          <span className="h-full w-full bg-foreground/70" />
        )}
      </span>
    </span>
  )
}
