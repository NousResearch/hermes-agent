import { compactNumber } from '@hermes/shared'

import { useI18n } from '@/i18n'
import { formatSourceUsd } from '@/lib/currency'
import { formatCostUsd } from '@/lib/statusbar'
import type { UsageStats } from '@/types/hermes'

interface SessionUsagePanelProps {
  usage: UsageStats
}

/** Presentational: the focused session's running totals, opened from the status
 *  bar's token / spend readout. Everything here is already streamed by the
 *  backend (`session.usage` ticks mid-turn, `message.complete` at turn end) —
 *  no extra RPC. Figures the owner omits (no cache reads yet, a route with no
 *  pricing entry) render as `—` rather than a fabricated 0 or $0.00. */
export function SessionUsagePanel({ usage }: SessionUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.sessionUsagePanel
  const missing = '—'
  const costKnown = typeof usage.cost_usd === 'number' && Number.isFinite(usage.cost_usd)

  const cacheHit =
    typeof usage.cache_hit_pct === 'number' && Number.isFinite(usage.cache_hit_pct)
      ? `${Math.round(usage.cache_hit_pct)}%`
      : missing

  const rows: { label: string; value: string }[] = [
    { label: copy.input, value: compactNumber(usage.input ?? 0) },
    { label: copy.output, value: compactNumber(usage.output ?? 0) },
    { label: copy.total, value: compactNumber(usage.total ?? 0) },
    { label: copy.cacheRead, value: usage.cache_read ? compactNumber(usage.cache_read) : missing },
    { label: copy.cacheHit, value: cacheHit },
    { label: copy.cacheWrite, value: usage.cache_write ? compactNumber(usage.cache_write) : missing },
    { label: copy.calls, value: String(usage.calls ?? 0) }
  ]

  const costValue =
    usage.cost_status === 'included'
      ? copy.costIncluded
      : costKnown
        ? `${usage.cost_status === 'estimated' ? '~' : ''}${formatCostUsd(usage.cost_usd as number)}`
        : copy.costUnknown

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="session-usage-panel">
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-foreground">{copy.title}</p>
        <span className="text-[0.6875rem] text-muted-foreground">{copy.subtitle}</span>
      </div>

      <ul className="flex flex-col gap-1.5">
        {rows.map(row => (
          <li className="flex items-center justify-between gap-2" key={row.label}>
            <span className="truncate text-muted-foreground">{row.label}</span>
            <span className="shrink-0 tabular-nums text-foreground">{row.value}</span>
          </li>
        ))}
      </ul>

      <div className="flex items-center justify-between gap-2 border-t border-(--ui-stroke-secondary) pt-2">
        <span className="text-muted-foreground">{copy.cost}</span>
        <span className="shrink-0 tabular-nums font-medium text-foreground">{costValue}</span>
      </div>

      {costKnown && usage.cost_status !== 'included' && (
        <p className="text-[0.6875rem] text-muted-foreground" data-testid="session-cost-usd">
          {formatSourceUsd(usage.cost_usd as number)}
        </p>
      )}

      {usage.cost_status === 'estimated' && (
        <p className="text-[0.6875rem] leading-snug text-muted-foreground">{copy.costNote}</p>
      )}
    </div>
  )
}
