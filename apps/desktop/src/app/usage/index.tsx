import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useMemo, useState } from 'react'

import { EmptyState } from '@/components/ui/empty-state'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { compactNumber } from '@/lib/format'
import { cn } from '@/lib/utils'

import type { GatewayRequester } from '../contrib/types'

/**
 * Local usage + cost surface. Reads the SAME aggregation `/insights` prints —
 * via the usage.overview JSON-RPC method — and renders it as a GitHub-style
 * intensity heatmap (green shades, theme-aware, tooltip on hover), a
 * cumulative-spend line, summary cards, and cost buckets. Pure renderer: no
 * metering lives here; the backend stays the single source of truth.
 */

type DailyBucket = {
  date: string
  sessions: number
  input_tokens: number
  output_tokens: number
  cache_read_tokens: number
  cache_write_tokens: number
  estimated_cost_usd: number
}

type CostBucket = {
  sessions: number
  cost_usd: number
  input_tokens: number
  output_tokens: number
  at_market_cost_usd?: number
}

type UsageReport = {
  empty: boolean
  days: number
  overview: {
    total_sessions: number
    total_input_tokens: number
    total_output_tokens: number
    total_cache_read_tokens: number
    total_tokens: number
    estimated_cost: number
    cost_buckets?: Record<'estimated' | 'included' | 'unknown', CostBucket>
    models_with_pricing?: string[]
    models_without_pricing?: string[]
  }
  daily_series?: DailyBucket[]
  models?: Array<{ model: string; cost: number; input_tokens: number; output_tokens: number }>
}

type ViewMode = 'daily' | 'weekly' | 'monthly'

function intensityClass(level: number): string {
  // level 0..4; GitHub-style: empty = faint, 4 = solid green.
  const i = Math.max(0, Math.min(4, Math.round(level)))

  return `usage-cell-${i}`
}

function fmtUsd(v: number): string {
  if (v >= 100) {return `$${v.toFixed(0)}`}

  if (v >= 1) {return `$${v.toFixed(2)}`}

  if (v >= 0.01) {return `$${v.toFixed(3)}`}

  return `$${v.toFixed(5)}`
}

function SummaryCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="min-w-0 rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
      <div className="truncate text-[0.68rem] text-(--ui-text-tertiary)">{label}</div>
      <div
        className={cn(
          'mt-1 truncate font-mono text-base font-semibold tabular-nums',
          accent ? 'text-(--ui-green)' : 'text-foreground'
        )}
      >
        {value}
      </div>
    </div>
  )
}

function HeatmapCell({ day, maxCost }: { day: DailyBucket; maxCost: number }) {
  const level = maxCost > 0 ? day.estimated_cost_usd / maxCost : 0
  const cls = level === 0 && day.sessions === 0 ? 'usage-cell-0' : intensityClass(level * 4)
  const tokens = day.input_tokens + day.output_tokens
  const tip = `${day.date} · ${day.sessions} session${day.sessions === 1 ? '' : 's'} · ${compactNumber(tokens)} tok · ${fmtUsd(day.estimated_cost_usd)}`

  return (
    <rect className={cls} height="11" rx="2" width="11">
      <title>{tip}</title>
    </rect>
  )
}

function Heatmap({ series, mode }: { series: DailyBucket[]; mode: ViewMode }) {
  const maxCost = useMemo(() => Math.max(0, ...series.map(d => d.estimated_cost_usd)), [series])

  if (mode === 'daily') {
    // GitHub layout: columns = weeks, rows = Mon..Sun. Buckets are
    // chronological; anchor to the last day (today) and walk backwards.
    const cols: DailyBucket[][] = []
    let col: DailyBucket[] = []

    for (let i = series.length - 1; i >= 0; i--) {
      col.unshift(series[i])
      const dow = new Date(series[i].date + 'T00:00:00').getDay() // 0=Sun

      if (dow === 0 || i === 0) {
        cols.unshift(col)
        col = []
      }
    }

    const colCount = cols.length

    return (
      <div className="overflow-x-auto">
        <svg aria-label="Daily usage heatmap" height={13 * 7 + 8} role="img" width={Math.max(colCount * 14, 200)}>
          {cols.map((c, ci) => (
            <g key={ci} transform={`translate(${ci * 14}, 0)`}>
              {c.map((day, ri) => (
                <g key={day.date} transform={`translate(0, ${ri * 13})`}>
                  <HeatmapCell day={day} maxCost={maxCost} />
                </g>
              ))}
            </g>
          ))}
        </svg>
      </div>
    )
  }

  // weekly / monthly: aggregate buckets into bars.
  const groups: Array<{ label: string; cost: number; tokens: number; sessions: number }> = []
  const seen = new Map<string, number>()

  for (const d of series) {
    const dt = new Date(d.date + 'T00:00:00')

    const key =
      mode === 'weekly'
        ? (() => {
            const wk = new Date(dt)
            wk.setDate(wk.getDate() - wk.getDay())

            return wk.toISOString().slice(0, 10)
          })()
        : d.date.slice(0, 7)

    let gi = seen.get(key)

    if (gi === undefined) {
      gi = groups.length
      seen.set(key, gi)
      groups.push({ label: key, cost: 0, tokens: 0, sessions: 0 })
    }

    const g = groups[gi]
    g.cost += d.estimated_cost_usd
    g.tokens += d.input_tokens + d.output_tokens
    g.sessions += d.sessions
  }

  const gMax = Math.max(0, ...groups.map(g => g.cost))
  const W = 560
  const H = 120
  const barW = Math.max(4, Math.min(28, (W - groups.length) / groups.length))

  return (
    <div className="overflow-x-auto">
      <svg aria-label={`${mode} usage chart`} height={H + 24} role="img" width={Math.max(W, groups.length * (barW + 4))}>
        {groups.map((g, i) => {
          const h = gMax > 0 ? Math.max(2, (g.cost / gMax) * (H - 20)) : 2
          const tip = `${g.label} · ${g.sessions} sessions · ${compactNumber(g.tokens)} tok · ${fmtUsd(g.cost)}`

          return (
            <g key={g.label}>
              <rect className="usage-cell-4" height={h} rx="2" width={barW} x={i * (barW + 4)} y={H - h}>
                <title>{tip}</title>
              </rect>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function SpendLine({ series }: { series: DailyBucket[] }) {
  const points = useMemo(() => {
    const W = 560
    const H = 120
    let cum = 0
    const max = Math.max(0, ...series.map(d => (cum += d.estimated_cost_usd)))
    cum = 0

    const pts = series.map((d, i) => {
      cum += d.estimated_cost_usd
      const x = (i / Math.max(1, series.length - 1)) * W
      const y = max > 0 ? H - (cum / max) * (H - 12) : H

      return `${x.toFixed(1)},${y.toFixed(1)}`
    })

    return { pts: pts.join(' '), total: cum }
  }, [series])

  return (
    <div>
      <svg aria-label="Cumulative spend" className="h-32 w-full" role="img" viewBox="0 0 560 120">
        <polyline
          fill="none"
          points={points.pts}
          stroke="var(--ui-green)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
        />
      </svg>
      <div className="mt-1 text-[0.68rem] text-(--ui-text-tertiary)">
        Cumulative estimated spend across {series.length} days:{' '}
        <span className="font-mono text-(--ui-green)">{fmtUsd(points.total)}</span>
      </div>
    </div>
  )
}

function CostBuckets({ buckets }: { buckets: NonNullable<UsageReport['overview']['cost_buckets']> }) {
  const rows: Array<{ name: string; bucket: CostBucket; note?: string }> = [
    { name: 'estimated', bucket: buckets.estimated, note: 'market-rate estimate' },
    { name: 'included', bucket: buckets.included, note: 'subscription-included' },
    { name: 'unknown', bucket: buckets.unknown, note: 'no pricing signal' }
  ]

  return (
    <div className="grid grid-cols-3 gap-3">
      {rows.map(r => (
        <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3" key={r.name}>
          <div className="text-[0.68rem] text-(--ui-text-tertiary)">{r.name}</div>
          <div className="mt-1 font-mono text-sm font-semibold tabular-nums">{r.bucket.sessions} sessions</div>
          <div className="font-mono text-[0.7rem] text-(--ui-text-tertiary)">{fmtUsd(r.bucket.cost_usd)}</div>
          {typeof r.bucket.at_market_cost_usd === 'number' && r.bucket.at_market_cost_usd > 0 && (
            <div className="mt-0.5 font-mono text-[0.65rem] text-(--ui-text-tertiary)">
              at market: {fmtUsd(r.bucket.at_market_cost_usd)}
            </div>
          )}
          <div className="mt-0.5 text-[0.62rem] text-(--ui-text-quaternary)">{r.note}</div>
        </div>
      ))}
    </div>
  )
}

const VIEWS: Array<{ id: ViewMode; label: string }> = [
  { id: 'daily', label: 'Daily' },
  { id: 'weekly', label: 'Weekly' },
  { id: 'monthly', label: 'Monthly' }
]

export function UsageView({ requestGateway }: { requestGateway: GatewayRequester }) {
  const [view, setView] = useState<ViewMode>('daily')
  const [days, setDays] = useState(90)

  const { data, isLoading, isError } = useQuery({
    queryKey: ['usage', 'overview', days],
    queryFn: () => requestGateway<UsageReport>('usage.overview', { days })
  })

  const o = data?.overview
  const series = data?.daily_series ?? []

  const header = (
    <div className="flex items-center gap-2">
      {VIEWS.map(v => (
        <button
          className={cn(
            'rounded-md px-2 py-0.5 text-[0.68rem] transition-colors',
            view === v.id
              ? 'bg-(--ui-accent) text-(--ui-accent-foreground)'
              : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
          )}
          key={v.id}
          onClick={() => setView(v.id)}
          type="button"
        >
          {v.label}
        </button>
      ))}
    </div>
  )

  const windowSelect = (
    <select
      className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) px-1.5 py-0.5 text-[0.68rem] text-(--ui-text-secondary)"
      onChange={e => setDays(Number(e.target.value))}
      value={days}
    >
      {[7, 30, 90, 365].map(n => (
        <option key={n} value={n}>
          {n}d
        </option>
      ))}
    </select>
  )

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden">
      <UsageHeatmapStyles />
      <header className="flex shrink-0 items-start justify-between gap-3 px-5 pt-5 pb-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">Usage</h2>
          <p className="truncate text-xs text-muted-foreground/80">Local token &amp; cost metering</p>
        </div>
        <div className="flex shrink-0 items-center gap-1.5">{windowSelect}</div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto px-5 pb-5">
        {isLoading && (
          <div className="flex h-full items-center justify-center">
            <GlyphSpinner ariaLabel="Loading usage" />
          </div>
        )}
        {isError && (
          <div className="flex h-full items-center justify-center">
            <div className="text-sm text-(--ui-text-tertiary)">Could not load usage data. Is the gateway connected?</div>
          </div>
        )}
        {!isLoading && !isError && data?.empty && (
          <div className="flex h-full items-center justify-center">
            <EmptyState description="Sessions will appear here as you use Hermes." title="No usage yet" />
          </div>
        )}
        {!isLoading && !isError && !data?.empty && o && (
          <div className="flex h-full flex-col gap-4 overflow-auto p-1">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <SummaryCard label="Sessions" value={compactNumber(o.total_sessions)} />
              <SummaryCard label="Total tokens" value={compactNumber(o.total_tokens)} />
              <SummaryCard accent label="Est. spend" value={fmtUsd(o.estimated_cost)} />
              <SummaryCard label="Cache reads" value={compactNumber(o.total_cache_read_tokens)} />
            </div>

            {o.cost_buckets && <CostBuckets buckets={o.cost_buckets} />}

            <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
              <div className="mb-2 text-[0.68rem] text-(--ui-text-tertiary)">{view} intensity</div>
              {header}
              <div className="mt-3">
                <Heatmap mode={view} series={series} />
              </div>
            </div>

            <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
              <div className="mb-2 text-[0.68rem] text-(--ui-text-tertiary)">cumulative spend</div>
              <SpendLine series={series} />
            </div>

            {data.models && data.models.length > 0 && (
              <div className="rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-quaternary) p-3">
                <div className="mb-2 text-[0.68rem] text-(--ui-text-tertiary)">by model</div>
                <div className="flex flex-col gap-1">
                  {data.models.slice(0, 8).map(m => (
                    <div className="flex items-center justify-between gap-3 text-xs" key={m.model}>
                      <span className="truncate font-mono text-(--ui-text-secondary)">{m.model}</span>
                      <span className="shrink-0 font-mono tabular-nums text-(--ui-text-tertiary)">
                        {compactNumber(m.input_tokens + m.output_tokens)} tok · {fmtUsd(m.cost)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

/**
 * The heatmap's green scale. GitHub's graph uses 5 shades; we derive ours from
 * the theme's green (opacity ladder) so it reskins with every theme. Defined
 * once here; the cell classes are used by the SVG above.
 */
export function UsageHeatmapStyles(): ReactNode {
  return (
    <style>{`
      .usage-cell-0 { fill: color-mix(in srgb, var(--ui-green) 8%, transparent); }
      .usage-cell-1 { fill: color-mix(in srgb, var(--ui-green) 28%, transparent); }
      .usage-cell-2 { fill: color-mix(in srgb, var(--ui-green) 52%, transparent); }
      .usage-cell-3 { fill: color-mix(in srgb, var(--ui-green) 76%, transparent); }
      .usage-cell-4 { fill: var(--ui-green); }
    `}</style>
  )
}
