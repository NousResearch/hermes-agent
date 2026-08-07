import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { EmptyState } from '@/components/ui/empty-state'
import { Loader } from '@/components/ui/loader'
import { Badge } from '@/components/ui/badge'
import { Button, buttonVariants } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { getUsageAnalytics } from '@/hermes'
import { useI18n } from '@/i18n'
import { compactNumber } from '@/lib/format'
import { cn } from '@/lib/utils'
import { $activeGatewayProfile } from '@/store/profile'

import type { AnalyticsResponse } from '@/types/hermes'

import { BarSparkline } from './sparkline'
import { $usageDays, setUsageDays } from './store'

const USAGE_QUERY_KEY = ['usage-summary'] as const

const DAY_PRESETS = [
  { label: '7d', value: 7 },
  { label: '14d', value: 14 },
  { label: '30d', value: 30 },
  { label: '90d', value: 90 },
  { label: '365d', value: 365 },
] as const

const CARD = 'rounded-lg border p-3 space-y-2'
const CARD_LABEL = 'text-[10px] font-mono text-(--ui-text-quaternary) uppercase'
const CARD_VALUE = 'text-base font-semibold text-(--ui-text-primary)'
const CARD_SUB = 'text-[10px] text-(--ui-text-quaternary)'

function formatUsd(n: number): string {
  if (n === 0 || n == null) return '$0.00'
  if (n >= 1) return `$${n.toFixed(2)}`
  return `$${n.toFixed(4)}`
}

function sparklineColor(percent: number, hasData: boolean): string {
  return hasData ? 'var(--ui-primary)' : 'var(--ui-text-quaternary)'
}

function ModelRow({ entry, topTotal }: { entry: AnalyticsResponse['by_model'][number]; topTotal: number }) {
  const tokens = (entry.input_tokens ?? 0) + (entry.output_tokens ?? 0)
  const pct = topTotal > 0 ? (tokens / topTotal) * 100 : 0
  const est = (entry.estimated_cost ?? entry.estimated_cost_usd ?? 0) as number

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="grid grid-cols-[1fr_60px_60px_60px] items-center gap-2 px-1 py-1.5 text-[11px] rounded hover:bg-(--ui-surface-hover)">
            <div className="min-w-0 overflow-hidden">
              <div className="font-mono truncate text-(--ui-text-primary)">
                {entry.model ?? 'unknown'}
              </div>
              <div className="text-[9px] text-(--ui-text-quaternary)">
                {compactNumber(entry.sessions ?? 0)} sessions
              </div>
            </div>
            <div className="text-(--ui-text-secondary) font-mono text-right">
              {pct >= 2 ? (
                <span className="inline-flex h-[10px] w-[60px] items-center">
                  <span
                    className="h-[6px] bg-(--ui-primary) rounded"
                    style={{ width: `${Math.max(pct, 1)}%` }}
                  />
                </span>
              ) : (
                <span className="text-[9px] text-(--ui-text-quaternary)">{pct.toFixed(1)}%</span>
              )}
            </div>
            <div className="text-[10px] font-mono text-(--ui-text-secondary) text-right">
              {compactNumber(tokens)}
            </div>
            <div className="text-[10px] font-mono text-(--ui-text-secondary) text-right">
              {est !== 0 ? formatUsd(est) : '—'}
            </div>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <div className="font-mono text-[10px]">{entry.model ?? 'unknown'}</div>
          <div>in: {compactNumber(entry.input_tokens ?? 0)}</div>
          <div>out: {compactNumber(entry.output_tokens ?? 0)}</div>
          <div>est: {formatUsd(est)}</div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export function UsageView() {
  const { t } = useI18n()
  const profile = useStore($activeGatewayProfile)
  const days = useStore($usageDays)

  const [preset, setPreset] = useState(() => DAY_PRESETS.find(p => p.value === days) ?? DAY_PRESETS[2])
  const [showCost, setShowCost] = useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['usage-summary', days, profile],
    queryFn: () => getUsageAnalytics(days),
    refetchOnWindowFocus: false,
    staleTime: 60_000,
  })

  const total = data?.totals ?? {}
  const daily = data?.daily ?? []
  const byModel = data?.by_model ?? []

  const topTotal = useMemo(() => {
    return byModel.reduce((s, e) => s + (e.input_tokens ?? 0) + (e.output_tokens ?? 0), 0)
  }, [byModel])

  const totalIn = (total.total_input ?? total.input_tokens ?? 0) as number
  const totalOut = (total.total_output ?? total.output_tokens ?? 0) as number
  const totalCache = (total.total_cache_read ?? total.cache_read_tokens ?? 0) as number
  const totalEst = (total.total_estimated_cost ?? total.estimated_cost ?? 0) as number
  const totalAct = (total.total_actual_cost ?? total.actual_cost ?? 0) as number
  const totalSessions = total.total_sessions ?? 0

  const dailyCost = useMemo(
    () => daily.map(d => ((d.estimated_cost ?? d.estimated_cost_usd) as number) ?? 0),
    [daily]
  )
  const dailyTokens = useMemo(
    () => daily.map(d => ((d.input_tokens ?? 0) + (d.output_tokens ?? 0)) as number),
    [daily]
  )

  const displayCost = showCost ? totalEst : totalAct
  const costLabel = showCost ? (t.usage?.estimated ?? 'Estimated') : (t.usage?.actual ?? 'Actual')

  const updateDays = (v: number) => {
    setUsageDays(v)
    setPreset(DAY_PRESETS.find(p => p.value === v) ?? DAY_PRESETS[2])
  }

  const hasData = totalSessions > 0

  if (isLoading) {
    return (
      <div className="flex h-full flex-col items-center justify-center">
        <Loader type="small" />
      </div>
    )
  }

  if (error) {
    return (
      <EmptyState
        title="Failed to load usage data"
        description={error.message ?? String(error)}
        className="h-full"
      />
    )
  }

  if (!data) {
    return emptyState(days)
  }

  return (
    <section className="flex h-full flex-col p-4 gap-4" aria-label="token and cost usage">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">📊</span>
          <span className="font-semibold">{t.usage?.title ?? 'Usage'}</span>
          <Badge variant="muted" className="text-[10px]">
            {days}d · {profile ?? 'active'}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          {(totalEst !== 0 || totalAct !== 0) && (
            <Button
              variant="ghost"
              className={cn(
                'h-7 text-[10px]',
                buttonVariants({ variant: 'ghost' })
              )}
              onClick={() => setShowCost(!showCost)}
            >
              {showCost ? 'show actual' : 'show estimated'}
            </Button>
          )}
          <Select value={String(preset.value)} onValueChange={v => updateDays(Number(v))}>
            <SelectTrigger className="h-7 w-[60px] text-[10px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DAY_PRESETS.map(p => (
                <SelectItem key={p.value} value={String(p.value)}>
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {hasData ? (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className={CARD}>
              <div className={CARD_LABEL}>{t.usage?.statSessions ?? 'Sessions'}</div>
              <div className={CARD_VALUE}>{compactNumber(totalSessions)}</div>
            </div>
            <div className={CARD}>
              <div className={CARD_LABEL}>{t.usage?.statTokens ?? 'Tokens'}</div>
              <div className={CARD_VALUE}>{compactNumber(totalIn + totalOut)}</div>
              <div className={CARD_SUB}>
                {compactNumber(totalIn)} in · {compactNumber(totalOut)} out
              </div>
            </div>
            <div className={CARD}>
              <div className={CARD_LABEL}>{costLabel}</div>
              <div className={CARD_VALUE}>{formatUsd(displayCost)}</div>
              {totalEst !== totalAct && (
                <div className={CARD_SUB}>
                  est: {formatUsd(totalEst)}
                </div>
              )}
            </div>
            <div className={CARD}>
              <div className={CARD_LABEL}>{t.usage?.cache ?? 'Cache reads'}</div>
              <div className={CARD_VALUE}>{compactNumber(totalCache)}</div>
            </div>
          </div>

          {/* Sparklines */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className={cn(CARD, 'space-y-1')}>
              <div className="text-[11px] font-mono text-(--ui-text-secondary)">tokens/day</div>
              <BarSparkline
                data={dailyTokens}
                color={sparklineColor(0, true)}
                height={40}
                barGap={1}
                tooltipFormatter={n => compactNumber(n)}
              />
            </div>
            <div className={cn(CARD, 'space-y-1')}>
              <div className="text-[11px] font-mono text-(--ui-text-secondary)">
                {showCost ? 'estimated cost/day' : 'actual cost/day'}
              </div>
              <BarSparkline
                data={dailyCost}
                color="var(--ui-usage-cost)"
                height={40}
                barGap={1}
                tooltipFormatter={formatUsd}
              />
            </div>
          </div>

          {/* Models table */}
          <div className={cn(CARD, 'space-y-2')}>
            <div className="text-[11px] font-mono text-(--ui-text-secondary) flex items-center justify-between">
              <span>{t.usage?.topModels ?? 'By model'}</span>
              <span className="text-(--ui-text-quaternary)">{byModel.length}</span>
            </div>
            <div className="grid grid-cols-[1fr_60px_60px_60px] items-center gap-2 px-1 py-1 text-[9px] font-mono text-(--ui-text-quaternary)">
              <span>model</span>
              <span>%</span>
              <span className="text-right">tokens</span>
              <span className="text-right">cost</span>
            </div>
            <div className="max-h-[200px] overflow-y-auto space-y-0">
              {byModel.slice(0, 20).map((e, i) => (
                <ModelRow key={e.model ?? `m-${i}`} entry={e} topTotal={topTotal} />
              ))}
            </div>
          </div>
        </>
      ) : (
        emptyState(days)
      )}
    </section>
  )
}

function emptyState(days: number) {
  return (
    <EmptyState
      title="No sessions recorded in the last " + days + " days"
      description="Once a turn completes, tokens and cost appear here automatically."
      className="h-[60vh]"
    />
  )
}
