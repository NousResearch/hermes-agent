import { compactNumber } from '@hermes/shared'
import { useEffect, useMemo, useState } from 'react'

import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { useI18n } from '@/i18n'
import type { ModelUsageRoute, ModelUsageTotals, SessionModelUsage, UsageStats } from '@/types/hermes'

interface ModelUsageStatusbarOptions {
  activeSessionId: string | null
  currentModel: string
  currentProvider: string
  currentUsage: UsageStats
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

const EMPTY_TOTALS: ModelUsageTotals = {
  actual_cost_usd: 0,
  cache_read: 0,
  cache_write: 0,
  calls: 0,
  estimated_cost_usd: 0,
  input: 0,
  output: 0,
  reasoning: 0,
  total: 0
}

function displayModelName(model: string): string {
  const normalized = model.trim()
  const leaf = normalized.split('/').filter(Boolean).pop()

  return leaf || normalized
}

function sameModel(left: string, right: string): boolean {
  if (!left || !right) {
    return false
  }

  return left === right || displayModelName(left) === displayModelName(right)
}

function money(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return '$0'
  }

  if (value < 0.0001) {
    return '<$0.0001'
  }

  return `$${value
    .toFixed(value < 1 ? 4 : 2)
    .replace(/0+$/, '')
    .replace(/\.$/, '')}`
}

export function useModelUsageStatusbarItem({
  activeSessionId,
  currentModel,
  currentProvider,
  currentUsage,
  requestGateway
}: ModelUsageStatusbarOptions): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const [usage, setUsage] = useState<SessionModelUsage | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  useEffect(() => {
    if (!activeSessionId) {
      setUsage(null)
      setLoading(false)
      setError(false)

      return
    }

    let cancelled = false
    setLoading(true)
    setError(false)

    void requestGateway<SessionModelUsage>('session.model_usage', { session_id: activeSessionId })
      .then(data => {
        if (!cancelled) {
          setUsage(data)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError(true)
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
  }, [activeSessionId, currentUsage.calls, currentUsage.input, currentUsage.output, currentUsage.total, requestGateway])

  const fallbackRoute = useMemo<ModelUsageRoute | null>(() => {
    if (!currentModel || currentUsage.total <= 0 || (usage?.routes.length ?? 0) > 0) {
      return null
    }

    return {
      actual_cost_usd: 0,
      billing_mode: '',
      cache_read: 0,
      cache_write: 0,
      calls: currentUsage.calls,
      cost_source: '',
      cost_status: '',
      estimated_cost_usd: currentUsage.cost_usd ?? 0,
      input: currentUsage.input,
      last_seen: 0,
      model: currentModel,
      output: currentUsage.output,
      provider: currentProvider,
      reasoning: 0,
      total: currentUsage.total
    }
  }, [currentModel, currentProvider, currentUsage, usage?.routes.length])

  const routes = usage?.routes.length ? usage.routes : fallbackRoute ? [fallbackRoute] : []

  const activeRoute =
    routes.find(
      route =>
        sameModel(route.model, currentModel) &&
        (!currentProvider || !route.provider || route.provider === currentProvider)
    ) ?? routes.find(route => sameModel(route.model, currentModel))

  const modelLabel = displayModelName(currentModel || activeRoute?.model || '')

  const detail = activeRoute?.total
    ? `↑${compactNumber(activeRoute.input)} ↓${compactNumber(activeRoute.output)}`
    : undefined

  return {
    detail,
    hidden: !modelLabel && currentUsage.total <= 0,
    id: 'model-usage',
    label: modelLabel || copy.modelUsage,
    menuAlign: 'end',
    menuClassName: 'w-auto border-(--ui-stroke-secondary) p-0',
    menuContent: (
      <ModelUsagePanel
        activeModel={currentModel}
        error={error}
        loading={loading}
        routes={routes}
        totals={usage?.totals ?? fallbackTotals(fallbackRoute)}
      />
    ),
    title: copy.openModelUsage,
    variant: 'menu'
  }
}

function fallbackTotals(route: ModelUsageRoute | null): ModelUsageTotals {
  if (!route) {
    return EMPTY_TOTALS
  }

  return {
    actual_cost_usd: route.actual_cost_usd,
    cache_read: route.cache_read,
    cache_write: route.cache_write,
    calls: route.calls,
    estimated_cost_usd: route.estimated_cost_usd,
    input: route.input,
    output: route.output,
    reasoning: route.reasoning,
    total: route.total
  }
}

function ModelUsagePanel({
  activeModel,
  error,
  loading,
  routes,
  totals
}: {
  activeModel: string
  error: boolean
  loading: boolean
  routes: readonly ModelUsageRoute[]
  totals: ModelUsageTotals
}) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.modelUsagePanel

  return (
    <div className="flex w-80 flex-col gap-3 p-3 text-[0.75rem]" data-slot="model-usage-panel">
      <div className="flex items-baseline justify-between gap-3">
        <p className="font-medium text-foreground">{copy.title}</p>
        <span className="shrink-0 text-[0.6875rem] tabular-nums text-muted-foreground">
          {copy.totalTokens(compactNumber(totals.total))}
        </span>
      </div>

      {loading && routes.length === 0 && <p className="text-muted-foreground">{copy.loading}</p>}
      {error && routes.length === 0 && <p className="text-destructive">{copy.unavailable}</p>}
      {!loading && !error && routes.length === 0 && <p className="text-muted-foreground">{copy.empty}</p>}

      {routes.length > 0 && (
        <ul className="divide-y divide-(--ui-stroke-tertiary)">
          {routes.map((route, index) => {
            const active = sameModel(route.model, activeModel)

            const cost =
              route.actual_cost_usd > 0
                ? copy.actualCost(money(route.actual_cost_usd))
                : route.estimated_cost_usd > 0
                  ? copy.estimatedCost(money(route.estimated_cost_usd))
                  : null

            return (
              <li
                className="flex flex-col gap-1 py-2 first:pt-0 last:pb-0"
                key={`${route.model}:${route.provider}:${route.billing_mode}:${index}`}
              >
                <div className="flex min-w-0 items-center justify-between gap-3">
                  <span className="min-w-0 truncate font-medium text-foreground">{route.model}</span>
                  {active && <span className="shrink-0 text-[0.625rem] text-primary">{copy.active}</span>}
                </div>
                <div className="flex min-w-0 items-center justify-between gap-3 text-[0.6875rem] text-muted-foreground">
                  <span className="truncate">{route.provider || copy.unknownProvider}</span>
                  <span className="shrink-0 tabular-nums">
                    ↑{compactNumber(route.input)} · ↓{compactNumber(route.output)} · {copy.calls(route.calls)}
                  </span>
                </div>
                {(route.cache_read > 0 || route.reasoning > 0 || cost) && (
                  <div className="flex flex-wrap items-center gap-x-3 text-[0.625rem] text-muted-foreground">
                    {route.cache_read > 0 && <span>{copy.cacheRead(compactNumber(route.cache_read))}</span>}
                    {route.reasoning > 0 && <span>{copy.reasoning(compactNumber(route.reasoning))}</span>}
                    {cost && <span>{cost}</span>}
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
