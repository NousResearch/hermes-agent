import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { BarChart3, ExternalLink, Lock, Package, Plus, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useRouteEnumParam } from '../../hooks/use-route-enum-param'
import { ListRow, SectionHeading, SettingsContent } from '../primitives'

import { RowValue } from './account-row-value'
import { BillingApiProvider } from './api'
import { AutoReloadRow } from './auto-reload-row'
import { clampAmount, formatMoney } from './billing-amounts'
import { CurrentPlanCard } from './current-plan-card'
import { type BillingDevFixtureName, billingDevFixtures } from './dev-fixtures'
import { StepUpInlineAction } from './inline-feedback'
import { openExternal } from './open-external'
import { BillingPlansView } from './plans-view'
import { createSimulatedBillingApi } from './simulated-api'
import type { BillingStateResponse } from './types'
import {
  type BillingAccountRowView,
  type BillingNoticeView,
  type BillingUsageRowView,
  deriveBillingView,
  formatUsageUpdatedAgo,
  useBillingState,
  useSubscriptionState
} from './use-billing-state'
import { useChargeFlow } from './use-charge-poller'
import { useStepUpFlow } from './use-step-up'

// `bview` mirrors the settings pview/kview sub-view pattern (deep-linkable, replace
// navigation). `overview` is the default landing; `plans` is the in-app catalog.
const BILLING_VIEWS = ['overview', 'plans'] as const
type BillingSubView = (typeof BILLING_VIEWS)[number]

const FEATURE_BILLING_INVOICES = false

const BILLING_DEV_FIXTURE_NAMES = import.meta.env.DEV
  ? (Object.keys(billingDevFixtures) as BillingDevFixtureName[])
  : []

type BillingFixtureSelection = 'live' | BillingDevFixtureName

function SummaryCard({ label, value, tone }: { label: string; tone?: 'muted' | 'primary'; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">{label}</div>
      <div
        className={cn(
          'mt-1 min-w-0 truncate text-lg font-semibold tabular-nums',
          tone === 'primary' ? 'text-(--ui-green)' : tone === 'muted' ? 'text-(--ui-text-tertiary)' : 'text-foreground'
        )}
      >
        {value}
      </div>
    </div>
  )
}

function NoticeCard({ notice }: { notice: BillingNoticeView }) {
  return (
    <div className="mb-5 rounded-lg border border-border/70 bg-muted/20 p-4">
      <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">{notice.title}</div>
      <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
        {notice.message}
      </div>
      {notice.action && (
        <Button
          className="mt-3"
          onClick={() => openExternal(notice.action?.url)}
          size="sm"
          type="button"
          variant="outline"
        >
          {notice.action.label}
          <ExternalLink className="size-3.5" />
        </Button>
      )}
    </div>
  )
}

function AccountRow({ billing, row }: { billing?: BillingStateResponse; row: BillingAccountRowView }) {
  if (row.id === 'buy_credits' && row.action && row.chips && billing?.can_charge && billing.cli_billing_enabled) {
    return <BuyCreditsRow billing={billing} row={row} />
  }

  if (row.id === 'auto_reload' && billing?.auto_reload) {
    return <AutoReloadRow autoReload={billing.auto_reload} bounds={billing} row={row} />
  }

  return (
    <ListRow
      action={<RowValue row={row} />}
      below={
        row.caption ? (
          <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {row.caption}
          </div>
        ) : undefined
      }
      description={row.description}
      key={row.id}
      title={row.title}
    />
  )
}

function BuyCreditsRow({ billing, row }: { billing: BillingStateResponse; row: BillingAccountRowView }) {
  const presets = useMemo(
    () =>
      billing.charge_presets.map((amount, index) => ({
        amount,
        label: billing.charge_presets_display[index] || formatMoney(amount)
      })),
    [billing.charge_presets, billing.charge_presets_display]
  )

  const initialAmount = presets[0]?.amount ?? billing.min_usd ?? ''
  const [amount, setAmount] = useState(initialAmount)
  const flow = useChargeFlow()
  const busy = flow.phase === 'charging' || flow.phase === 'polling'
  const controlsDisabled = busy || !billing.card
  const clampedAmount = clampAmount(amount, billing)
  const canBuy = !controlsDisabled && clampedAmount !== ''

  const startBuy = () => {
    if (!canBuy) {
      return
    }

    setAmount(clampedAmount)
    void flow.start(clampedAmount)
  }

  return (
    <ListRow
      action={
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
          {presets.map(preset => (
            <Button
              aria-pressed={amount === preset.amount}
              disabled={controlsDisabled}
              key={preset.amount}
              onClick={() => setAmount(preset.amount)}
              size="sm"
              type="button"
              variant={amount === preset.amount ? 'default' : 'outline'}
            >
              {preset.label}
            </Button>
          ))}
          <Input
            aria-label="Custom credit amount"
            className="w-24 py-[3px]"
            disabled={controlsDisabled}
            inputMode="decimal"
            max={billing.max_usd ?? undefined}
            min={billing.min_usd ?? undefined}
            onBlur={() => setAmount(clampedAmount)}
            onChange={event => {
              flow.reset()
              setAmount(event.target.value)
            }}
            placeholder={billing.min_usd ? formatMoney(billing.min_usd) : '$'}
            size="sm"
            step="0.01"
            type="number"
            value={amount}
          />
          <Button disabled={!canBuy} onClick={startBuy} size="sm" type="button" variant="outline">
            Buy
          </Button>
        </div>
      }
      below={
        <BuyCreditsOutcome
          amount={clampedAmount}
          busy={busy}
          onPortal={openExternal}
          onRetry={() => {
            if (!clampedAmount) {
              return
            }

            void flow.start(clampedAmount)
          }}
          outcome={flow.outcome}
        />
      }
      description={row.description}
      key={row.id}
      title={row.title}
    />
  )
}

function BuyCreditsOutcome({
  amount,
  busy,
  onPortal,
  onRetry,
  outcome
}: {
  amount: string
  busy: boolean
  onPortal: (url?: string) => void
  onRetry: () => void
  outcome: ReturnType<typeof useChargeFlow>['outcome']
}) {
  const stepUp = useStepUpFlow()

  if (busy) {
    return (
      <div className="mt-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        Processing… checking settlement
      </div>
    )
  }

  if (!outcome) {
    return null
  }

  if (outcome.kind === 'success') {
    return (
      <div className="mt-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        {formatMoney(outcome.amountUsd ?? amount)} added. Balance is refreshing.
      </div>
    )
  }

  if (outcome.kind === 'ambiguous') {
    return (
      <div className="mt-2 flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        <span>
          {outcome.title}: {outcome.message}
        </span>
        {outcome.portalUrl && (
          <Button onClick={() => onPortal(outcome.portalUrl)} size="sm" type="button" variant="outline">
            Open portal
            <ExternalLink className="size-3.5" />
          </Button>
        )}
      </div>
    )
  }

  const portalUrl = outcome.action?.type === 'portal' ? outcome.action.url : undefined

  return (
    <div className="mt-2 flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
      <span>
        {outcome.title}: {outcome.message}
      </span>
      {outcome.action?.type === 'retry' && (
        <Button onClick={onRetry} size="sm" type="button" variant="outline">
          Retry
        </Button>
      )}
      {outcome.action?.type === 'step_up' && <StepUpInlineAction flow={stepUp} />}
      {portalUrl && (
        <Button onClick={() => onPortal(portalUrl)} size="sm" type="button" variant="outline">
          Open portal
          <ExternalLink className="size-3.5" />
        </Button>
      )}
    </div>
  )
}

function UsageBar({ bar, fallbackLabel }: { bar?: BillingUsageRowView['bar']; fallbackLabel: string }) {
  const resolvedBar = bar ?? {
    label: `${fallbackLabel} usage`,
    state: 'neutral',
    tone: 'topup',
    value: 0
  }

  const width = Math.round(resolvedBar.value * 100)
  const isEmpty = resolvedBar.value === 0
  const showDangerNub = resolvedBar.track === 'danger' && resolvedBar.state === 'danger' && width === 0

  return (
    <div
      aria-label={resolvedBar.label}
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={width}
      className={cn(
        // Radius follows the app-wide rounded-full progress-bar idiom.
        'relative h-2 w-full overflow-hidden rounded-full',
        resolvedBar.track === 'danger'
          ? 'dither text-destructive/60 bg-destructive/10'
          : isEmpty
            ? 'dither bg-(--ui-bg-elevated)'
            : 'bg-muted shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--ui-stroke-secondary)_50%,transparent)]'
      )}
      role="progressbar"
    >
      {showDangerNub && <div className="absolute inset-y-0 left-0 z-10 w-2 rounded-full bg-destructive" />}
      <div
        className={cn(
          'relative h-full rounded-full transition-[width] duration-300 ease-out',
          resolvedBar.state === 'danger'
            ? 'bg-destructive'
            : resolvedBar.state === 'ok' && (resolvedBar.tone === 'subscription' || resolvedBar.tone === 'topup')
              ? 'bg-(--ui-green)'
              : 'bg-muted-foreground/45'
        )}
        style={{
          minWidth: resolvedBar.value > 0 ? 4 : undefined,
          width: `${width}%`
        }}
      />
    </div>
  )
}

function UsageRow({ row }: { row: BillingUsageRowView }) {
  return (
    <div className="@container">
      <div className="grid min-w-0 gap-2 py-3 @2xl:grid-cols-[minmax(0,180px)_minmax(0,1fr)_220px] @2xl:items-center @2xl:gap-4">
        <div className="min-w-0">
          <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {row.title}
          </div>
          <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {row.caption}
          </div>
        </div>
        <div className="min-w-0">
          <UsageBar bar={row.bar} fallbackLabel={row.title} />
        </div>
        <div
          className={cn(
            'min-w-0 whitespace-nowrap text-[length:var(--conversation-text-font-size)] font-medium tabular-nums @2xl:w-[220px] @2xl:flex-none @2xl:text-right',
            row.bar?.state === 'danger' ? 'text-destructive' : 'text-foreground'
          )}
        >
          {row.value}
        </div>
      </div>
    </div>
  )
}

function UsageRefreshRow({
  fixtureName,
  isFetching,
  onRefresh,
  updatedAt
}: {
  fixtureName?: BillingFixtureSelection
  isFetching: boolean
  onRefresh: () => void
  updatedAt: number
}) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 30_000)

    return () => window.clearInterval(interval)
  }, [])

  if (fixtureName && fixtureName !== 'live') {
    return (
      <div className="flex items-center justify-end pt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
        fixture: {fixtureName}
      </div>
    )
  }

  return (
    <div className="flex min-w-0 items-center justify-end gap-1.5 pt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
      <span>Updated {formatUsageUpdatedAgo(updatedAt, now)}</span>
      <Tip label="Refresh">
        <Button
          aria-label="Refresh"
          className="size-7 p-0 text-(--ui-text-tertiary)"
          disabled={isFetching}
          onClick={onRefresh}
          size="sm"
          type="button"
          variant="ghost"
        >
          <RefreshCw className={cn('size-3.5', isFetching && 'animate-spin')} />
        </Button>
      </Tip>
    </div>
  )
}

function BillingFixtureSelect({
  onValueChange,
  value
}: {
  onValueChange: (value: BillingFixtureSelection) => void
  value: BillingFixtureSelection
}) {
  return (
    <Select onValueChange={value => onValueChange(value as BillingFixtureSelection)} value={value}>
      <SelectTrigger
        aria-label="Billing fixture"
        className="h-7 w-32 border-transparent bg-transparent px-1.5 text-xs font-normal text-(--ui-text-tertiary) shadow-none hover:bg-muted/40 focus-visible:ring-0 focus-visible:ring-offset-0 data-[state=open]:bg-muted/40"
        size="sm"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end">
        <SelectItem value="live">live</SelectItem>
        {BILLING_DEV_FIXTURE_NAMES.map(name => (
          <SelectItem key={name} value={name}>
            {name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function BillingHeader({
  fixtureName,
  onFixtureChange
}: {
  fixtureName?: BillingFixtureSelection
  onFixtureChange?: (value: BillingFixtureSelection) => void
}) {
  return (
    <div className="mb-2.5 flex items-center justify-between gap-3 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
      <div className="flex min-w-0 items-center gap-2">
        <BarChart3 className="size-4 shrink-0 text-muted-foreground" />
        <span>Billing</span>
      </div>
      {import.meta.env.DEV && fixtureName && onFixtureChange ? (
        <BillingFixtureSelect onValueChange={onFixtureChange} value={fixtureName} />
      ) : null}
    </div>
  )
}

function BillingSettingsContent({
  fixtureName,
  onFixtureChange
}: {
  fixtureName?: BillingFixtureSelection
  onFixtureChange?: (value: BillingFixtureSelection) => void
}) {
  const [subView, setSubView] = useRouteEnumParam<BillingSubView>('bview', BILLING_VIEWS, 'overview')

  // Fixture mode flows through the SAME query path — the simulated api (supplied by
  // BillingApiProvider in the DEV wrapper) backs these fetches — so there is no
  // fixture short-circuit here.
  const billingState = useBillingState()
  const subscriptionState = useSubscriptionState()
  const billingResult = billingState.data
  const subscriptionResult = subscriptionState.data
  const view = deriveBillingView(billingResult, subscriptionResult)
  const billing = billingResult?.ok ? billingResult.data : undefined
  const usageUpdatedAt = oldestUpdatedAt(billingState.dataUpdatedAt, subscriptionState.dataUpdatedAt)
  const usageIsFetching = billingState.isFetching || subscriptionState.isFetching

  const refreshUsage = () => {
    void Promise.all([billingState.refetch(), subscriptionState.refetch()])
  }

  const { paymentRow, refillRow, topupRow } = view

  // Gate the plans sub-view on the SAME capability that renders the in-app button
  // (`plan.action`): a team / non-changer deep-linking `bview=plans` must never
  // reach a grid of live Choose buttons — it falls back to the overview.
  const showPlans = subView === 'plans' && view.status === 'normal' && Boolean(view.plan?.action)

  if (showPlans) {
    return (
      <SettingsContent>
        <BillingHeader fixtureName={fixtureName} onFixtureChange={onFixtureChange} />
        <BillingPlansView onBack={() => setSubView('overview')} tiers={view.tiers} />
      </SettingsContent>
    )
  }

  return (
    <SettingsContent>
      <BillingHeader fixtureName={fixtureName} onFixtureChange={onFixtureChange} />

      <div className="@container mb-5">
        <div className="grid gap-3 rounded-lg border border-border/70 bg-muted/20 p-4 @2xl:grid-cols-3">
          {view.summary.map(item => (
            <SummaryCard key={item.label} label={item.label} tone={item.tone} value={item.value} />
          ))}
        </div>
      </div>

      {view.notice && <NoticeCard notice={view.notice} />}

      {view.plan && (
        <div className="mb-5">
          <SectionHeading icon={Package} title="Plan" />
          <CurrentPlanCard onViewPlans={() => setSubView('plans')} plan={view.plan} />
        </div>
      )}

      {paymentRow && (
        <div className="mb-5">
          <SectionHeading icon={Lock} title="Payment" />
          <AccountRow billing={billing} row={paymentRow} />
        </div>
      )}

      {topupRow && (
        <div className="mb-5">
          <SectionHeading icon={Plus} title="One-time top-up" />
          <AccountRow billing={billing} row={topupRow} />
        </div>
      )}

      {refillRow && (
        <div className="mb-5">
          <SectionHeading icon={RefreshCw} title="Automatic refill" />
          <AccountRow billing={billing} row={refillRow} />
        </div>
      )}

      {view.usageRows.length > 0 && (
        <>
          <SectionHeading icon={BarChart3} title="Usage" />
          <div className="@container rounded-lg border border-border/70 bg-muted/20 px-4 py-2">
            {view.usageRows.map(row => (
              <UsageRow key={row.id} row={row} />
            ))}
            <UsageRefreshRow
              fixtureName={fixtureName}
              isFetching={usageIsFetching}
              onRefresh={refreshUsage}
              updatedAt={usageUpdatedAt}
            />
          </div>
        </>
      )}

      {
        // no endpoint yet — NAS capability-board gap
        FEATURE_BILLING_INVOICES ? <SectionHeading icon={BarChart3} title="Invoices" /> : null
      }
    </SettingsContent>
  )
}

function BillingSettingsWithDevFixtures() {
  const [fixtureName, setFixtureName] = useState<BillingFixtureSelection>('live')
  const queryClient = useQueryClient()

  // DEV-only: a picked fixture is served by a simulated api (in-memory, mutable) that
  // the whole subtree resolves via BillingApiProvider → useBillingApi. `live` → null →
  // the real gateway api. Rebuilt per fixture so switching starts from a fresh copy.
  const simulatedApi = useMemo(
    () => (fixtureName !== 'live' ? createSimulatedBillingApi(billingDevFixtures[fixtureName]) : null),
    [fixtureName]
  )

  // Switching fixtures (or its simulated api) must refetch, since the billing queries
  // are keyed the same across fixtures.
  useEffect(() => {
    void queryClient.invalidateQueries({ queryKey: ['billing'] })
  }, [queryClient, simulatedApi])

  return (
    <BillingApiProvider value={simulatedApi}>
      <BillingSettingsContent fixtureName={fixtureName} onFixtureChange={setFixtureName} />
    </BillingApiProvider>
  )
}

export function BillingSettings() {
  if (import.meta.env.DEV) {
    return <BillingSettingsWithDevFixtures />
  }

  return <BillingSettingsContent />
}

function oldestUpdatedAt(...timestamps: number[]): number {
  const populated = timestamps.filter(timestamp => timestamp > 0)

  return populated.length > 0 ? Math.min(...populated) : Date.now()
}
