import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { BarChart3, CreditCard, ExternalLink, Package, Wrench } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useRouteEnumParam } from '../../hooks/use-route-enum-param'
import {
  ListRow,
  ListRowSkeleton,
  SectionHeading,
  SectionHeadingSkeleton,
  SettingsContent,
  SettingsSection
} from '../primitives'

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
  const warn = notice.tone === 'warn'

  return (
    <div className={cn('mb-6 rounded-xl p-4', warn ? 'bg-(--ui-yellow)/10' : 'bg-(--ui-bg-quaternary)')}>
      <div
        className={cn(
          'text-[length:var(--conversation-text-font-size)] font-medium',
          warn ? 'text-(--ui-yellow)' : 'text-foreground'
        )}
      >
        {notice.title}
      </div>
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

// The payment method as it rides in the "Payment & credits" heading: the current
// card (muted) plus a single underline text action (Update / Add payment method).
function PaymentMethodAside({ row }: { row: BillingAccountRowView }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      {row.value && (
        <span className="min-w-0 truncate text-[length:var(--conversation-caption-font-size)] font-normal text-(--ui-text-tertiary)">
          {row.value}
        </span>
      )}
      {row.action && (
        <Button
          disabled={row.action.disabled}
          onClick={row.action.url ? () => openExternal(row.action?.url) : undefined}
          size="sm"
          type="button"
          variant="textStrong"
        >
          {row.action.label}
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
          <SegmentedControl
            disabled={controlsDisabled}
            onChange={value => setAmount(value)}
            options={presets.map(preset => ({ id: preset.amount, label: preset.label }))}
            value={amount}
          />
          <Input
            aria-label="Custom credit amount"
            containerClassName="w-16"
            disabled={controlsDisabled}
            inputMode="decimal"
            max={billing.max_usd ?? undefined}
            min={billing.min_usd ?? undefined}
            onBlur={() => setAmount(clampedAmount)}
            onChange={event => {
              flow.reset()
              setAmount(event.target.value)
            }}
            placeholder={billing.min_usd ?? ''}
            prefix="$"
            size="xs"
            step="0.01"
            type="number"
            value={amount}
          />
          <Button disabled={!canBuy} onClick={startBuy} size="xs" type="button" variant="secondary">
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

  // Plain shared primitive — no bespoke track chrome. Only the fill tone carries
  // billing meaning: destructive when over-limit, green for healthy remaining
  // credits, muted otherwise. Color rides the sanctioned `fillClassName` override.
  const isOk = resolvedBar.state === 'ok' && (resolvedBar.tone === 'subscription' || resolvedBar.tone === 'topup')

  return (
    <Progress
      aria-label={resolvedBar.label}
      destructive={resolvedBar.state === 'danger'}
      fillClassName={resolvedBar.state === 'danger' ? undefined : isOk ? 'bg-(--ui-green)' : 'bg-muted-foreground/45'}
      fillStyle={{ minWidth: resolvedBar.value > 0 ? 4 : undefined }}
      size="lg"
      value={resolvedBar.value}
    />
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

// DEV-only preview switcher: swaps the whole page onto a canned fixture so every
// billing state can be reviewed without a matching live account. Marked with a
// wrench + "preview" so it never reads as a shipping control (it's compiled out of
// production builds entirely).
function BillingFixtureSelect({
  onValueChange,
  value
}: {
  onValueChange: (value: BillingFixtureSelection) => void
  value: BillingFixtureSelection
}) {
  return (
    <div className="flex items-center gap-1.5 text-(--ui-text-tertiary)">
      <Wrench className="size-3.5 shrink-0" />
      <span className="text-xs font-normal">preview</span>
      <Select onValueChange={value => onValueChange(value as BillingFixtureSelection)} value={value}>
        <SelectTrigger
          aria-label="Billing preview fixture (dev only)"
          className="h-7 w-36 border-dashed border-(--ui-stroke-secondary) bg-transparent px-2 text-xs font-normal text-(--ui-text-tertiary) shadow-none hover:bg-(--ui-bg-tertiary) focus-visible:ring-0 focus-visible:ring-offset-0 data-[state=open]:bg-(--ui-bg-tertiary)"
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
    </div>
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

// Loading shape for the billing overview: three summary cards over the Plan /
// Payment & credits / Usage sections. Rendered under the real header.
function BillingSkeleton() {
  return (
    <>
      <div className="@container mb-6">
        <div className="grid gap-3 @2xl:grid-cols-3">
          {[0, 1, 2].map(i => (
            <div className="min-w-0 space-y-2" key={i}>
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-6 w-20" />
            </div>
          ))}
        </div>
      </div>
      {[0, 1, 2].map(section => (
        <section className="mb-6" key={section}>
          <SectionHeadingSkeleton />
          <div className="grid gap-1">
            <ListRowSkeleton />
            <ListRowSkeleton />
          </div>
        </section>
      ))}
    </>
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

  // First load keeps the page's shape via a skeleton instead of flashing "—"
  // summary cards (background refetches leave `isPending` false, so no flicker).
  if (billingState.isPending) {
    return (
      <SettingsContent>
        <BillingHeader fixtureName={fixtureName} onFixtureChange={onFixtureChange} />
        <BillingSkeleton />
      </SettingsContent>
    )
  }

  const billingResult = billingState.data
  const subscriptionResult = subscriptionState.data
  const view = deriveBillingView(billingResult, subscriptionResult)
  const billing = billingResult?.ok ? billingResult.data : undefined

  const { paymentRow, refillRow, topupRow } = view

  // The payment method rides in the section header (right-aligned) — the
  // "Payment & credits" title already names it, so a full labelled row would just
  // repeat "Payment method". The stacked rows are the remaining money controls.
  const accountRows = [topupRow, refillRow].filter((row): row is BillingAccountRowView => row !== undefined)

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

      {view.notice && <NoticeCard notice={view.notice} />}

      <div className="@container mb-6">
        <div className="grid gap-3 @2xl:grid-cols-3">
          {view.summary.map(item => (
            <SummaryCard key={item.label} label={item.label} tone={item.tone} value={item.value} />
          ))}
        </div>
      </div>

      {view.plan && (
        <SettingsSection icon={Package} title="Plan">
          <CurrentPlanCard onViewPlans={() => setSubView('plans')} plan={view.plan} />
        </SettingsSection>
      )}

      {(paymentRow || accountRows.length > 0) && (
        <SettingsSection
          aside={paymentRow ? <PaymentMethodAside row={paymentRow} /> : undefined}
          icon={CreditCard}
          title="Payment & credits"
        >
          {accountRows.map(row => (
            <AccountRow billing={billing} key={row.id} row={row} />
          ))}
        </SettingsSection>
      )}

      {view.usageRows.length > 0 && (
        <SettingsSection icon={BarChart3} title="Usage">
          <div className="@container">
            {view.usageRows.map(row => (
              <UsageRow key={row.id} row={row} />
            ))}
          </div>
        </SettingsSection>
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
