import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { BarChart3, ExternalLink, Lock, Package, Plus, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { useRouteEnumParam } from '../../hooks/use-route-enum-param'
import { ListRow, Pill, SectionHeading, SettingsContent } from '../primitives'

import type { BillingRefusal } from './api'
import { useBillingApi } from './api'
import { type BillingDevFixtureName, billingDevFixtures } from './dev-fixtures'
import { resolveRefusal } from './errors'
import { BillingPlansView } from './plans-view'
import { TierArt } from './tier-art'
import type { BillingAutoReload, BillingStateResponse } from './types'
import {
  type BillingAccountRowView,
  type BillingNoticeView,
  type BillingPlanCardView,
  type BillingUsageRowView,
  deriveBillingView,
  EMPTY_BILLING_VALUE,
  formatUsageUpdatedAgo,
  useBillingState,
  useSubscriptionState
} from './use-billing-state'

// `bview` mirrors the settings pview/kview sub-view pattern (deep-linkable, replace
// navigation). `overview` is the default landing; `plans` is the in-app catalog.
const BILLING_VIEWS = ['overview', 'plans'] as const
type BillingSubView = (typeof BILLING_VIEWS)[number]
import { useChargeFlow } from './use-charge-poller'
import { useStepUpFlow } from './use-step-up'

const FEATURE_BILLING_INVOICES = false

const BILLING_DEV_FIXTURE_NAMES = import.meta.env.DEV
  ? (Object.keys(billingDevFixtures) as BillingDevFixtureName[])
  : []

type BillingFixtureSelection = 'live' | BillingDevFixtureName

function openExternal(url?: string) {
  if (!url) {
    return
  }

  void window.hermesDesktop?.openExternal?.(url)
}

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

function RowValue({ onAction, row }: { onAction?: () => void; row: BillingAccountRowView }) {
  return (
    <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
      {row.value && (
        <span className="min-w-0 truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
          {row.value}
        </span>
      )}
      {row.pill && <Pill tone={row.pill.tone}>{row.pill.label}</Pill>}
      {row.secondaryPill && <Pill>{row.secondaryPill}</Pill>}
      {row.chips?.map(chip => (
        <Button
          disabled={chip.disabled}
          key={chip.label}
          onClick={chip.url ? () => openExternal(chip.url) : undefined}
          size="sm"
          type="button"
          variant="outline"
        >
          {chip.label}
        </Button>
      ))}
      {row.action && (
        <Button
          disabled={row.action.disabled}
          onClick={row.action.disabled ? undefined : onAction ? onAction : () => openExternal(row.action?.url)}
          size="sm"
          type="button"
          variant="outline"
        >
          {row.action.label}
          {!row.action.disabled && row.action.url && <ExternalLink className="size-3.5" />}
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

function CurrentPlanCard({ onViewPlans, plan }: { onViewPlans: () => void; plan: BillingPlanCardView }) {
  return (
    <div className="@container">
      <div className="grid gap-3 py-3 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <TierArt name={plan.tierName} />
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
              <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
                {plan.tierName}
              </span>
              {plan.price && (
                <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {plan.price}/mo
                </span>
              )}
            </div>
            <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {plan.caption}
            </div>
          </div>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
          {plan.action && (
            <Button onClick={onViewPlans} size="sm" type="button" variant="outline">
              {plan.action.label}
            </Button>
          )}
          {plan.link && (
            <Button onClick={() => openExternal(plan.link?.url)} size="sm" type="button" variant="outline">
              {plan.link.label}
              <ExternalLink className="size-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

function AutoReloadRow({
  autoReload,
  bounds,
  row
}: {
  autoReload: BillingAutoReload
  bounds: Pick<BillingStateResponse, 'max_usd' | 'min_usd'>
  row: BillingAccountRowView
}) {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [confirmDisable, setConfirmDisable] = useState(false)
  const [editing, setEditing] = useState(false)
  // Validation errors are silent until the user edits a field or attempts a
  // save — opening Manage on a prefilled (possibly below-min) config must not
  // flash an error (spec §9).
  const [showErrors, setShowErrors] = useState(false)
  const [message, setMessage] = useState<null | { kind: 'error' | 'success'; text: string }>(null)
  const [refusal, setRefusal] = useState<BillingRefusal | null>(null)

  const [reloadTo, setReloadTo] = useState(
    initialAutoReloadAmount(autoReload.reload_to_usd, autoReload.reload_to_display)
  )

  const [saving, setSaving] = useState(false)

  const [threshold, setThreshold] = useState(
    initialAutoReloadAmount(autoReload.threshold_usd, autoReload.threshold_display)
  )

  const validation = validateAutoReloadInputs(threshold, reloadTo, bounds)
  const busy = saving
  const maxBound = bounds.max_usd ?? undefined
  const minBound = bounds.min_usd ?? undefined

  // Only the canonical-card enabled state edits in place. Off / divergent-card
  // rows have no Manage affordance (or a portal link) and render read-only.
  const editable = row.action?.label === 'Manage' && !row.action.url

  const resetFeedback = () => {
    setConfirmDisable(false)
    setMessage(null)
    setRefusal(null)
  }

  const openEdit = () => {
    resetFeedback()
    setShowErrors(false)
    setEditing(true)
  }

  const cancelEdit = () => {
    resetFeedback()
    setEditing(false)
  }

  const save = async () => {
    if (!validation.values || busy) {
      return
    }

    resetFeedback()
    setSaving(true)

    const result = await api.updateAutoReload({
      enabled: true,
      reload_to_usd: validation.values.reloadTo,
      threshold_usd: validation.values.threshold
    })

    setSaving(false)

    if (!result.ok) {
      setRefusal(result.refusal)

      return
    }

    await queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
    setMessage({ kind: 'success', text: 'Auto-refill updated.' })
    setEditing(false)
  }

  const disable = async () => {
    if (busy) {
      return
    }

    resetFeedback()
    setSaving(true)

    // The gateway's billing.auto_reload handler unconditionally requires threshold
    // + top_up_amount (invalid_request otherwise), so a disable must still carry the
    // current amounts — mirroring the TUI, which always sends both.
    const result = await api.updateAutoReload({
      enabled: false,
      reload_to_usd: initialAutoReloadAmount(autoReload.reload_to_usd, autoReload.reload_to_display),
      threshold_usd: initialAutoReloadAmount(autoReload.threshold_usd, autoReload.threshold_display)
    })

    setSaving(false)

    if (!result.ok) {
      setRefusal(result.refusal)

      return
    }

    await queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
    setMessage({ kind: 'success', text: 'Auto-refill turned off.' })
    setEditing(false)
  }

  // Read-only states (off / divergent card) keep the original ListRow shape.
  if (!editable) {
    return (
      <ListRow
        action={<RowValue row={row} />}
        below={
          <>
            {row.caption ? (
              <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {row.caption}
              </div>
            ) : null}
            <BillingRefusalInline refusal={refusal} />
            {message && <InlineMessage kind={message.kind}>{message.text}</InlineMessage>}
          </>
        }
        description={row.description}
        key={row.id}
        title={row.title}
      />
    )
  }

  const onField = (setter: (value: string) => void) => (event: { target: { value: string } }) => {
    resetFeedback()
    setShowErrors(true)
    setter(event.target.value)
  }

  // Zero-shift by exact reservation, not a magic min-height: the edit form is
  // ALWAYS rendered and both states share a single grid cell (`[grid-area:stack]`),
  // so the row's height always equals the tallest state at EVERY container width —
  // no breakpoint math that under-reserves when the two inputs stack on narrow
  // panes. The form is `invisible` + `aria-hidden` when not editing.
  return (
    <div className="@container">
      <div className="grid gap-3 py-3 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-start">
        <div className="min-w-0">
          <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {row.title}
          </div>
          <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
            {row.description}
          </div>
          <div className="mt-3 grid [grid-template-areas:'stack']">
            {/* EDIT layer — always in layout (reserves exact height); hidden until editing. */}
            <div
              aria-hidden={!editing}
              className={cn('space-y-2 [grid-area:stack]', !editing && 'invisible')}
            >
              <div className="grid gap-2 @2xl:grid-cols-2">
                <label className="min-w-0 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  Threshold
                  <Input
                    aria-label="Auto-refill threshold"
                    className="mt-1 h-8"
                    disabled={busy || !editing}
                    inputMode="decimal"
                    max={maxBound}
                    min={minBound}
                    onChange={onField(setThreshold)}
                    step="0.01"
                    tabIndex={editing ? undefined : -1}
                    type="number"
                    value={threshold}
                  />
                </label>
                <label className="min-w-0 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  Reload to
                  <Input
                    aria-label="Auto-refill reload-to amount"
                    className="mt-1 h-8"
                    disabled={busy || !editing}
                    inputMode="decimal"
                    max={maxBound}
                    min={minBound}
                    onChange={onField(setReloadTo)}
                    step="0.01"
                    tabIndex={editing ? undefined : -1}
                    type="number"
                    value={reloadTo}
                  />
                </label>
              </div>
              {/* Pre-allocated error line — occupies height whether or not shown. */}
              <div className="min-h-4 text-[length:var(--conversation-caption-font-size)] text-destructive">
                {showErrors && validation.error ? validation.error : ''}
              </div>
              {confirmDisable ? (
                <div className="flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  <span>Turn off auto-refill?</span>
                  <Button disabled={busy} onClick={() => void disable()} size="sm" type="button" variant="outline">
                    Turn off
                  </Button>
                  <Button disabled={busy} onClick={() => setConfirmDisable(false)} size="sm" type="button" variant="ghost">
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  disabled={busy}
                  onClick={() => setConfirmDisable(true)}
                  size="sm"
                  tabIndex={editing ? undefined : -1}
                  type="button"
                  variant="outline"
                >
                  Disable
                </Button>
              )}
              {/* Refusal stays INSIDE the reserved layer so it never pushes Usage. */}
              <BillingRefusalInline refusal={refusal} />
            </div>
            {/* VIEW layer — success feedback overlaid in the same cell when not editing. */}
            {!editing && message && (
              <div className="[grid-area:stack]">
                <InlineMessage kind={message.kind}>{message.text}</InlineMessage>
              </div>
            )}
          </div>
        </div>
        {/* Action column swaps Manage ↔ Save/Cancel in place (top-aligned, no move). */}
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
          {row.pill && <Pill tone={row.pill.tone}>{row.pill.label}</Pill>}
          {editing ? (
            <>
              <Button disabled={busy || !validation.values} onClick={() => void save()} size="sm" type="button">
                {busy ? 'Saving…' : 'Save'}
              </Button>
              <Button disabled={busy} onClick={cancelEdit} size="sm" type="button" variant="outline">
                Cancel
              </Button>
            </>
          ) : (
            <Button onClick={openEdit} size="sm" type="button" variant="outline">
              Manage
            </Button>
          )}
        </div>
      </div>
    </div>
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
            className="h-8 w-24"
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

function BillingRefusalInline({ refusal }: { refusal: BillingRefusal | null }) {
  const stepUp = useStepUpFlow()

  if (!refusal) {
    return null
  }

  const resolved = resolveRefusal(refusal)
  const portalUrl = resolved.action.type === 'portal' ? resolved.action.url : undefined

  return (
    <div className="mt-2 flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
      <span>
        <span className="font-medium text-foreground">{resolved.title}:</span> {resolved.message}
      </span>
      {resolved.action.type === 'step_up' && <StepUpInlineAction flow={stepUp} />}
      {portalUrl && (
        <Button onClick={() => openExternal(portalUrl)} size="sm" type="button" variant="outline">
          Open portal
          <ExternalLink className="size-3.5" />
        </Button>
      )}
    </div>
  )
}

function StepUpInlineAction({ flow }: { flow: ReturnType<typeof useStepUpFlow> }) {
  if (flow.verification) {
    return (
      <span className="inline-flex min-w-0 flex-wrap items-center gap-2">
        <span className="font-mono text-[0.72rem] font-semibold text-foreground">{flow.verification.code}</span>
        <Button onClick={flow.openVerification} size="sm" type="button" variant="outline">
          Open verification page
          <ExternalLink className="size-3.5" />
        </Button>
      </span>
    )
  }

  if (flow.message) {
    return (
      <span className="inline-flex min-w-0 flex-wrap items-center gap-2">
        <span>
          {flow.message.title}: {flow.message.text}
        </span>
        <Button onClick={flow.dismiss} size="sm" type="button" variant="outline">
          Dismiss
        </Button>
      </span>
    )
  }

  if (flow.phase === 'waiting') {
    return <span>Waiting for verification link…</span>
  }

  return (
    <Button onClick={() => void flow.start()} size="sm" type="button" variant="outline">
      Verify to continue
    </Button>
  )
}

function InlineMessage({ children, kind }: { children: string; kind: 'error' | 'success' }) {
  return (
    <div
      className={cn(
        'mt-2 text-[length:var(--conversation-caption-font-size)]',
        kind === 'error' ? 'text-destructive' : 'text-(--ui-text-tertiary)'
      )}
    >
      {children}
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
  const fixture =
    import.meta.env.DEV && fixtureName && fixtureName !== 'live' ? billingDevFixtures[fixtureName] : undefined

  const [subView, setSubView] = useRouteEnumParam<BillingSubView>('bview', BILLING_VIEWS, 'overview')

  const billingState = useBillingState(!fixture)
  const subscriptionState = useSubscriptionState(!fixture)
  const billingResult = fixture?.billing ?? billingState.data
  const subscriptionResult = fixture?.subscription ?? subscriptionState.data
  const view = deriveBillingView(billingResult, subscriptionResult)
  const billing = billingResult?.ok ? billingResult.data : undefined
  const usageUpdatedAt = oldestUpdatedAt(billingState.dataUpdatedAt, subscriptionState.dataUpdatedAt)
  const usageIsFetching = billingState.isFetching || subscriptionState.isFetching

  const refreshUsage = () => {
    void Promise.all([billingState.refetch(), subscriptionState.refetch()])
  }

  const paymentRow = view.accountRows.find(row => row.id === 'payment_method')
  const topupRow = view.accountRows.find(row => row.id === 'buy_credits')
  const refillRow = view.accountRows.find(row => row.id === 'auto_reload')

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

  return <BillingSettingsContent fixtureName={fixtureName} onFixtureChange={setFixtureName} />
}

export function BillingSettings() {
  if (import.meta.env.DEV) {
    return <BillingSettingsWithDevFixtures />
  }

  return <BillingSettingsContent />
}

function clampAmount(raw: string, billing: Pick<BillingStateResponse, 'max_usd' | 'min_usd'>): string {
  const amount = parseAmount(raw)

  if (amount == null) {
    return ''
  }

  const min = parseAmount(billing.min_usd)
  const max = parseAmount(billing.max_usd)
  const clampedMin = min == null ? amount : Math.max(min, amount)
  const clamped = max == null ? clampedMin : Math.min(max, clampedMin)

  return formatAmountForRequest(clamped)
}

function parseAmount(value?: null | number | string): null | number {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }

  if (typeof value !== 'string') {
    return null
  }

  const parsed = Number(value.replace(/[$,\s]/g, ''))

  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function formatAmountForRequest(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')
}

function oldestUpdatedAt(...timestamps: number[]): number {
  const populated = timestamps.filter(timestamp => timestamp > 0)

  return populated.length > 0 ? Math.min(...populated) : Date.now()
}

function initialAutoReloadAmount(...candidates: Array<null | string | undefined>): string {
  for (const candidate of candidates) {
    const amount = parseAmount(candidate)

    if (amount != null) {
      return formatAmountForRequest(amount)
    }
  }

  return ''
}

function validateAutoReloadInputs(
  thresholdRaw: string,
  reloadToRaw: string,
  bounds: Pick<BillingStateResponse, 'max_usd' | 'min_usd'>
): { error?: string; values?: { reloadTo: string; threshold: string } } {
  const threshold = validateBillingAmount('Threshold', thresholdRaw, bounds)

  if (threshold.error || threshold.amount == null) {
    return { error: threshold.error }
  }

  const reloadTo = validateBillingAmount('Reload-to', reloadToRaw, bounds)

  if (reloadTo.error || reloadTo.amount == null) {
    return { error: reloadTo.error }
  }

  if (reloadTo.amount <= threshold.amount) {
    return { error: 'Reload-to amount must be greater than the threshold.' }
  }

  return {
    values: {
      reloadTo: formatAmountForRequest(reloadTo.amount),
      threshold: formatAmountForRequest(threshold.amount)
    }
  }
}

function validateBillingAmount(
  label: string,
  raw: string,
  bounds: Pick<BillingStateResponse, 'max_usd' | 'min_usd'>
): { amount?: number; error?: string } {
  const cleaned = raw.trim().replace(/^\$/, '').trim()

  if (!cleaned || !/^\d+(\.\d{1,2})?$/.test(cleaned)) {
    return { error: `${label}: enter a dollar amount with at most 2 decimal places.` }
  }

  const amount = Number(cleaned)

  if (!(amount > 0)) {
    return { error: `${label}: amount must be greater than $0.` }
  }

  const min = parseAmount(bounds.min_usd)

  if (min != null && amount < min) {
    return { error: `${label}: minimum is ${formatMoney(min)}.` }
  }

  const max = parseAmount(bounds.max_usd)

  if (max != null && amount > max) {
    return { error: `${label}: maximum is ${formatMoney(max)}.` }
  }

  return { amount }
}

function formatMoney(value?: null | number | string): string {
  const amount = parseAmount(value)

  if (amount == null) {
    return EMPTY_BILLING_VALUE
  }

  return new Intl.NumberFormat(undefined, {
    currency: 'USD',
    maximumFractionDigits: amount % 1 === 0 ? 0 : 2,
    minimumFractionDigits: amount % 1 === 0 ? 0 : 2,
    style: 'currency'
  }).format(amount)
}
