import { useQuery } from '@tanstack/react-query'

import { fmtDate } from '@/lib/time'

import type { BillingRefusal, BillingResult } from './api'
import { useBillingApi } from './api'
import { resolveRefusal } from './errors'
import type { BillingStateResponse, SubscriptionStateResponse, SubscriptionTierOption, UsageModelData } from './types'

export const EMPTY_BILLING_VALUE = '—'
export const FALLBACK_PORTAL_BILLING_URL = 'https://portal.nousresearch.com/billing'
export const FALLBACK_PORTAL_URL = 'https://portal.nousresearch.com'

// The billing endpoint is the authoritative source of truth for balance / cap /
// plan — the inference `x-nous-credits-*` headers are best-effort and can drift
// out of sync (notably in team/org accounts where another member's spend moves
// the shared balance without ever touching THIS client's headers). So the page
// never trusts a cache: `staleTime: 0` + `refetchOnMount: 'always'` force a
// fresh fetch every time it opens or regains focus, and it keeps polling every
// 30s while mounted (react-query only ticks an active observer; it pauses when
// the window is backgrounded — refetchIntervalInBackground defaults to false).
// A `credits.*` notice crossing additionally invalidates ['billing','state'] to
// pull the change in immediately rather than waiting for the next poll tick.
const BILLING_QUERY_OPTIONS = {
  refetchInterval: 30_000,
  refetchOnMount: 'always',
  refetchOnWindowFocus: true,
  retry: false,
  staleTime: 0
} as const

export interface BillingSummaryItemView {
  label: 'Auto-refill' | 'Balance' | 'Plan'
  tone?: 'muted' | 'primary'
  value: string
}

export interface BillingNoticeView {
  action?: {
    label: string
    url: string
  }
  message: string
  title: string
  /** `warn` = an actionable blocker (e.g. no card); `info` = neutral guidance. */
  tone?: 'info' | 'warn'
}

export interface BillingRowActionView {
  disabled?: boolean
  label: string
  url?: string
}

export interface BillingChipView {
  disabled: boolean
  label: string
  /** When set, clicking the chip opens this URL externally. */
  url?: string
}

export interface BillingAccountRowView {
  action?: BillingRowActionView
  caption?: string
  chips?: BillingChipView[]
  description: string
  id: 'auto_reload' | 'buy_credits' | 'payment_method'
  /** The auto-refill row that edits its amounts in place (canonical-card enabled). */
  manageInApp?: true
  pill?: {
    label: string
    tone: 'muted' | 'primary'
  }
  secondaryPill?: string
  title: string
  value?: string
}

/**
 * A change scheduled at period end that `subscription.resume` can undo. A downgrade
 * names its target tier (and marks it in the grid); a cancellation has no target
 * (the whole plan lapses), so the grid shows no marker for it.
 */
export type PendingPlanTransition =
  | { kind: 'cancellation'; when: string }
  | { kind: 'downgrade'; tierName: string; when: string }

/**
 * The current-plan summary that replaces the old subscription row. Carries EITHER
 * one in-app `action` (View plans / Change plan) OR a portal `link` ("Adjust plan
 * ↗"), never both — a discriminated pair so consumers don't guard for the impossible
 * "both present" / "neither present" cases.
 */
export type BillingPlanCardView = {
  caption: string
  /** A scheduled downgrade / cancellation waiting at period end (drives the undo). */
  pending?: PendingPlanTransition
  price?: string
  tierName: string
} & ({ action: { label: string }; link?: undefined } | { action?: undefined; link: { label: string; url: string } })

interface BillingPlanTierBase {
  creditsDisplay?: string
  name: string
  priceDisplay: string
  tierId: string
}

/**
 * One card in the `bview=plans` grid, discriminated by `state`: `upgrade` carries its
 * portal `action`; `downgrade` is actionable IN-APP (the flow keys off `tierId`, so it
 * needs no url/caption); `scheduled` is the inert pending-downgrade target; `current`
 * is inert. The union lets consumers read `action` without defensive `?.`.
 */
export type BillingPlanTierView =
  | (BillingPlanTierBase & { state: 'current' })
  | (BillingPlanTierBase & { state: 'downgrade' })
  | (BillingPlanTierBase & { state: 'scheduled' })
  | (BillingPlanTierBase & { action: { label: string; url: string }; state: 'upgrade' })

export interface BillingUsageRowView {
  bar?: {
    label: string
    state: 'danger' | 'neutral' | 'ok'
    tone: 'cap' | 'subscription' | 'topup'
    track?: 'danger'
    value: number
  }
  caption: string
  id: 'monthly_cap' | 'subscription_credits' | 'topup_credits'
  title: string
  value: string
}

export interface BillingView {
  notice?: BillingNoticeView
  /** Payment section row. Absent outside the normal (logged-in) state. */
  paymentRow?: BillingAccountRowView
  /** Current-plan card (Plan section). Absent until billing.state resolves. */
  plan?: BillingPlanCardView
  /** Automatic-refill section row. */
  refillRow?: BillingAccountRowView
  status: 'loading' | 'logged_out' | 'normal' | 'refusal'
  summary: BillingSummaryItemView[]
  /** Live tier catalog for the plans sub-view (empty when unavailable). */
  tiers: BillingPlanTierView[]
  /** One-time top-up section row. */
  topupRow?: BillingAccountRowView
  usageRows: BillingUsageRowView[]
}

export function useBillingState(enabled = true) {
  const api = useBillingApi()

  return useQuery({
    ...BILLING_QUERY_OPTIONS,
    enabled,
    queryFn: () => api.fetchBillingState(),
    queryKey: ['billing', 'state']
  })
}

export function useSubscriptionState(enabled = true) {
  const api = useBillingApi()

  return useQuery({
    ...BILLING_QUERY_OPTIONS,
    enabled,
    queryFn: () => api.fetchSubscriptionState(),
    queryKey: ['billing', 'subscription']
  })
}

export function deriveBillingView(
  stateResult?: BillingResult<BillingStateResponse>,
  subscriptionResult?: BillingResult<SubscriptionStateResponse>
): BillingView {
  if (!stateResult) {
    return {
      status: 'loading',
      summary: emptySummary(),
      tiers: [],
      usageRows: []
    }
  }

  if (!stateResult.ok) {
    return {
      notice: refusalNotice(stateResult.refusal),
      status: 'refusal',
      summary: emptySummary(),
      tiers: [],
      usageRows: []
    }
  }

  const billing = stateResult.data
  const subscription = subscriptionResult?.ok ? subscriptionResult.data : null

  if (!billing.logged_in || subscription?.logged_in === false) {
    return {
      notice: {
        action: { label: 'Open portal ↗', url: billing.portal_url ?? subscription?.portal_url ?? FALLBACK_PORTAL_URL },
        message: 'Run /portal in the TUI or open the Nous portal to connect your account.',
        title: 'Connect your Nous account'
      },
      status: 'logged_out',
      summary: emptySummary(),
      tiers: [],
      usageRows: []
    }
  }

  // One "can change plans in-app" verdict, shared by the plan card (button vs portal
  // link) and the grid (whether upgrade tiles are actionable) so the invariant lives
  // in one place.
  const capable = plansCapable(subscription, subscriptionResult)
  // Computed once and threaded to both the card (caption + undo) and the grid
  // (Scheduled marker), so the two never disagree about what's pending.
  const pending = pendingTransition(subscription?.current)
  const tiers = derivePlanTiers(subscription, billing.portal_url, capable, pending)

  return {
    notice: noCardNotice(billing),
    paymentRow: paymentMethodRow(billing),
    plan: derivePlanCard(billing, subscription, subscriptionResult, tiers, capable, pending),
    refillRow: autoReloadRow(billing),
    status: 'normal',
    summary: [
      { label: 'Balance', value: displayBalance(billing) },
      { label: 'Plan', value: displayPlan(subscription, billing.usage) },
      {
        label: 'Auto-refill',
        tone: billing.auto_reload?.enabled ? 'primary' : billing.auto_reload ? 'muted' : undefined,
        value: billing.auto_reload ? (billing.auto_reload.enabled ? 'Enabled' : 'Off') : EMPTY_BILLING_VALUE
      }
    ],
    tiers,
    topupRow: buyCreditsRow(billing),
    usageRows: deriveUsageRows(billing, subscription)
  }
}

export function buildManageSubscriptionUrl(
  subscription?: null | Pick<SubscriptionStateResponse, 'org_id' | 'portal_url'>,
  fallbackPortalUrl?: null | string,
  // Optional tier to pre-select on the portal, appended as `plan=<tierId>`
  // (validated server-side by the NAS reader, draft #748).
  tierId?: null | string
): string {
  // The hard-coded portal is the LAST-RESORT origin, not a bare early return:
  // org_id / plan must still be applied to it so a null portal_url never silently
  // strips the params that route the user to the right org + pre-selected tier.
  const portalUrls = [subscription?.portal_url, fallbackPortalUrl, FALLBACK_PORTAL_BILLING_URL].filter(
    (url): url is string => typeof url === 'string' && url.length > 0
  )

  for (const portalUrl of portalUrls) {
    try {
      const url = new URL('/manage-subscription', new URL(portalUrl).origin)

      if (subscription?.org_id) {
        url.searchParams.set('org_id', subscription.org_id)
      }

      if (tierId) {
        url.searchParams.set('plan', tierId)
      }

      return url.toString()
    } catch {
      // Try the next candidate; malformed portal URLs should not break settings.
    }
  }

  return FALLBACK_PORTAL_BILLING_URL
}

export function formatBillingDate(value?: null | string): string {
  if (!value) {
    return EMPTY_BILLING_VALUE
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return EMPTY_BILLING_VALUE
  }

  return fmtDate.format(date)
}

function emptySummary(): BillingSummaryItemView[] {
  return [
    { label: 'Balance', value: EMPTY_BILLING_VALUE },
    { label: 'Plan', value: EMPTY_BILLING_VALUE },
    { label: 'Auto-refill', value: EMPTY_BILLING_VALUE }
  ]
}

function refusalNotice(refusal: BillingRefusal): BillingNoticeView {
  const resolved = resolveRefusal(refusal)
  const portalUrl = resolved.action.type === 'portal' ? resolved.action.url : undefined

  return {
    action: portalUrl ? { label: 'Open portal ↗', url: portalUrl } : undefined,
    message: resolved.message,
    title: resolved.title,
    tone: 'warn'
  }
}

// A logged-in account with no card can't buy credits or manage auto-refill, and
// every one of those controls disables silently — so lead the page with a single
// warn banner that names the blocker and links straight to the fix.
function noCardNotice(billing: BillingStateResponse): BillingNoticeView | undefined {
  if (billing.card) {
    return undefined
  }

  return {
    action: { label: 'Add card ↗', url: billing.portal_url ?? FALLBACK_PORTAL_BILLING_URL },
    message: 'Buying top-up credits and auto-refill stay disabled until a card is on file. Add one on the portal.',
    title: 'No payment method on file',
    tone: 'warn'
  }
}

// The active tier from the UNFILTERED catalog — a grandfathered current tier is
// is_enabled:false, so it must still resolve here (by is_current or matching id).
function findCurrentTier(subscription: null | SubscriptionStateResponse): SubscriptionTierOption | undefined {
  const current = subscription?.current

  return subscription?.tiers?.find(tier => tier.is_current || tier.tier_id === current?.tier_id)
}

// Whether this account can change plans in-app: a personal (non-team) subscription
// the server says the user can change, whose payload actually loaded.
function plansCapable(
  subscription: null | SubscriptionStateResponse,
  subscriptionResult: BillingResult<SubscriptionStateResponse> | undefined
): boolean {
  if (!subscription || (subscriptionResult && !subscriptionResult.ok)) {
    return false
  }

  return subscription.context !== 'team' && Boolean(subscription.can_change_plan)
}

// Monthly credits are dollars; NAS sends a bare decimal string. Never render a
// bare number — always "$110 credits/mo" (mirrors the retired subscriptionTierChips).
function creditsPerMonthDisplay(monthlyCredits: null | string): string | undefined {
  const credits = Number((monthlyCredits ?? '').replace(/,/g, ''))

  return Number.isFinite(credits) && credits > 0 ? `$${credits.toLocaleString('en-US')} credits/mo` : undefined
}

/**
 * A monthly-credits delta from a plan-change preview. NAS sends a bare dollar
 * decimal ("-88"); credits are DOLLARS, so render it as signed dollars
 * ("−$88/mo"), never the raw number. Zero / absent → null so the caller hides
 * the line entirely.
 */
export function formatMonthlyCreditsDelta(delta?: null | string): null | string {
  const amount = parseAmount(delta)

  if (amount == null || amount === 0) {
    return null
  }

  return `${amount < 0 ? '−' : '+'}${formatMoney(Math.abs(amount))}/mo`
}

/**
 * The current-plan card. It offers the in-app "View plans" / "Change plan" button
 * ONLY when the account is plans-capable AND the grid has an actual UPGRADE to offer
 * — a top-tier subscriber (only downgrades / current below them) would otherwise open
 * a grid with nothing to do. In every no-button case (teams, non-changers, refused
 * subscription, top tier, empty catalog) the card ALWAYS carries the portal
 * escape-hatch link so the user is never stranded on an info-only card.
 */
function derivePlanCard(
  billing: BillingStateResponse,
  subscription: null | SubscriptionStateResponse,
  subscriptionResult: BillingResult<SubscriptionStateResponse> | undefined,
  tiers: BillingPlanTierView[],
  capable: boolean,
  pending: PendingPlanTransition | undefined
): BillingPlanCardView {
  const current = subscription?.current
  const tierName = current?.tier_name ?? billing.usage?.plan_name ?? 'Free'
  // Price resolves against the UNFILTERED catalog so a grandfathered current tier
  // still shows its price.
  const price = findCurrentTier(subscription)?.dollars_per_month_display
  const renewal = formatBillingDate(current?.cycle_ends_at ?? billing.usage?.renews_at)
  const unavailable = subscriptionResult ? !subscriptionResult.ok : false

  const caption = unavailable
    ? 'Subscription details are unavailable; opening the portal is still available.'
    : pending
      ? pending.kind === 'downgrade'
        ? `Changes to ${pending.tierName} on ${pending.when}.`
        : `Cancels on ${pending.when}.`
      : current
        ? `Renews ${renewal}`
        : 'No active subscription — paid models draw down top-up credits.'

  // Actionable = a paid tier above (upgrade) or an in-app downgrade below the current
  // one. Ticket 11 counts downgrades (they act in-app, so they carry no `action`); a
  // top-tier subscriber with neither still gets the portal-link fallback below.
  const hasActionableTier = tiers.some(tier => tier.state === 'upgrade' || tier.state === 'downgrade')

  if (capable && hasActionableTier) {
    return { action: { label: current ? 'Change plan' : 'View plans' }, caption, pending, price, tierName }
  }

  return {
    caption,
    // No in-app action → always hand off to the portal so the user isn't stranded.
    link: {
      label: 'Adjust plan ↗',
      url: buildManageSubscriptionUrl(subscription, subscription?.portal_url ?? billing.portal_url)
    },
    pending,
    price,
    tierName
  }
}

// The change scheduled at period end (undoable via subscription.resume). NAS may
// carry a pending downgrade (`pending_downgrade_*`, with a target tier name) and/or a
// scheduled cancellation (`cancel_at_period_end` + `cancellation_effective_*`).
// Precedence: a downgrade WINS if both are somehow set — it names a concrete target
// tier, the stronger, more specific signal, and is what the grid marks.
function pendingTransition(
  current: null | undefined | NonNullable<SubscriptionStateResponse['current']>
): PendingPlanTransition | undefined {
  if (current?.pending_downgrade_tier_name && current.pending_downgrade_at) {
    return {
      kind: 'downgrade',
      tierName: current.pending_downgrade_tier_name,
      when: current.pending_downgrade_display ?? formatBillingDate(current.pending_downgrade_at)
    }
  }

  if (current?.cancel_at_period_end && current.cancellation_effective_at) {
    return {
      kind: 'cancellation',
      when: current.cancellation_effective_display ?? formatBillingDate(current.cancellation_effective_at)
    }
  }

  return undefined
}

/**
 * The plans-grid catalog. Each card's state depends on its order relative to the
 * current tier: current = inert marker; higher = "Choose ↗" opening the portal with
 * the tier pre-selected; lower = an in-app "Downgrade" (chargeless, scheduled via the
 * gateway). The already-scheduled downgrade target renders as an inert "Scheduled"
 * marker; other lower tiers stay actionable (picking one reschedules). With no active
 * subscription the lowest-order ($0 / free) tier stands in as the current plan, so
 * there is no "subscribe to Free" upgrade and no downgrade state.
 *
 * Empty unless `capable`: only a plans-capable account gets actionable tiles, and the
 * plan card / deep-link gate on the same verdict — so the grid never mints an
 * upgrade action nobody may take. `fallbackPortalUrl` (billing.portal_url) backs the
 * Choose URLs when the subscription payload has no portal_url, so org_id + plan are
 * never dropped.
 */
function derivePlanTiers(
  subscription: null | SubscriptionStateResponse,
  fallbackPortalUrl: null | string,
  capable: boolean,
  pending: PendingPlanTransition | undefined
): BillingPlanTierView[] {
  if (!capable || !subscription) {
    return []
  }

  const allTiers = subscription.tiers ?? []
  const current = subscription.current
  const explicitCurrent = findCurrentTier(subscription)

  // The grid shows the enabled catalog plus the grandfathered current tier (so it
  // still renders as the inert "Current plan" card), sorted low→high.
  const gridTiers = allTiers
    .filter(tier => tier.is_enabled || tier.tier_id === explicitCurrent?.tier_id)
    .slice()
    .sort((a, b) => a.tier_order - b.tier_order)

  if (gridTiers.length === 0) {
    return []
  }

  // No active subscription → the lowest-order ($0 / free) tier stands in as the
  // current plan: inert, never a "subscribe to Free" upgrade, and (being lowest)
  // never leaving room for a downgrade.
  const currentTier = explicitCurrent ?? (current == null ? gridTiers[0] : undefined)
  const currentOrder = currentTier?.tier_order
  const manageBase = subscription.portal_url ?? fallbackPortalUrl
  // Only a downgrade has a target tier to mark; a cancellation has none.
  const pendingName = pending?.kind === 'downgrade' ? pending.tierName : null

  return gridTiers.map((tier): BillingPlanTierView => {
    const base: BillingPlanTierBase = {
      creditsDisplay: creditsPerMonthDisplay(tier.monthly_credits),
      name: tier.name,
      priceDisplay: tier.dollars_per_month_display,
      tierId: tier.tier_id
    }

    if (currentTier && tier.tier_id === currentTier.tier_id) {
      return { ...base, state: 'current' }
    }

    // A scheduled downgrade target is inert (matched by name — NAS sends no id for
    // the pending target). Name is a safe key: SubscriptionTypes.name is @unique in
    // NAS, so two tiers can't collide. Checked before the downgrade branch since the
    // target IS a lower tier.
    if (pendingName && tier.name === pendingName) {
      return { ...base, state: 'scheduled' }
    }

    // Downgrade = strictly below the current tier's order → an in-app chargeless
    // change (the PlanCard wires the confirm flow by tierId).
    if (currentOrder != null && tier.tier_order < currentOrder) {
      return { ...base, state: 'downgrade' }
    }

    return {
      ...base,
      action: { label: 'Choose ↗', url: buildManageSubscriptionUrl(subscription, manageBase, tier.tier_id) },
      state: 'upgrade'
    }
  })
}

function paymentMethodRow(billing: BillingStateResponse): BillingAccountRowView {
  const portalUrl = billing.portal_url ?? FALLBACK_PORTAL_BILLING_URL
  const card = billing.card

  if (!card) {
    // No card → a single "Add payment method" link, the way every other app does
    // it. The reason (buys/auto-refill are blocked) already leads the page as a
    // notice, so the row stays a bare call-to-action with no redundant status text.
    return {
      action: { label: 'Add payment method', url: portalUrl },
      description: '',
      id: 'payment_method',
      title: 'Payment method'
    }
  }

  return {
    action: { label: 'Update', url: portalUrl },
    description: 'Manage the card used for top-ups and subscription renewals.',
    id: 'payment_method',
    title: 'Payment method',
    value: `${capitalize(card.brand)} •••• ${card.last4}${provenanceSuffix(card.resolved_via)}`
  }
}

function buyCreditsRow(billing: BillingStateResponse): BillingAccountRowView {
  if (!billing.card) {
    // The no-card blocker is already spelled out by the page-level warn banner
    // (noCardNotice); repeating it here — emoji and all — just clutters the row,
    // so keep the plain "what buying does" line and let the controls sit disabled.
    return {
      action: { disabled: true, label: 'Buy' },
      chips: billing.charge_presets.map(amount => ({ disabled: true, label: formatMoney(amount) })),
      description: 'A single charge on your card, added to your balance today.',
      id: 'buy_credits',
      title: 'Buy credits now'
    }
  }

  const disabledReason = buyCreditsDisabledReason(billing)

  if (disabledReason) {
    return {
      description: disabledReason,
      id: 'buy_credits',
      title: 'Buy credits now'
    }
  }

  return {
    action: { disabled: true, label: 'Buy' },
    chips: billing.charge_presets.map(amount => ({ disabled: true, label: formatMoney(amount) })),
    description: 'A single charge on your card, added to your balance today.',
    id: 'buy_credits',
    title: 'Buy credits now'
  }
}

// The generic first sentence shared by the off / absent / divergent states,
// where the concrete amounts aren't the headline. The configured state overrides
// this with the disambiguating "Charges $X … below $Y." sentence (spec §8).
const AUTO_REFILL_GENERIC = 'Keep your balance topped up when it drops below your threshold.'

function autoReloadRow(billing: BillingStateResponse): BillingAccountRowView {
  const autoReload = billing.auto_reload

  if (!autoReload) {
    return {
      action: { disabled: true, label: 'Manage' },
      caption: 'Manage auto-refill from the portal.',
      description: AUTO_REFILL_GENERIC,
      id: 'auto_reload',
      pill: { label: EMPTY_BILLING_VALUE, tone: 'muted' },
      title: 'Refill when low'
    }
  }

  if (!autoReload.enabled) {
    return {
      caption: 'Turn on auto-refill from the portal',
      description: AUTO_REFILL_GENERIC,
      id: 'auto_reload',
      pill: { label: 'Off', tone: 'muted' },
      title: 'Refill when low'
    }
  }

  // A null card (gateway emits it for a missing/unknown-kind card) falls through to
  // the default enabled path below — the same treatment as a canonical card.
  if (autoReload.card?.kind === 'distinct') {
    const { brand, last4 } = autoReload.card
    const cardLabel = brand && last4 ? `${capitalize(brand)} ••${last4}` : 'a different card'
    const portalUrl = billing.portal_url ?? FALLBACK_PORTAL_BILLING_URL

    return {
      action: { label: 'Reconcile ↗', url: portalUrl },
      caption: `Auto-refill charges ${cardLabel} — reconcile on the portal`,
      description: AUTO_REFILL_GENERIC,
      id: 'auto_reload',
      pill: { label: 'Enabled', tone: 'primary' },
      title: 'Refill when low'
    }
  }

  const reloadTo = autoReload.reload_to_display || formatMoney(autoReload.reload_to_usd)
  const threshold = autoReload.threshold_display || formatMoney(autoReload.threshold_usd)

  return {
    action: { label: 'Manage' },
    // Numbers live in the first sentence (spec §8); the swap region below carries
    // the editable fields, so no redundant caption here.
    description: `Charges ${reloadTo} automatically when your balance falls below ${threshold}.`,
    id: 'auto_reload',
    // The only row that edits in place — AutoReloadRow keys its swap layout off this
    // flag rather than sniffing the action label.
    manageInApp: true,
    pill: { label: 'Enabled', tone: 'primary' },
    title: 'Refill when low'
  }
}

function deriveUsageRows(
  billing: BillingStateResponse,
  subscription: null | SubscriptionStateResponse
): BillingUsageRowView[] {
  const rows: BillingUsageRowView[] = []
  const current = subscription?.current
  const remaining = parseAmount(current?.credits_remaining)
  const monthly = parseAmount(current?.monthly_credits)
  const usage = subscription?.usage ?? billing.usage

  // Remaining can go slightly negative (usage settles after credits hit zero).
  // A raw "-$0.79 left" reads as broken — clamp to $0 and name the overage.
  const subscriptionValue =
    remaining != null && monthly != null
      ? remaining < 0
        ? `${formatMoney(0)} of ${formatMoney(monthly)} left · ${formatMoney(Math.abs(remaining))} over`
        : `${formatMoney(remaining)} of ${formatMoney(monthly)} left`
      : (usage?.subscription_remaining_display ?? usage?.plan_bar?.remaining_display ?? EMPTY_BILLING_VALUE)

  const remainingFraction = remaining != null && monthly != null && monthly > 0 ? remaining / monthly : null

  rows.push({
    bar:
      remainingFraction != null
        ? {
            label: 'Subscription credits remaining',
            state: remainingFraction <= 0.1 ? 'danger' : 'ok',
            tone: 'subscription',
            track: remaining != null && remaining <= 0 ? 'danger' : undefined,
            value: clamp01(remainingFraction)
          }
        : undefined,
    caption: `Resets ${formatBillingDate(current?.cycle_ends_at ?? usage?.renews_at)}`,
    id: 'subscription_credits',
    title: 'Subscription credits',
    value: subscriptionValue
  })

  const topupValue = topupCreditsValue(billing, usage)

  // No bar: top-ups have no denominator (the wire carries only the current
  // balance, and the pool is open-ended), so a fill fraction would be fiction.
  rows.push({
    caption: 'Does not expire',
    id: 'topup_credits',
    title: 'Top-up credits',
    value: topupValue
  })

  const cap = billing.monthly_cap

  if (cap && cap.limit_usd != null) {
    const limit = parseAmount(cap.limit_usd)
    const spent = parseAmount(cap.spent_this_month_usd) ?? 0
    const usedFraction = limit != null && limit > 0 ? spent / limit : null
    const value = `${cap.spent_display || formatMoney(spent)} of ${cap.limit_display || formatMoney(limit)} used`

    rows.push({
      bar:
        usedFraction != null
          ? {
              label: 'Monthly spend cap used',
              state: usedFraction >= 0.9 ? 'danger' : 'ok',
              tone: 'cap',
              track: usedFraction >= 1 ? 'danger' : undefined,
              value: clamp01(usedFraction)
            }
          : undefined,
      caption: cap.is_default_ceiling ? 'Default ceiling' : 'Monthly remote spending',
      id: 'monthly_cap',
      title: 'Monthly spend cap',
      value
    })
  }

  return rows
}

function displayBalance(billing: BillingStateResponse): string {
  return nonEmpty(billing.balance_display) ?? formatMoney(billing.balance_usd)
}

function displayPlan(subscription: null | SubscriptionStateResponse, usage?: UsageModelData): string {
  const current = subscription?.current
  const tier = current?.tier_name ?? usage?.plan_name

  if (!tier) {
    return EMPTY_BILLING_VALUE
  }

  const price = findCurrentTier(subscription)?.dollars_per_month_display

  return price ? `${tier} · ${price}/mo` : tier
}

function topupCreditsValue(billing: BillingStateResponse, usage?: UsageModelData): string {
  return (
    usage?.topup_remaining_display ??
    usage?.topup_bar?.remaining_display ??
    nonEmpty(billing.balance_display) ??
    formatMoney(billing.balance_usd)
  )
}

function buyCreditsDisabledReason(billing: BillingStateResponse): null | string {
  if (!billing.is_admin) {
    return resolveRefusal({ kind: 'role_required', message: '' }).message
  }

  if (!billing.cli_billing_enabled) {
    return resolveRefusal({ kind: 'cli_billing_disabled', message: '', portalUrl: billing.portal_url ?? undefined })
      .message
  }

  if (!billing.can_charge) {
    return resolveRefusal({ kind: 'remote_spending_disabled', message: '', portalUrl: billing.portal_url ?? undefined })
      .message
  }

  return null
}

function provenanceSuffix(resolvedVia?: null | string): string {
  if (!resolvedVia) {
    return ''
  }

  const labels: Record<string, string> = {
    autoRefill: 'auto-refill card',
    customerDefault: 'customer default',
    subPin: 'subscription card'
  }

  return ` - ${labels[resolvedVia] ?? resolvedVia}`
}

function capitalize(value: string): string {
  return value ? `${value.charAt(0).toUpperCase()}${value.slice(1)}` : value
}

function nonEmpty(value?: null | string): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value : undefined
}

function parseAmount(value?: null | number | string): null | number {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }

  if (typeof value !== 'string') {
    return null
  }

  const parsed = Number(value.replace(/[$,\s]/g, ''))

  return Number.isFinite(parsed) ? parsed : null
}

function formatMoney(value?: null | number | string): string {
  const amount = parseAmount(value)

  if (amount == null) {
    return EMPTY_BILLING_VALUE
  }

  // Pin en-US so the symbol is always "$" — the server's *_display strings
  // ("$996.47") sit next to these, and other locales render USD as "US$".
  return new Intl.NumberFormat('en-US', {
    currency: 'USD',
    maximumFractionDigits: amount % 1 === 0 ? 0 : 2,
    minimumFractionDigits: amount % 1 === 0 ? 0 : 2,
    style: 'currency'
  }).format(amount)
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.max(0, Math.min(1, value))
}
