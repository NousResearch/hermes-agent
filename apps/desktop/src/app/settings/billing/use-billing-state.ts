import { useQuery } from '@tanstack/react-query'

import { fmtDate } from '@/lib/time'

import type { BillingRefusal, BillingResult } from './api'
import { useBillingApi } from './api'
import { resolveRefusal } from './errors'
import type { BillingStateResponse, SubscriptionStateResponse, UsageModelData } from './types'

export const EMPTY_BILLING_VALUE = '—'
export const FALLBACK_PORTAL_BILLING_URL = 'https://portal.nousresearch.com/billing'
export const FALLBACK_PORTAL_URL = 'https://portal.nousresearch.com'

const BILLING_QUERY_OPTIONS = {
  refetchOnWindowFocus: true,
  retry: false,
  staleTime: 30_000
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
  id: 'auto_reload' | 'buy_credits' | 'payment_method' | 'subscription'
  pill?: {
    label: string
    tone: 'muted' | 'primary'
  }
  secondaryPill?: string
  title: string
  value?: string
}

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
  accountRows: BillingAccountRowView[]
  notice?: BillingNoticeView
  status: 'loading' | 'logged_out' | 'normal' | 'refusal'
  summary: BillingSummaryItemView[]
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
      accountRows: [],
      status: 'loading',
      summary: emptySummary(),
      usageRows: []
    }
  }

  if (!stateResult.ok) {
    return {
      accountRows: [],
      notice: refusalNotice(stateResult.refusal),
      status: 'refusal',
      summary: emptySummary(),
      usageRows: []
    }
  }

  const billing = stateResult.data
  const subscription = subscriptionResult?.ok ? subscriptionResult.data : null

  if (!billing.logged_in || subscription?.logged_in === false) {
    return {
      accountRows: [],
      notice: {
        action: { label: 'Open portal ↗', url: billing.portal_url ?? subscription?.portal_url ?? FALLBACK_PORTAL_URL },
        message: 'Run /portal in the TUI or open the Nous portal to connect your account.',
        title: 'Connect your Nous account'
      },
      status: 'logged_out',
      summary: emptySummary(),
      usageRows: []
    }
  }

  return {
    accountRows: deriveAccountRows(billing, subscription, subscriptionResult),
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
    usageRows: deriveUsageRows(billing, subscription)
  }
}

export function buildManageSubscriptionUrl(
  subscription?: null | Pick<SubscriptionStateResponse, 'org_id' | 'portal_url'>,
  fallbackPortalUrl?: null | string,
  tierId?: string
): string {
  const portalUrls = [subscription?.portal_url, fallbackPortalUrl].filter(
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

export function formatUsageUpdatedAgo(updatedAt: number, now: number): string {
  const elapsedSeconds = Math.max(0, Math.floor((now - updatedAt) / 1000))

  if (elapsedSeconds < 1) {
    return 'just now'
  }

  if (elapsedSeconds < 60) {
    return `${elapsedSeconds}s ago`
  }

  const elapsedMinutes = Math.floor(elapsedSeconds / 60)

  if (elapsedMinutes < 60) {
    return `${elapsedMinutes}m ago`
  }

  return `${Math.floor(elapsedMinutes / 60)}h ago`
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
    title: resolved.title
  }
}

function deriveAccountRows(
  billing: BillingStateResponse,
  subscription: null | SubscriptionStateResponse,
  subscriptionResult?: BillingResult<SubscriptionStateResponse>
): BillingAccountRowView[] {
  return [
    paymentMethodRow(billing),
    subscriptionRow(billing, subscription, subscriptionResult),
    buyCreditsRow(billing),
    autoReloadRow(billing)
  ]
}

function paymentMethodRow(billing: BillingStateResponse): BillingAccountRowView {
  const portalUrl = billing.portal_url ?? FALLBACK_PORTAL_BILLING_URL
  const card = billing.card

  if (!card) {
    return {
      action: { label: 'Update ↗', url: portalUrl },
      description: 'Add a payment method on the portal before buying top-up credits.',
      id: 'payment_method',
      title: 'Payment method',
      value: 'No card on file'
    }
  }

  return {
    action: { label: 'Update ↗', url: portalUrl },
    description: 'Manage the card used for top-ups and subscription renewals.',
    id: 'payment_method',
    title: 'Payment method',
    value: `${capitalize(card.brand)} •••• ${card.last4}${provenanceSuffix(card.resolved_via)}`
  }
}

/**
 * Tier catalog as chips for accounts that can change plans; the current plan is
 * inert, every other opens the portal where the change/start happens, deep-linked
 * to that tier via `?plan=`.
 */
function subscriptionTierChips(
  subscription: null | SubscriptionStateResponse,
  fallbackPortalUrl?: null | string
): BillingChipView[] | undefined {
  // Teams have no personal subscription to sell into.
  if (!subscription?.can_change_plan || subscription.context === 'team') {
    return undefined
  }

  const tiers = (subscription.tiers ?? [])
    .filter(tier => tier.is_enabled && tier.tier_order > 0)
    .sort((a, b) => a.tier_order - b.tier_order)

  if (tiers.length === 0) {
    return undefined
  }

  return tiers.map(tier => {
    // Monthly credits are dollars; NAS sends a bare decimal string.
    const credits = Number((tier.monthly_credits ?? '').replace(/,/g, ''))
    const suffix = Number.isFinite(credits) && credits > 0 ? ` · $${credits.toLocaleString('en-US')} credits/mo` : ''
    const label = `${tier.name} · ${tier.dollars_per_month_display}/mo${suffix}`

    return tier.is_current
      ? { disabled: true, label: `✓ ${label}` }
      : { disabled: false, label, url: buildManageSubscriptionUrl(subscription, fallbackPortalUrl, tier.tier_id) }
  })
}

function subscriptionRow(
  billing: BillingStateResponse,
  subscription: null | SubscriptionStateResponse,
  subscriptionResult?: BillingResult<SubscriptionStateResponse>
): BillingAccountRowView {
  const fallbackPortalUrl = subscription?.portal_url ?? billing.portal_url
  const manageUrl = buildManageSubscriptionUrl(subscription, fallbackPortalUrl)
  const current = subscription?.current
  const fallbackPlan = billing.usage?.plan_name ?? EMPTY_BILLING_VALUE
  const value = current?.tier_name ?? fallbackPlan
  const renewal = formatBillingDate(current?.cycle_ends_at ?? billing.usage?.renews_at)
  const unavailable = subscriptionResult && !subscriptionResult.ok
  const chips = subscriptionTierChips(subscription, fallbackPortalUrl)

  return {
    action: { label: 'Adjust plan ↗', url: manageUrl },
    caption: unavailable
      ? 'Subscription details are unavailable; opening the portal is still available.'
      : `Renews ${renewal}`,
    chips,
    description:
      !current && chips
        ? 'Paid models need a subscription — pick a plan to start it on the portal.'
        : 'Review your plan and change it from the billing portal.',
    id: 'subscription',
    secondaryPill: 'opens portal',
    title: 'Subscription',
    value
  }
}

function buyCreditsRow(billing: BillingStateResponse): BillingAccountRowView {
  if (!billing.card) {
    return {
      action: { disabled: true, label: 'Buy' },
      chips: billing.charge_presets.map(amount => ({ disabled: true, label: formatMoney(amount) })),
      description: resolveRefusal({
        kind: 'no_payment_method',
        message: '',
        portalUrl: billing.portal_url ?? undefined
      }).message,
      id: 'buy_credits',
      title: 'Buy credits'
    }
  }

  const disabledReason = buyCreditsDisabledReason(billing)

  if (disabledReason) {
    return {
      description: disabledReason,
      id: 'buy_credits',
      title: 'Buy credits'
    }
  }

  return {
    action: { disabled: true, label: 'Buy' },
    chips: billing.charge_presets.map(amount => ({ disabled: true, label: formatMoney(amount) })),
    description: 'Add top-up credits for agent runs outside your plan.',
    id: 'buy_credits',
    title: 'Buy credits'
  }
}

function autoReloadRow(billing: BillingStateResponse): BillingAccountRowView {
  const autoReload = billing.auto_reload

  if (!autoReload) {
    return {
      action: { disabled: true, label: 'Manage' },
      caption: 'Manage auto-refill from the portal.',
      description: 'Keep your balance topped up when it drops below your threshold.',
      id: 'auto_reload',
      pill: { label: EMPTY_BILLING_VALUE, tone: 'muted' },
      title: 'Auto-refill'
    }
  }

  if (!autoReload.enabled) {
    return {
      caption: 'Turn on auto-refill from the portal',
      description: 'Keep your balance topped up when it drops below your threshold.',
      id: 'auto_reload',
      pill: { label: 'Off', tone: 'muted' },
      title: 'Auto-refill'
    }
  }

  if (autoReload.card.kind === 'distinct') {
    const { brand, last4 } = autoReload.card
    const cardLabel = brand && last4 ? `${capitalize(brand)} ••${last4}` : 'a different card'
    const portalUrl = billing.portal_url ?? FALLBACK_PORTAL_BILLING_URL

    return {
      action: { label: 'Reconcile ↗', url: portalUrl },
      caption: `Auto-refill charges ${cardLabel} — reconcile on the portal`,
      description: 'Keep your balance topped up when it drops below your threshold.',
      id: 'auto_reload',
      pill: { label: 'Enabled', tone: 'primary' },
      title: 'Auto-refill'
    }
  }

  return {
    action: { label: 'Manage' },
    caption: `Refill ${autoReload.reload_to_display || formatMoney(autoReload.reload_to_usd)} when balance falls below ${
      autoReload.threshold_display || formatMoney(autoReload.threshold_usd)
    }`,
    description: 'Keep your balance topped up when it drops below your threshold.',
    id: 'auto_reload',
    pill: { label: 'Enabled', tone: 'primary' },
    title: 'Auto-refill'
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

  const currentTier = subscription?.tiers.find(t => t.is_current || t.tier_id === current?.tier_id)
  const price = currentTier?.dollars_per_month_display

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
