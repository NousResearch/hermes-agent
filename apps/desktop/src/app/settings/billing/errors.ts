import type { BillingRefusal } from './api'

export interface BillingRefusalPresentation {
  action: { type: 'none' } | { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
  message: string
  title: string
}

const portalAction = (url?: string): BillingRefusalPresentation['action'] => ({ type: 'portal', url })

const retryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? ` (try again in ~${Math.max(1, Math.round(refusal.retryAfter / 60))} min)` : ''

  return `🟡 Too many charges right now${mins}. This isn't a payment failure.`
}

const stripeRetryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? ` (try again in ~${Math.max(1, Math.round(refusal.retryAfter / 60))} min)` : ''

  return `Stripe is having trouble — try again shortly${mins}`
}

export const resolveRefusal = (refusal: BillingRefusal): BillingRefusalPresentation => {
  switch (refusal.kind) {
    case 'consent_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: 'Confirm this card for terminal charges in the portal',
        title: 'Card confirmation needed'
      }

    case 'insufficient_scope':
      return {
        action: { type: 'step_up' },
        message: 'This needs Remote Spending allowed. Start a top-up to allow it, then retry.',
        title: 'Remote Spending needs approval'
      }
    case 'remote_spending_revoked': {
      const who =
        refusal.actor === 'admin'
          ? 'An admin stopped remote spending for this terminal.'
          : 'You stopped remote spending for this terminal.'

      return {
        action: portalAction(refusal.portalUrl),
        message: `${who} Reconnect from Settings → Gateway to re-authorize this device.`,
        title: 'Remote spending was stopped'
      }
    }

    case 'session_revoked':
      return {
        action: portalAction(refusal.portalUrl),
        message: 'Your session was logged out. Sign in again from Settings → Gateway.',
        title: 'Session logged out'
      }

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      return {
        action: portalAction(refusal.portalUrl),
        message:
          "Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.",
        title: 'Remote spending is off'
      }

    case 'role_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: 'Adding funds needs an org admin/owner. Ask an admin, or manage on the portal.',
        title: 'Admin role required'
      }

    case 'idempotency_conflict':
      return {
        action: { type: 'none' },
        message: '🔴 That charge key was already used for a different amount. Start a fresh top-up.',
        title: 'Start a fresh top-up'
      }

    case 'no_payment_method':
      return {
        action: portalAction(refusal.portalUrl),
        message:
          '💳 No saved card for terminal charges yet. Set one up on the portal ' +
          "(one-time credit buys don't save a reusable card).",
        title: 'No saved card'
      }

    case 'org_access_denied':
      return {
        action: { type: 'none' },
        message: "This token isn't bound to an org you can manage",
        title: 'Org access denied'
      }
    case 'monthly_cap_exceeded': {
      const remaining = refusal.payload?.remainingUsd

      return {
        action: portalAction(refusal.portalUrl),
        message:
          remaining != null
            ? `🔴 Monthly spend cap reached — $${remaining} headroom left.`
            : '🔴 Monthly spend cap reached.',
        title: 'Monthly spend cap reached'
      }
    }

    case 'rate_limited':

    case 'temporarily_unavailable':
      return {
        action: { type: 'retry' },
        message: retryMessage(refusal),
        title: 'Too many charges right now'
      }

    case 'stripe_unavailable':
      return {
        action: { type: 'retry' },
        message: stripeRetryMessage(refusal),
        title: 'Stripe is having trouble'
      }

    case 'upgrade_cap_exceeded':
      return {
        action: { type: 'none' },
        message: 'Daily plan-change limit reached — try again tomorrow',
        title: 'Daily plan-change limit reached'
      }

    case 'endpoint_unavailable':
      return {
        action: { type: 'retry' },
        message:
          refusal.message ||
          'Billing endpoint returned a non-JSON response (it may not be available on this deployment).',
        title: 'Billing endpoint unavailable'
      }

    case 'timeout':
      return {
        action: { type: 'retry' },
        message: refusal.message || 'Billing request timed out.',
        title: 'Billing request timed out'
      }

    case 'transport':
      return {
        action: { type: 'retry' },
        message: refusal.message || 'Billing request failed before reaching the gateway.',
        title: 'Billing connection failed'
      }

    default:
      return {
        action: { type: 'none' },
        message: refusal.message || 'Billing request failed.',
        title: 'Billing request failed'
      }
  }
}
