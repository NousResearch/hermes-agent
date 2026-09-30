import { translateNow } from '@/i18n'

import type { BillingRefusal } from './api'

export interface BillingRefusalPresentation {
  action: { type: 'none' } | { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
  message: string
  title: string
}

const portalAction = (url?: string): BillingRefusalPresentation['action'] => ({ type: 'portal', url })

const retryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter
    ? translateNow('billingRisk.refusal.retryAfter', Math.max(1, Math.round(refusal.retryAfter / 60)))
    : ''

  return translateNow('billingRisk.refusal.rateMessage', mins)
}

const stripeRetryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter
    ? translateNow('billingRisk.refusal.retryAfter', Math.max(1, Math.round(refusal.retryAfter / 60)))
    : ''

  return translateNow('billingRisk.refusal.stripeMessage', mins)
}

export const resolveRefusal = (refusal: BillingRefusal): BillingRefusalPresentation => {
  switch (refusal.kind) {
    case 'consent_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.consentMessage'),
        title: translateNow('billingRisk.refusal.consentTitle')
      }

    case 'insufficient_scope':
      return {
        action: { type: 'step_up' },
        message: translateNow('billingRisk.refusal.scopeMessage'),
        title: translateNow('billingRisk.refusal.scopeTitle')
      }
    case 'remote_spending_revoked': {
      const who =
        refusal.actor === 'admin'
          ? translateNow('billingRisk.refusal.revokedAdmin')
          : translateNow('billingRisk.refusal.revokedSelf')

      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.revokedMessage', who),
        title: translateNow('billingRisk.refusal.revokedTitle')
      }
    }

    case 'session_revoked':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.sessionMessage'),
        title: translateNow('billingRisk.refusal.sessionTitle')
      }

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.disabledMessage'),
        title: translateNow('billingRisk.refusal.disabledTitle')
      }

    case 'role_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.roleMessage'),
        title: translateNow('billingRisk.refusal.roleTitle')
      }

    case 'idempotency_conflict':
      return {
        action: { type: 'none' },
        message: translateNow('billingRisk.refusal.conflictMessage'),
        title: translateNow('billingRisk.refusal.conflictTitle')
      }

    case 'no_payment_method':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('billingRisk.refusal.noCardMessage'),
        title: translateNow('billingRisk.refusal.noCardTitle')
      }

    case 'org_access_denied':
      return {
        action: { type: 'none' },
        message: translateNow('billingRisk.refusal.orgMessage'),
        title: translateNow('billingRisk.refusal.orgTitle')
      }
    case 'monthly_cap_exceeded': {
      const remaining = refusal.payload?.remainingUsd

      return {
        action: portalAction(refusal.portalUrl),
        message:
          remaining != null
            ? translateNow('billingRisk.refusal.capRemaining', String(remaining))
            : translateNow('billingRisk.refusal.capMessage'),
        title: translateNow('billingRisk.refusal.capTitle')
      }
    }

    case 'rate_limited':

    case 'temporarily_unavailable':
      return {
        action: { type: 'retry' },
        message: retryMessage(refusal),
        title: translateNow('billingRisk.refusal.rateTitle')
      }

    case 'stripe_unavailable':
      return {
        action: { type: 'retry' },
        message: stripeRetryMessage(refusal),
        title: translateNow('billingRisk.refusal.stripeTitle')
      }

    case 'upgrade_cap_exceeded':
      return {
        action: { type: 'none' },
        message: translateNow('billingRisk.refusal.upgradeMessage'),
        title: translateNow('billingRisk.refusal.upgradeTitle')
      }

    case 'endpoint_unavailable':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('billingRisk.refusal.endpointMessage'),
        title: translateNow('billingRisk.refusal.endpointTitle')
      }

    case 'timeout':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('billingRisk.refusal.timeoutMessage'),
        title: translateNow('billingRisk.refusal.timeoutTitle')
      }

    case 'transport':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('billingRisk.refusal.transportMessage'),
        title: translateNow('billingRisk.refusal.transportTitle')
      }

    default:
      return {
        action: { type: 'none' },
        message: refusal.message || translateNow('billingRisk.refusal.failedMessage'),
        title: translateNow('billingRisk.refusal.failedTitle')
      }
  }
}
