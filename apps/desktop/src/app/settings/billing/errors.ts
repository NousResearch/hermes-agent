import { translateNow } from '@/i18n'

import type { BillingRefusal } from './api'

export interface BillingRefusalPresentation {
  action: { type: 'none' } | { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
  message: string
  title: string
}

const portalAction = (url?: string): BillingRefusalPresentation['action'] => ({ type: 'portal', url })

const retryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : 0

  return translateNow('settings.billing.errors.rateLimited.message', mins)
}

const stripeRetryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : 0

  return translateNow('settings.billing.errors.stripeUnavailable.message', mins)
}

export const resolveRefusal = (refusal: BillingRefusal): BillingRefusalPresentation => {
  switch (refusal.kind) {
    case 'consent_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.consentRequired.message'),
        title: translateNow('settings.billing.errors.consentRequired.title')
      }

    case 'insufficient_scope':
      return {
        action: { type: 'step_up' },
        message: translateNow('settings.billing.errors.insufficientScope.message'),
        title: translateNow('settings.billing.errors.insufficientScope.title')
      }
    case 'remote_spending_revoked': {
      const who =
        refusal.actor === 'admin'
          ? translateNow('settings.billing.errors.remoteSpendingRevoked.messageByAdmin')
          : translateNow('settings.billing.errors.remoteSpendingRevoked.messageBySelf')

      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.remoteSpendingReconnect', who),
        title: translateNow('settings.billing.errors.remoteSpendingRevoked.title')
      }
    }

    case 'session_revoked':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.sessionRevoked.message'),
        title: translateNow('settings.billing.errors.sessionRevoked.title')
      }

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.cliBillingDisabled.message'),
        title: translateNow('settings.billing.errors.cliBillingDisabled.title')
      }

    case 'role_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.roleRequired.message'),
        title: translateNow('settings.billing.errors.roleRequired.title')
      }

    case 'idempotency_conflict':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.idempotencyConflict.message'),
        title: translateNow('settings.billing.errors.idempotencyConflict.title')
      }

    case 'no_payment_method':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.noPaymentMethod.message'),
        title: translateNow('settings.billing.errors.noPaymentMethod.title')
      }

    case 'org_access_denied':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.orgAccessDenied.message'),
        title: translateNow('settings.billing.errors.orgAccessDenied.title')
      }
    case 'monthly_cap_exceeded': {
      const remaining = refusal.payload?.remainingUsd

      return {
        action: portalAction(refusal.portalUrl),
        message:
          remaining != null
            ? translateNow('settings.billing.errors.monthlyCapExceeded.messageHeadroom', remaining)
            : translateNow('settings.billing.errors.monthlyCapExceeded.messageReached'),
        title: translateNow('settings.billing.errors.monthlyCapExceeded.title')
      }
    }

    case 'rate_limited':

    case 'temporarily_unavailable':
      return {
        action: { type: 'retry' },
        message: retryMessage(refusal),
        title: translateNow('settings.billing.errors.rateLimited.title')
      }

    case 'stripe_unavailable':
      return {
        action: { type: 'retry' },
        message: stripeRetryMessage(refusal),
        title: translateNow('settings.billing.errors.stripeUnavailable.title')
      }

    case 'upgrade_cap_exceeded':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.upgradeCapExceeded.message'),
        title: translateNow('settings.billing.errors.upgradeCapExceeded.title')
      }

    case 'endpoint_unavailable':
      return {
        action: { type: 'retry' },
        message:
          refusal.message ||
          translateNow('settings.billing.errors.endpointUnavailable.message'),
        title: translateNow('settings.billing.errors.endpointUnavailable.title')
      }

    case 'timeout':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('settings.billing.errors.timeout.message'),
        title: translateNow('settings.billing.errors.timeout.title')
      }

    case 'transport':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('settings.billing.errors.transport.message'),
        title: translateNow('settings.billing.errors.transport.title')
      }

    default:
      return {
        action: { type: 'none' },
        message: refusal.message || translateNow('settings.billing.errors.default.message'),
        title: translateNow('settings.billing.errors.default.title')
      }
  }
}
