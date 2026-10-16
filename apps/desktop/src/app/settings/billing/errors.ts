import type { Translations } from '@/i18n'
import { en } from '@/i18n/en'

import type { BillingRefusal } from './api'

export interface BillingRefusalPresentation {
  action: { type: 'none' } | { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
  message: string
  title: string
}

const portalAction = (url?: string): BillingRefusalPresentation['action'] => ({ type: 'portal', url })

const retryMessage = (refusal: BillingRefusal, copy: Translations['settings']['billing']['errors']): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : 0

  return copy.rateLimited.message(mins)
}

const stripeRetryMessage = (refusal: BillingRefusal, copy: Translations['settings']['billing']['errors']): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : 0

  return copy.stripeUnavailable.message(mins)
}

export const resolveRefusal = (
  refusal: BillingRefusal,
  copy: Translations['settings']['billing']['errors'] = en.settings.billing.errors
): BillingRefusalPresentation => {
  switch (refusal.kind) {
    case 'consent_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: copy.consentRequired.message,
        title: copy.consentRequired.title
      }

    case 'insufficient_scope':
      return {
        action: { type: 'step_up' },
        message: copy.insufficientScope.message,
        title: copy.insufficientScope.title
      }
    case 'remote_spending_revoked': {
      const who =
        refusal.actor === 'admin' ? copy.remoteSpendingRevoked.messageByAdmin : copy.remoteSpendingRevoked.messageBySelf

      return {
        action: portalAction(refusal.portalUrl),
        message: copy.remoteSpendingReconnect(who),
        title: copy.remoteSpendingRevoked.title
      }
    }

    case 'session_revoked':
      return {
        action: portalAction(refusal.portalUrl),
        message: copy.sessionRevoked.message,
        title: copy.sessionRevoked.title
      }

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      return {
        action: portalAction(refusal.portalUrl),
        message: copy.cliBillingDisabled.message,
        title: copy.cliBillingDisabled.title
      }

    case 'role_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: copy.roleRequired.message,
        title: copy.roleRequired.title
      }

    case 'idempotency_conflict':
      return {
        action: { type: 'none' },
        message: copy.idempotencyConflict.message,
        title: copy.idempotencyConflict.title
      }

    case 'no_payment_method':
      return {
        action: portalAction(refusal.portalUrl),
        message: copy.noPaymentMethod.message,
        title: copy.noPaymentMethod.title
      }

    case 'org_access_denied':
      return {
        action: { type: 'none' },
        message: copy.orgAccessDenied.message,
        title: copy.orgAccessDenied.title
      }
    case 'monthly_cap_exceeded': {
      const remaining = refusal.payload?.remainingUsd

      return {
        action: portalAction(refusal.portalUrl),
        message:
          remaining != null
            ? copy.monthlyCapExceeded.messageHeadroom(remaining)
            : copy.monthlyCapExceeded.messageReached,
        title: copy.monthlyCapExceeded.title
      }
    }

    case 'rate_limited':

    case 'temporarily_unavailable':
      return {
        action: { type: 'retry' },
        message: retryMessage(refusal, copy),
        title: copy.rateLimited.title
      }

    case 'stripe_unavailable':
      return {
        action: { type: 'retry' },
        message: stripeRetryMessage(refusal, copy),
        title: copy.stripeUnavailable.title
      }

    case 'upgrade_cap_exceeded':
      return {
        action: { type: 'none' },
        message: copy.upgradeCapExceeded.message,
        title: copy.upgradeCapExceeded.title
      }

    case 'endpoint_unavailable':
      return {
        action: { type: 'retry' },
        message: refusal.message || copy.endpointUnavailable.message,
        title: copy.endpointUnavailable.title
      }

    case 'timeout':
      return {
        action: { type: 'retry' },
        message: refusal.message || copy.timeout.message,
        title: copy.timeout.title
      }

    case 'transport':
      return {
        action: { type: 'retry' },
        message: refusal.message || copy.transport.message,
        title: copy.transport.title
      }

    default:
      return {
        action: { type: 'none' },
        message: refusal.message || copy.default.message,
        title: copy.default.title
      }
  }
}
