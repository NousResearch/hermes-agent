import { translateNow } from '@/i18n'

import type { BillingRefusal } from './api'

export interface BillingRefusalPresentation {
  action: { type: 'none' } | { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
  message: string
  title: string
}

const portalAction = (url?: string): BillingRefusalPresentation['action'] => ({ type: 'portal', url })

const retryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : null

  return mins
    ? translateNow('settings.billing.errors.tooManyChargesRetry', mins)
    : translateNow('settings.billing.errors.tooManyCharges')
}

const stripeRetryMessage = (refusal: BillingRefusal): string => {
  const mins = refusal.retryAfter ? Math.max(1, Math.round(refusal.retryAfter / 60)) : null

  return mins
    ? translateNow('settings.billing.errors.stripeRetry', mins)
    : translateNow('settings.billing.errors.stripeTrouble')
}

export const resolveRefusal = (refusal: BillingRefusal): BillingRefusalPresentation => {
  switch (refusal.kind) {
    case 'consent_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.confirmCard'),
        title: translateNow('settings.billing.errors.cardConfirmationNeeded')
      }

    case 'insufficient_scope':
      return {
        action: { type: 'step_up' },
        message: translateNow('settings.billing.errors.terminalBillingApprovalMessage'),
        title: translateNow('settings.billing.errors.terminalBillingApprovalTitle')
      }
    case 'remote_spending_revoked': {
      const who =
        refusal.actor === 'admin'
          ? translateNow('settings.billing.errors.adminDisabledTerminalBilling')
          : translateNow('settings.billing.errors.youDisabledTerminalBilling')

      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.reauthorizeDevice', who),
        title: translateNow('settings.billing.errors.terminalBillingOffTitle')
      }
    }

    case 'session_revoked':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.sessionLoggedOutMessage'),
        title: translateNow('settings.billing.errors.sessionLoggedOutTitle')
      }

    case 'cli_billing_disabled':

    case 'remote_spending_disabled':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.terminalBillingDisabledMessage'),
        title: translateNow('settings.billing.errors.terminalBillingDisabledTitle')
      }

    case 'role_required':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.adminRoleMessage'),
        title: translateNow('settings.billing.errors.adminRoleTitle')
      }

    case 'idempotency_conflict':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.idempotencyMessage'),
        title: translateNow('settings.billing.errors.idempotencyTitle')
      }

    case 'no_payment_method':
      return {
        action: portalAction(refusal.portalUrl),
        message: translateNow('settings.billing.errors.noPaymentMethodMessage'),
        title: translateNow('settings.billing.errors.noPaymentMethodTitle')
      }

    case 'org_access_denied':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.orgAccessMessage'),
        title: translateNow('settings.billing.errors.orgAccessTitle')
      }
    case 'monthly_cap_exceeded': {
      const remaining = refusal.payload?.remainingUsd

      return {
        action: portalAction(refusal.portalUrl),
        message:
          remaining != null
            ? translateNow('settings.billing.errors.monthlyCapRemaining', remaining)
            : translateNow('settings.billing.errors.monthlyCapMessage'),
        title: translateNow('settings.billing.errors.monthlyCapTitle')
      }
    }

    case 'rate_limited':

    case 'temporarily_unavailable':
      return {
        action: { type: 'retry' },
        message: retryMessage(refusal),
        title: translateNow('settings.billing.errors.tooManyChargesTitle')
      }

    case 'stripe_unavailable':
      return {
        action: { type: 'retry' },
        message: stripeRetryMessage(refusal),
        title: translateNow('settings.billing.errors.stripeTroubleTitle')
      }

    case 'upgrade_cap_exceeded':
      return {
        action: { type: 'none' },
        message: translateNow('settings.billing.errors.planLimitMessage'),
        title: translateNow('settings.billing.errors.planLimitTitle')
      }

    case 'endpoint_unavailable':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('settings.billing.errors.endpointUnavailableMessage'),
        title: translateNow('settings.billing.errors.endpointUnavailableTitle')
      }

    case 'timeout':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('settings.billing.errors.timeoutMessage'),
        title: translateNow('settings.billing.errors.timeoutTitle')
      }

    case 'transport':
      return {
        action: { type: 'retry' },
        message: refusal.message || translateNow('settings.billing.errors.transportMessage'),
        title: translateNow('settings.billing.errors.transportTitle')
      }

    default:
      return {
        action: { type: 'none' },
        message: refusal.message || translateNow('settings.billing.errors.defaultMessage'),
        title: translateNow('settings.billing.errors.defaultTitle')
      }
  }
}
