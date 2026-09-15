import type { BillingResult } from './api'
import type { BillingStateResult, SubscriptionStateResult } from './types'

export {
  billingDevFixtures,
  loggedOutBillingState,
  loggedOutSubscriptionState,
  OK_ENVELOPE,
  postTrainBillingState,
  postTrainSubscriptionState,
  todayBillingState,
  todaySubscriptionState,
  usageModel
} from './dev-fixtures'

export const okBilling = (data: BillingStateResult): BillingResult<BillingStateResult> => ({ data, ok: true })

export const okSubscription = (data: SubscriptionStateResult): BillingResult<SubscriptionStateResult> => ({
  data,
  ok: true
})

export const endpointUnavailableBilling = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Billing endpoint returned a non-JSON response.'
  }
} satisfies BillingResult<BillingStateResult>

export const endpointUnavailableSubscription = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Subscription endpoint is not available.'
  }
} satisfies BillingResult<SubscriptionStateResult>
