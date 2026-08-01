import type { BillingResult } from './api'
import type { BillingStateResponse, SubscriptionStateResponse } from './types'

export {
  billingDevFixtures,
  loggedOutBillingState,
  loggedOutSubscriptionState,
  postTrainBillingState,
  postTrainSubscriptionState,
  todayBillingState,
  todaySubscriptionState
} from './dev-fixtures'

export const okBilling = (data: BillingStateResponse): BillingResult<BillingStateResponse> => ({ data, ok: true })

export const okSubscription = (data: SubscriptionStateResponse): BillingResult<SubscriptionStateResponse> => ({
  data,
  ok: true
})

export const endpointUnavailableBilling = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Billing endpoint returned a non-JSON response.'
  }
} satisfies BillingResult<BillingStateResponse>

export const endpointUnavailableSubscription = {
  ok: false,
  refusal: {
    kind: 'endpoint_unavailable',
    message: 'Subscription endpoint is not available.'
  }
} satisfies BillingResult<SubscriptionStateResponse>
