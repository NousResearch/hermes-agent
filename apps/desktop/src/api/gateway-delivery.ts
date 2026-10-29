/** Errors of a request the gateway hook resent after a dropped connection, when the first attempt may
 *  already have reached the backend. Such an error answers the resend, not the first attempt: a refusal
 *  here can be about what that first attempt already did. A WeakSet, because the error is often a
 *  `JsonRpcGatewayError` (readonly fields, and a wrapper would break `instanceof`). */
const resentAfterPossibleDelivery = new WeakSet<object>()

export function markResentAfterPossibleDelivery(outcome: unknown): void {
  if (typeof outcome === 'object' && outcome !== null) {
    resentAfterPossibleDelivery.add(outcome)
  }
}

export function wasResentAfterPossibleDelivery(outcome: unknown): boolean {
  return typeof outcome === 'object' && outcome !== null && resentAfterPossibleDelivery.has(outcome)
}
