/**
 * Compile-time guard for the generated `BillingStateResult['payment_method']` union.
 *
 * There is nothing to run here — the point is that `tsc` accepts this file.
 * An earlier revision typed the fallback arm's `kind` as `string & {}`, which
 * makes the discriminant non-literal and silently defeats narrowing for every
 * arm: the `pm.brand` read below stops compiling. Keeping this file honest
 * keeps `kind` narrowable.
 */

import type { BillingStateResult } from './gateway-contract.generated.js'

type BillingPaymentMethod = NonNullable<BillingStateResult['payment_method']>

export function describePaymentMethod(pm: BillingPaymentMethod): string {
  switch (pm.kind) {
    case 'card':
      return pm.wallet ? `${pm.wallet} ${pm.brand} ${pm.last4}` : `${pm.brand} ${pm.last4}`

    case 'link':
      return pm.email ?? 'Link'

    case 'unknown':
      return pm.raw_kind ?? 'unknown'
  }
}
