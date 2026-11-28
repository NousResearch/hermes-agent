/** The backend prices sessions in USD — that is what providers bill, and what state.db stores
 *  (`estimated_cost_usd`). The status bar shows the owner's currency instead, converted here at
 *  display time so the stored figure never changes meaning.
 *
 *  A fixed rate, not a live fetch: the readout must work offline and stay stable between ticks,
 *  and this is an estimate of spend, not an invoice. Change the two values to switch currency.
 */
export const DISPLAY_CURRENCY = {
  /** Prefix shown on every money readout. */
  symbol: '¥',
  /** Multiply a USD figure by this to get the display amount. */
  rate: 7.2
} as const

/** Money formatting sized to what these sessions actually cost: sub-cent turns keep three or four
 *  decimals instead of collapsing to a useless `¥0.00`. */
export function formatDisplayCost(usd: number): string {
  const value = usd * DISPLAY_CURRENCY.rate

  if (value >= 1) {
    return `${DISPLAY_CURRENCY.symbol}${value.toFixed(2)}`
  }

  if (value >= 0.01) {
    return `${DISPLAY_CURRENCY.symbol}${value.toFixed(3)}`
  }

  return `${DISPLAY_CURRENCY.symbol}${value.toFixed(4)}`
}

/** The provider-billed figure behind a display amount, for the detail panel: `≈ $1.41 USD`. */
export function formatSourceUsd(usd: number): string {
  return usd >= 1 ? `≈ $${usd.toFixed(2)} USD` : `≈ $${usd.toFixed(4)} USD`
}
