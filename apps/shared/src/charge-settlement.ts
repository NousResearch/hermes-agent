import { refusalPolicy } from './billing-policy.js'
import type { BillingChargeStatusResult } from './gateway-contract.generated.js'

export interface SettlementDeps {
  fetchStatus(): Promise<BillingChargeStatusResult>
  sleep(ms: number): Promise<void>
  isCancelled(): boolean
  now(): number
}

export type SettlementOutcome =
  | { kind: 'settled'; status: BillingChargeStatusResult }
  | { kind: 'failed'; status: BillingChargeStatusResult }
  | { kind: 'refused'; error: string; status: BillingChargeStatusResult }
  | {
      kind: 'ambiguous'
      error: string
      status?: BillingChargeStatusResult
      cause?: unknown
    }
  | { kind: 'timed_out' }
  | { kind: 'cancelled' }

export const SETTLEMENT_POLL_INTERVAL_MS = 2000
export const SETTLEMENT_POLL_CAP_MS = 5 * 60 * 1000
export const SETTLEMENT_MAX_RETRY_AFTER_MS = 30000

export async function driveChargeSettlement(deps: SettlementDeps): Promise<SettlementOutcome> {
  const start = deps.now()
  const timedOut = (): boolean => deps.now() - start >= SETTLEMENT_POLL_CAP_MS

  while (true) {
    if (deps.isCancelled()) {
      return { kind: 'cancelled' }
    }

    let status: BillingChargeStatusResult

    try {
      status = await deps.fetchStatus()
    } catch (cause) {
      return { kind: 'ambiguous', error: 'transport', cause }
    }

    if (!status.ok) {
      const error = status.error ?? ''
      const policy = refusalPolicy(error)

      if (policy.ambiguousMidPoll) {
        return { kind: 'ambiguous', error: error || 'unknown', status }
      }

      if (policy.recovery === 'retry') {
        if (timedOut()) {
          return { kind: 'timed_out' }
        }

        const wait = Math.min(
          (status.retry_after ?? 5) * 1000,
          SETTLEMENT_MAX_RETRY_AFTER_MS
        )

        await deps.sleep(wait)

        continue
      }

      return {
        kind: 'refused',
        error: status.error ?? status.message ?? 'error',
        status
      }
    }

    if (status.status === 'settled') {
      return { kind: 'settled', status }
    }

    if (status.status === 'failed') {
      return { kind: 'failed', status }
    }

    if (timedOut()) {
      return { kind: 'timed_out' }
    }

    await deps.sleep(SETTLEMENT_POLL_INTERVAL_MS)
  }
}
