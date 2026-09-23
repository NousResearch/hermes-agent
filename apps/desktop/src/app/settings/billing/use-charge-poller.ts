import { refusalPolicy } from '@hermes/shared/billing-policy'
import {
  driveChargeSettlement,
  SETTLEMENT_POLL_CAP_MS,
  SETTLEMENT_POLL_INTERVAL_MS
} from '@hermes/shared/charge-settlement'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useRef, useState } from 'react'

import { type Translations, useI18n } from '@/i18n'
import { en } from '@/i18n/en'

import type { BillingApi, BillingRefusal } from './api'
import { useBillingApi } from './api'
import { resolveRefusal } from './errors'
import type { BillingChargeStatusResponse } from './types'

export const CHARGE_POLL_INTERVAL_MS = SETTLEMENT_POLL_INTERVAL_MS
export const CHARGE_POLL_CAP_MS = SETTLEMENT_POLL_CAP_MS

export type ChargeFlowPhase = 'charging' | 'done' | 'idle' | 'polling'

export type ChargeFlowOutcome =
  | {
      amountUsd?: string | null
      kind: 'success'
      message: string
    }
  | {
      action?: { type: 'portal'; url?: string } | { type: 'retry' } | { type: 'step_up' }
      kind: 'failure'
      copy: 'failed' | 'check' | 'untracked' | 'refusal'
      refusal?: BillingRefusal
      reason?: null | string
      message: string
      retryFreshKey: boolean
      title: string
    }
  | {
      kind: 'ambiguous'
      copy: 'unconfirmed' | 'timeout'
      refusal?: BillingRefusal
      message: string
      portalUrl?: string
      title: string
    }

export interface ChargePollClock {
  now?: () => number
  sleep?: (ms: number) => Promise<void>
}

export interface ChargePollOptions extends ChargePollClock {
  portalUrl?: null | string
}

interface PendingChargeIntent {
  amountUsd: string
  idempotencyKey: string
}

const defaultSleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

const retryableSendKinds = new Set([
  'endpoint_unavailable',
  'rate_limited',
  'temporarily_unavailable',
  'timeout',
  'transport'
])

export async function pollChargeSettlement(
  api: Pick<BillingApi, 'chargeStatus'>,
  chargeId: string,
  opts: ChargePollOptions = {}
): Promise<ChargeFlowOutcome> {
  const sleep = opts.sleep ?? defaultSleep
  const now = opts.now ?? Date.now
  const observed: { refusal?: BillingRefusal; status?: BillingChargeStatusResponse } = {}

  const settlement = await driveChargeSettlement({
    fetchStatus: async () => {
      const result = await api.chargeStatus(chargeId)

      if (result.ok) {
        observed.refusal = undefined
        observed.status = result.data

        return result.data
      }

      observed.refusal = result.refusal
      observed.status = statusFromRefusal(result.refusal)

      return observed.status
    },
    isCancelled: () => false,
    now,
    sleep
  })

  switch (settlement.kind) {
    case 'settled':
      return {
        amountUsd: settlement.status.amount_usd,
        kind: 'success',
        message: en.settings.billing.charge.added(settlement.status.amount_usd ?? '')
      }

    case 'failed':
      return {
        action: { type: 'retry' },
        kind: 'failure',
        copy: 'failed',
        reason: settlement.status.reason,
        message: renderChargeFailed(settlement.status.reason),
        retryFreshKey: true,
        title: en.settings.billing.charge.failedTitle
      }
    case 'ambiguous': {
      if (settlement.status && refusalPolicy(settlement.error).ambiguousMidPoll) {
        const refusal = observed.refusal ?? refusalFromStatus(settlement.error, settlement.status)
        const resolved = resolveRefusal(refusal)
        const portalUrl = resolved.action.type === 'portal' ? resolved.action.url : refusal.portalUrl

        return {
          kind: 'ambiguous',
          copy: 'unconfirmed',
          refusal,
          message: en.settings.billing.charge.unconfirmedBody(resolved.message),
          portalUrl: portalUrl ?? opts.portalUrl ?? undefined,
          title: en.settings.billing.charge.unconfirmedTitle
        }
      }

      return {
        kind: 'failure',
        copy: 'check',
        reason: observed.refusal?.message,
        message: observed.refusal?.message || en.settings.billing.charge.checkBody,
        retryFreshKey: true,
        title: en.settings.billing.charge.checkTitle
      }
    }

    case 'refused':
      return {
        kind: 'failure',
        copy: 'check',
        reason: observed.refusal?.message || settlement.status.message,
        message: observed.refusal?.message || settlement.status.message || en.settings.billing.charge.checkBody,
        retryFreshKey: true,
        title: en.settings.billing.charge.checkTitle
      }

    case 'cancelled':

    case 'timed_out':
      return timeoutOutcome(observed.status?.ok ? (observed.status.portal_url ?? opts.portalUrl) : opts.portalUrl)
  }
}

function statusFromRefusal(refusal: BillingRefusal): BillingChargeStatusResponse {
  const raw = isRecord(refusal.raw) ? refusal.raw : {}

  return {
    ...raw,
    error: refusal.kind,
    message: refusal.message,
    ok: false,
    ...(refusal.payload !== undefined ? { payload: refusal.payload } : {}),
    ...(refusal.portalUrl !== undefined ? { portal_url: refusal.portalUrl } : {}),
    ...(refusal.retryAfter !== undefined ? { retry_after: refusal.retryAfter } : {})
  } as BillingChargeStatusResponse
}

function refusalFromStatus(error: string, status: BillingChargeStatusResponse): BillingRefusal {
  return {
    kind: error,
    message: status.message || error,
    payload: status.payload,
    portalUrl: status.portal_url ?? undefined,
    retryAfter: status.retry_after ?? undefined
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function useChargeFlow() {
  const { t } = useI18n()
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [phase, setPhase] = useState<ChargeFlowPhase>('idle')
  const [outcome, setOutcome] = useState<ChargeFlowOutcome | null>(null)
  const phaseRef = useRef<ChargeFlowPhase>('idle')
  const retryIntentRef = useRef<PendingChargeIntent | null>(null)

  const setPhaseState = useCallback((next: ChargeFlowPhase) => {
    phaseRef.current = next
    setPhase(next)
  }, [])

  const reset = useCallback(() => {
    retryIntentRef.current = null
    setOutcome(null)
    setPhaseState('idle')
  }, [setPhaseState])

  const start = useCallback(
    async (amountUsd: string) => {
      if (phaseRef.current === 'charging' || phaseRef.current === 'polling') {
        return
      }

      const retryIntent = retryIntentRef.current
      const idempotencyKey = retryIntent?.amountUsd === amountUsd ? retryIntent.idempotencyKey : undefined

      setOutcome(null)
      setPhaseState('charging')

      const chargeResult = await api.charge(amountUsd, idempotencyKey)

      if (!chargeResult.ok) {
        const resolved = resolveRefusal(chargeResult.refusal)

        const action =
          resolved.action.type === 'portal'
            ? ({ type: 'portal', url: resolved.action.url } as const)
            : resolved.action.type === 'retry'
              ? ({ type: 'retry' } as const)
              : resolved.action.type === 'step_up'
                ? ({ type: 'step_up' } as const)
                : undefined

        retryIntentRef.current = shouldReuseIdempotencyKey(chargeResult.refusal)
          ? { amountUsd, idempotencyKey: chargeResult.idempotencyKey }
          : null
        setOutcome({
          action,
          kind: 'failure',
          copy: 'refusal',
          refusal: chargeResult.refusal,
          message: resolved.message,
          retryFreshKey: false,
          title: resolved.title
        })
        setPhaseState('done')

        return
      }

      retryIntentRef.current = null

      const chargeId = chargeResult.data.charge_id

      if (!chargeId) {
        setOutcome({
          kind: 'failure',
          copy: 'untracked',
          message: en.settings.billing.charge.untrackedBody,
          retryFreshKey: true,
          title: en.settings.billing.charge.untrackedTitle
        })
        setPhaseState('done')

        return
      }

      setPhaseState('polling')

      const pollOutcome = await pollChargeSettlement(api, chargeId, {
        portalUrl: chargeResult.data.portal_url
      })

      setOutcome(pollOutcome)
      setPhaseState('done')

      if (pollOutcome.kind === 'success') {
        void queryClient.invalidateQueries({ queryKey: ['billing', 'state'] })
      }
    },
    [api, queryClient, setPhaseState]
  )

  return { outcome: outcome ? localizeChargeOutcome(outcome, t.settings.billing) : null, phase, reset, start }
}

function shouldReuseIdempotencyKey(refusal: BillingRefusal): boolean {
  return retryableSendKinds.has(refusal.kind)
}

function timeoutOutcome(portalUrl?: null | string): ChargeFlowOutcome {
  return {
    kind: 'ambiguous',
    copy: 'timeout',
    message: en.settings.billing.charge.timeoutBody,
    portalUrl: portalUrl ?? undefined,
    title: en.settings.billing.charge.timeoutTitle
  }
}

function renderChargeFailed(reason: null | string | undefined, copy = en.settings.billing.charge): string {
  switch ((reason || '').trim()) {
    case 'authentication_required':
      return copy.authenticationRequired

    case 'payment_method_expired':
      return copy.expired

    case 'card_declined':
      return copy.declined

    default:
      return copy.failedBody(reason || 'processing_error')
  }
}

// Repaint completed feedback on a locale switch without replaying a charge.
function localizeChargeOutcome(outcome: ChargeFlowOutcome, b: Translations['settings']['billing']): ChargeFlowOutcome {
  if (outcome.kind === 'success') {return { ...outcome, message: b.charge.added(outcome.amountUsd ?? '') }}

  switch (outcome.copy) {
    case 'refusal': {
      if (!outcome.refusal) {return outcome}
      const resolved = resolveRefusal(outcome.refusal, b.errors)

      return { ...outcome, message: resolved.message, title: resolved.title }
    }

    case 'unconfirmed':
      return outcome.refusal
        ? {
            ...outcome,
            message: b.charge.unconfirmedBody(resolveRefusal(outcome.refusal, b.errors).message),
            title: b.charge.unconfirmedTitle
          }
        : outcome

    case 'failed':
      return { ...outcome, message: renderChargeFailed(outcome.reason, b.charge), title: b.charge.failedTitle }

    case 'check':
      return { ...outcome, message: outcome.reason || b.charge.checkBody, title: b.charge.checkTitle }

    case 'untracked':
      return { ...outcome, message: b.charge.untrackedBody, title: b.charge.untrackedTitle }

    case 'timeout':
      return { ...outcome, message: b.charge.timeoutBody, title: b.charge.timeoutTitle }
  }
}
