import { createContext, useContext, useMemo } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'

import type {
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMutationResponse,
  BillingRefusalCode,
  BillingStateResponse,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse
} from './types'

export type BillingErrorKind = BillingRefusalCode

export interface BillingRefusal {
  actor?: string
  code?: string
  kind: BillingErrorKind | 'timeout' | 'transport'
  message: string
  payload?: BillingErrorPayload
  portalUrl?: string
  raw?: unknown
  recovery?: string
  retryAfter?: number
}

export type BillingResult<T> = { data: T; ok: true } | { ok: false; refusal: BillingRefusal }

export type BillingChargeResult = BillingResult<BillingChargeResponse> & { idempotencyKey: string }

export interface UpdateAutoReloadInput {
  enabled: boolean
  reload_to_usd?: string
  threshold_usd?: string
}

export type BillingRequestGateway = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

export interface BillingApi {
  charge: (amountUsd: string, idempotencyKey?: string) => Promise<BillingChargeResult>
  chargeStatus: (chargeId: string) => Promise<BillingResult<BillingChargeStatusResponse>>
  fetchBillingState: () => Promise<BillingResult<BillingStateResponse>>
  fetchSubscriptionState: () => Promise<BillingResult<SubscriptionStateResponse>>
  /** Chargeless quote for a plan change (POST /subscription/preview). */
  previewSubscriptionChange: (tierId: string) => Promise<BillingResult<SubscriptionPreviewResponse>>
  /** Clear a scheduled downgrade / cancellation — the undo (DELETE pending-change). */
  resumeSubscription: () => Promise<BillingResult<BillingMutationResponse>>
  /** Schedule a chargeless downgrade at period end (PUT pending-change). */
  scheduleSubscriptionChange: (tierId: string) => Promise<BillingResult<BillingMutationResponse>>
  stepUp: (sessionId?: string) => Promise<BillingResult<BillingMutationResponse>>
  updateAutoReload: (input: UpdateAutoReloadInput) => Promise<BillingResult<BillingMutationResponse>>
}

interface RefusalRecord {
  actor?: unknown
  code?: unknown
  error?: unknown
  kind?: unknown
  message?: unknown
  payload?: unknown
  portal_url?: unknown
  recovery?: unknown
  retry_after?: unknown
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null

const asOptionalString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.length > 0 ? value : undefined

const asOptionalNumber = (value: unknown): number | undefined => (typeof value === 'number' ? value : undefined)

const asPayload = (value: unknown): BillingErrorPayload | undefined =>
  isRecord(value) ? (value as BillingErrorPayload) : undefined

const getMessage = (value: unknown): string => {
  if (value instanceof Error && value.message) {
    return value.message
  }

  if (typeof value === 'string' && value.length > 0) {
    return value
  }

  return String(value || 'Billing request failed.')
}

const normalizeRefusal = (raw: Record<string, unknown>): BillingRefusal => {
  const rawError = raw.error
  const error = isRecord(rawError) ? (rawError as RefusalRecord) : undefined
  const kind = asOptionalString(error?.kind) ?? asOptionalString(error?.error) ?? asOptionalString(rawError) ?? 'error'
  const message = asOptionalString(error?.message) ?? asOptionalString(raw.message) ?? kind

  return {
    actor: asOptionalString(error?.actor) ?? asOptionalString(raw.actor),
    code: asOptionalString(error?.code) ?? asOptionalString(raw.code),
    kind,
    message,
    payload: asPayload(error?.payload) ?? asPayload(raw.payload),
    portalUrl: asOptionalString(error?.portal_url) ?? asOptionalString(raw.portal_url),
    raw,
    recovery: asOptionalString(error?.recovery) ?? asOptionalString(raw.recovery),
    retryAfter: asOptionalNumber(error?.retry_after) ?? asOptionalNumber(raw.retry_after)
  }
}

const normalizeThrown = (error: unknown): BillingRefusal => {
  const message = getMessage(error)
  const name = error instanceof Error ? error.name : ''

  return {
    kind: name === 'TimeoutError' || /timed?\s*out|timeout/i.test(message) ? 'timeout' : 'transport',
    message,
    raw: error
  }
}

const normalizeRpcResult = <T>(response: T): BillingResult<T> => {
  if (isRecord(response) && response.ok === false) {
    return { ok: false, refusal: normalizeRefusal(response) }
  }

  return { data: response, ok: true }
}

const callBilling = async <T>(
  requestGateway: BillingRequestGateway,
  method: string,
  params: Record<string, unknown> = {}
): Promise<BillingResult<T>> => {
  try {
    return normalizeRpcResult(await requestGateway<T>(method, params))
  } catch (error) {
    return { ok: false, refusal: normalizeThrown(error) }
  }
}

export const createBillingApi = (requestGateway: BillingRequestGateway): BillingApi => ({
  charge: async (amountUsd, idempotencyKey = crypto.randomUUID()) => {
    const result = await callBilling<BillingChargeResponse>(requestGateway, 'billing.charge', {
      amount_usd: amountUsd,
      idempotency_key: idempotencyKey
    })

    return { ...result, idempotencyKey }
  },
  chargeStatus: chargeId =>
    callBilling<BillingChargeStatusResponse>(requestGateway, 'billing.charge_status', { charge_id: chargeId }),
  fetchBillingState: () => callBilling<BillingStateResponse>(requestGateway, 'billing.state'),
  fetchSubscriptionState: () => callBilling<SubscriptionStateResponse>(requestGateway, 'subscription.state'),
  previewSubscriptionChange: tierId =>
    callBilling<SubscriptionPreviewResponse>(requestGateway, 'subscription.preview', {
      subscription_type_id: tierId
    }),
  resumeSubscription: () => callBilling<BillingMutationResponse>(requestGateway, 'subscription.resume', {}),
  scheduleSubscriptionChange: tierId =>
    callBilling<BillingMutationResponse>(requestGateway, 'subscription.change', {
      subscription_type_id: tierId
    }),
  stepUp: sessionId =>
    callBilling<BillingMutationResponse>(requestGateway, 'billing.step_up', {
      ...(sessionId !== undefined ? { session_id: sessionId } : {})
    }),
  updateAutoReload: input =>
    callBilling<BillingMutationResponse>(requestGateway, 'billing.auto_reload', {
      enabled: input.enabled,
      ...(input.threshold_usd !== undefined ? { threshold: input.threshold_usd } : {}),
      ...(input.reload_to_usd !== undefined ? { top_up_amount: input.reload_to_usd } : {})
    })
})

// An override for the gateway-backed api — DEV fixtures provide a simulated
// implementation here so every consumer (hooks, rows) transparently runs against it
// with no fixture awareness of their own. `null` (the default) = the real gateway api.
const BillingApiContext = createContext<BillingApi | null>(null)

export const BillingApiProvider = BillingApiContext.Provider

export function useBillingApi(): BillingApi {
  const override = useContext(BillingApiContext)
  const { requestGateway } = useGatewayRequest()
  const real = useMemo(() => createBillingApi(requestGateway), [requestGateway])

  return override ?? real
}
