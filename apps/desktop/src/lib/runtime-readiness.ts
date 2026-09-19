import type { RpcMethods, SetupRuntimeCheckResult, SetupStatusResult } from '@hermes/shared'

import type { GatewayRequest } from '@/lib/gateway-rpc'

export type SetupStatusSnapshot = SetupStatusResult

export type RuntimeCheckSnapshot = SetupRuntimeCheckResult

export interface RuntimeReadinessSignals {
  setup: null | SetupStatusSnapshot
  setupError: null | string
  runtime: null | RuntimeCheckSnapshot
  runtimeError: null | string
}

export interface RuntimeReadinessOptions {
  defaultReason?: string
  requestedProvider?: string
  unknownReady?: boolean
}

export interface RuntimeReadinessResult {
  checksDisagree: boolean
  /** Passed through from `setup.runtime_check`: the resolved route is the free
   *  tier. Undefined when the check did not answer (older backend, transport
   *  fallback) — never read it as "not free tier". */
  freeTier?: boolean
  /** Passed through from `setup.runtime_check`: the model the route resolved
   *  to. Undefined when the check did not answer. */
  model?: string
  ready: boolean
  reason: null | string
  source: 'fallback' | 'runtime_check' | 'setup_status'
}

export type RuntimeReadinessDisplay = 'checking' | 'needs_setup' | 'ready' | 'unavailable'

export type RuntimeReadinessRequester = GatewayRequest

const DEFAULT_NOT_READY_REASON = 'Add a provider credential before sending your first message.'

function toErrorMessage(error: unknown): null | string {
  if (error instanceof Error) {
    return error.message
  }

  if (typeof error === 'string') {
    return error
  }

  if (error === null || error === undefined) {
    return null
  }

  return String(error)
}

function normalizeMessage(value: null | string | undefined): null | string {
  const next = value?.trim()

  return next ? next : null
}

async function requestWithFallback<M extends keyof RpcMethods>(
  requestGateway: RuntimeReadinessRequester,
  method: M,
  params: RpcMethods[M]['params']
): Promise<{ error: null | string; value: null | RpcMethods[M]['result'] }> {
  try {
    return { error: null, value: await requestGateway(method, params) }
  } catch (error) {
    return { error: toErrorMessage(error), value: null }
  }
}

export async function fetchRuntimeReadinessSignals(
  requestGateway: RuntimeReadinessRequester,
  requestedProvider?: string
): Promise<RuntimeReadinessSignals> {
  const runtimeParams = requestedProvider?.trim() ? { provider: requestedProvider.trim() } : {}

  const [setup, runtime] = await Promise.all([
    requestWithFallback(requestGateway, 'setup.status', {}),
    requestWithFallback(requestGateway, 'setup.runtime_check', runtimeParams)
  ])

  return {
    setup: setup.value,
    setupError: setup.error,
    runtime: runtime.value,
    runtimeError: runtime.error
  }
}

export function interpretRuntimeReadiness(
  signals: RuntimeReadinessSignals,
  options: RuntimeReadinessOptions = {}
): RuntimeReadinessResult {
  const defaultReason = options.defaultReason ?? DEFAULT_NOT_READY_REASON
  const unknownReady = options.unknownReady ?? false

  const setupConfigured =
    typeof signals.setup?.provider_configured === 'boolean' ? Boolean(signals.setup.provider_configured) : undefined

  const runtimeOk = typeof signals.runtime?.ok === 'boolean' ? Boolean(signals.runtime.ok) : undefined
  const runtimeFailure = normalizeMessage(signals.runtime?.error) ?? normalizeMessage(signals.runtimeError)
  const setupFailure = normalizeMessage(signals.setupError)

  // Route facts the check reported, carried through untouched so consumers
  // (free-tier chrome) don't have to re-issue setup.runtime_check. Left
  // undefined when the check said nothing — "absent" and "false" differ.
  const route = {
    freeTier: typeof signals.runtime?.free_tier === 'boolean' ? signals.runtime.free_tier : undefined,
    model: normalizeMessage(signals.runtime?.model) ?? undefined
  }

  const checksDisagree =
    typeof setupConfigured === 'boolean' && typeof runtimeOk === 'boolean' && setupConfigured !== runtimeOk

  if (typeof runtimeOk === 'boolean') {
    if (runtimeOk) {
      return {
        ...route,
        checksDisagree,
        ready: true,
        reason: null,
        source: 'runtime_check'
      }
    }

    let reason = runtimeFailure ?? defaultReason

    if (checksDisagree && setupConfigured) {
      reason = `${reason} setup.status reports configured credentials, but runtime resolution still failed.`
    }

    return {
      ...route,
      checksDisagree,
      ready: false,
      reason,
      source: 'runtime_check'
    }
  }

  if (typeof setupConfigured === 'boolean') {
    return {
      ...route,
      checksDisagree: false,
      ready: setupConfigured,
      reason: setupConfigured ? null : (runtimeFailure ?? setupFailure ?? defaultReason),
      source: 'setup_status'
    }
  }

  return {
    ...route,
    checksDisagree: false,
    ready: unknownReady,
    reason: unknownReady ? null : (runtimeFailure ?? setupFailure ?? defaultReason),
    source: 'fallback'
  }
}

export function runtimeReadinessDisplay(status: RuntimeReadinessResult | null): RuntimeReadinessDisplay {
  if (status === null) {
    return 'checking'
  }

  if (status.ready) {
    return 'ready'
  }

  // Credentials exist but runtime resolution failed. Calling that "needs
  // setup" sends users back through onboarding for provider/quota failures
  // that setup cannot repair; the reason tooltip carries the specific cause.
  return status.checksDisagree ? 'unavailable' : 'needs_setup'
}

export async function evaluateRuntimeReadiness(
  requestGateway: RuntimeReadinessRequester,
  options: RuntimeReadinessOptions = {}
): Promise<RuntimeReadinessResult> {
  const signals = await fetchRuntimeReadinessSignals(requestGateway, options.requestedProvider)

  return interpretRuntimeReadiness(signals, options)
}
