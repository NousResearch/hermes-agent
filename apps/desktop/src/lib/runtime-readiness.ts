import type { HermesConnection } from '@/global'

export interface SetupStatusSnapshot {
  profile_name?: string
  provider_configured?: boolean
}

export interface RuntimeCheckSnapshot {
  error?: string
  ok?: boolean
  profile_name?: string
}

export interface RuntimeReadinessSignals {
  setup: null | SetupStatusSnapshot
  setupError: null | string
  runtime: null | RuntimeCheckSnapshot
  runtimeError: null | string
}

export interface RuntimeReadinessOptions {
  defaultReason?: string
  profile?: string
  requireProfileIdentity?: boolean
  requestedProvider?: string
  unknownReady?: boolean
}

export interface RuntimeReadinessResult {
  checksDisagree: boolean
  ready: boolean
  reason: null | string
  source: 'fallback' | 'runtime_check' | 'setup_status'
}

export type RuntimeReadinessRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

const DEFAULT_NOT_READY_REASON = 'Add a provider credential before sending your first message.'

export function requiresProfileIdentity(
  connection: null | Pick<HermesConnection, 'mode' | 'source'> | undefined
): boolean {
  return connection?.mode === 'remote' && (connection.source === 'env' || connection.source === 'settings')
}

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

async function requestWithFallback<T>(
  requestGateway: RuntimeReadinessRequester,
  method: string,
  params?: Record<string, unknown>
): Promise<{ error: null | string; value: null | T }> {
  try {
    return { error: null, value: await requestGateway<T>(method, params) }
  } catch (error) {
    return { error: toErrorMessage(error), value: null }
  }
}

export async function fetchRuntimeReadinessSignals(
  requestGateway: RuntimeReadinessRequester,
  requestedProvider?: string,
  profile?: string
): Promise<RuntimeReadinessSignals> {
  const requestedProfile = profile?.trim()
  const profileParams = requestedProfile ? { profile: requestedProfile } : undefined
  const requestedRuntimeProvider = requestedProvider?.trim()

  const runtimeParams =
    profileParams || requestedRuntimeProvider
      ? { ...profileParams, ...(requestedRuntimeProvider ? { provider: requestedRuntimeProvider } : {}) }
      : undefined

  const [setup, runtime] = await Promise.all([
    requestWithFallback<SetupStatusSnapshot>(requestGateway, 'setup.status', profileParams),
    requestWithFallback<RuntimeCheckSnapshot>(requestGateway, 'setup.runtime_check', runtimeParams)
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

  const checksDisagree =
    typeof setupConfigured === 'boolean' && typeof runtimeOk === 'boolean' && setupConfigured !== runtimeOk

  if (options.requireProfileIdentity) {
    const requestedProfile = options.profile?.trim() || 'default'

    const resolvedProfile =
      typeof runtimeOk === 'boolean'
        ? signals.runtime?.profile_name?.trim()
        : typeof setupConfigured === 'boolean'
          ? signals.setup?.profile_name?.trim()
          : undefined

    if (!resolvedProfile) {
      return {
        checksDisagree: false,
        ready: false,
        reason: `Update the Hermes backend before Desktop can verify readiness for profile "${requestedProfile}".`,
        source: 'fallback'
      }
    }

    if (resolvedProfile !== requestedProfile) {
      return {
        checksDisagree: false,
        ready: false,
        reason: `Hermes resolved readiness for profile "${resolvedProfile}" instead of requested profile "${requestedProfile}".`,
        source: 'fallback'
      }
    }
  }

  if (typeof runtimeOk === 'boolean') {
    if (runtimeOk) {
      return {
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
      checksDisagree,
      ready: false,
      reason,
      source: 'runtime_check'
    }
  }

  if (typeof setupConfigured === 'boolean') {
    return {
      checksDisagree: false,
      ready: setupConfigured,
      reason: setupConfigured ? null : (runtimeFailure ?? setupFailure ?? defaultReason),
      source: 'setup_status'
    }
  }

  return {
    checksDisagree: false,
    ready: unknownReady,
    reason: unknownReady ? null : (runtimeFailure ?? setupFailure ?? defaultReason),
    source: 'fallback'
  }
}

export async function evaluateRuntimeReadiness(
  requestGateway: RuntimeReadinessRequester,
  options: RuntimeReadinessOptions = {}
): Promise<RuntimeReadinessResult> {
  // Only an app-global remote backend needs the Desktop's local profile label.
  // Local and per-profile remote connections already serve their own launch
  // profile, and that label may not exist on a dedicated remote host.
  const gatewayProfile = options.requireProfileIdentity ? options.profile : undefined
  const signals = await fetchRuntimeReadinessSignals(requestGateway, options.requestedProvider, gatewayProfile)

  return interpretRuntimeReadiness(signals, options)
}
