export interface SetupStatusSnapshot {
  provider_configured?: boolean
  /** Additive launch-profile fields (newer backends only; absent on older
   *  ones). Carried for consumers that read the record — readiness itself
   *  still keys on `provider_configured` + `setup.runtime_check`. */
  ready?: boolean
  free_tier?: boolean
  other_providers?: boolean
  inference_provider?: string
  /** Present only when the boot bootstrap could not create the free-tier
   *  identity: the failure code, its sentence, and whether / when a retry can
   *  succeed. Same shape as `free_tier.status`. */
  error?: string
  error_code?: string
  retryable?: boolean
  retry_after?: number
}

export interface RuntimeCheckSnapshot {
  error?: string
  /** True when the resolved route is the free tier rather than a credential of
   *  the user's own. Absent on older backends. */
  free_tier?: boolean
  model?: string
  ok?: boolean
  provider?: string
}

export interface RuntimeReadinessSignals {
  setup: null | SetupStatusSnapshot
  setupError: null | string
  runtime: null | RuntimeCheckSnapshot
  runtimeError: null | string
}

export interface RuntimeReadinessOptions {
  defaultReason?: string
  /** Profile whose home and credentials the probe must answer for: the owner of
   *  the session the client is about to open. Absent (or a launch-scope alias)
   *  probes the backend's own home, which is what a single-profile pod serves. */
  profile?: string
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

export type RuntimeReadinessRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

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

/** Launch-scope aliases that must never ride as a *named* profile. `default` is the
 *  backend's alias for its own home and `current` is the sentinel the OAuth routes
 *  accept for "whoever is asking"; the readiness routes resolve a name instead, so
 *  either would be answered against a strict named scope (skipping the boot
 *  free-tier record and host-wide credential fallbacks) or refused when no such
 *  profile directory exists. Exact match only: a profile directory that happens
 *  to be named `Default` is a real named scope and must be probed as one. */
const LAUNCH_PROFILE_ALIASES = new Set(['current', 'default'])

/** The profile to probe, or undefined for the backend's own (launch) home. */
function probeProfile(profile: string | undefined): string | undefined {
  const next = profile?.trim()

  return next && !LAUNCH_PROFILE_ALIASES.has(next) ? next : undefined
}

/** Params for one readiness RPC, or undefined when the call carries no dimensions:
 *  an ownerless probe stays byte-identical to the historical unscoped call. */
function readinessParams(requestedProvider?: string, profile?: string): Record<string, unknown> | undefined {
  const params: Record<string, unknown> = {}
  const provider = requestedProvider?.trim()
  const owner = probeProfile(profile)

  if (provider) {
    params.provider = provider
  }

  if (owner) {
    params.profile = owner
  }

  return Object.keys(params).length > 0 ? params : undefined
}

export async function fetchRuntimeReadinessSignals(
  requestGateway: RuntimeReadinessRequester,
  requestedProvider?: string,
  profile?: string
): Promise<RuntimeReadinessSignals> {
  // One object per RPC: setup.status takes only the owner while setup.runtime_check
  // also takes the provider, so the two calls must not share a payload.
  const setupParams = readinessParams(undefined, profile)
  const runtimeParams = readinessParams(requestedProvider, profile)

  const [setup, runtime] = await Promise.all([
    requestWithFallback<SetupStatusSnapshot>(requestGateway, 'setup.status', setupParams),
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
  const signals = await fetchRuntimeReadinessSignals(requestGateway, options.requestedProvider, options.profile)

  return interpretRuntimeReadiness(signals, options)
}
