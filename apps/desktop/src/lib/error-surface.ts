// Structured turn-error descriptor forwarded by the gateway (see
// agent/error_surface.py). Names WHICH layer of the stack failed and the
// classifier's failure code so the error card can say what happened in plain
// words and offer code-appropriate recovery actions, instead of toasting an
// opaque string.
//
// Advisory contract: older backends never send this — every consumer must
// keep working when it is absent (legacy string-sniffing stays as fallback).

export const ERROR_SURFACE_LAYERS = [
  'provider',
  'endpoint',
  'streaming',
  'auth',
  'billing',
  'gateway',
  'runtime',
  'disk'
] as const

export type ErrorSurfaceLayer = (typeof ERROR_SURFACE_LAYERS)[number]

/** Failure codes the error card has dedicated copy for. Values are what the
 *  backend stamps in `failure_reason` (agent/error_classifier.py
 *  FailoverReason) plus the gateway's own site codes (SESSION_NOT_OWNED,
 *  disk_full, stream_drop). Anything else falls back to the layer copy. */
export const ERROR_CODE_KEYS = [
  'auth',
  'auth_permanent',
  'billing',
  'rate_limit',
  'upstream_rate_limit',
  'overloaded',
  'server_error',
  'timeout',
  'stream_drop',
  'ssl_cert_verification',
  'context_overflow',
  'payload_too_large',
  'model_not_found',
  'provider_policy_blocked',
  'content_policy_blocked',
  'format_error',
  'truncated',
  'invalid_response',
  'empty_response',
  'loop_error',
  'SESSION_NOT_OWNED',
  'disk_full',
  // The Nous free tier refused or could not serve the turn (agent/error_surface.py
  // `free_tier_<kind>`). The backend's sentence rides in `message` and is the card body.
  'free_tier_disabled',
  'free_tier_rate_limited',
  'free_tier_at_capacity',
  'free_tier_model_not_free',
  'free_tier_route',
  'free_tier_outage',
  'free_tier_refused'
] as const

export type ErrorCodeKey = (typeof ERROR_CODE_KEYS)[number]

export interface ErrorSurface {
  layer: ErrorSurfaceLayer
  /** Specific failure code (a FailoverReason value or site-specific code). */
  code: string
  /** False when retrying unchanged reproduces the same failure. */
  retryable: boolean
  /** The failing session's provider/model, captured at classification time —
   *  preferred over the foreground composer's atoms, which can point at a
   *  different model by the time the user clicks an action. */
  provider?: string
  model?: string
  /** Auth layer only: how the failing provider is credentialed. `oauth` means
   *  the fix is signing in again (expired/revoked grant); `api_key` means a
   *  key needs replacing. Absent from older backends. */
  authKind?: 'api_key' | 'oauth'
  /** Auth layer only: display name of the failing provider ("Nous Portal"). */
  providerLabel?: string
  /** Auth layer, api_key only: the env var holding the rejected key
   *  (OPENAI_API_KEY). Deep-links Settings → Keys to that row. Absent from
   *  older backends. */
  apiKeyEnv?: string
  /** Free-tier codes: the backend's own plain sentence for this failure (it
   *  names the wait, the model, the way forward). Shown as the card body. */
  message?: string
}

/** Validate a wire payload into an ErrorSurface, or null when absent/garbled. */
export function parseErrorSurface(value: unknown): ErrorSurface | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as {
    api_key_env?: unknown
    auth_kind?: unknown
    code?: unknown
    layer?: unknown
    message?: unknown
    model?: unknown
    provider?: unknown
    provider_label?: unknown
    retryable?: unknown
  }

  const layer = typeof raw.layer === 'string' ? (raw.layer as ErrorSurfaceLayer) : null

  if (!layer || !ERROR_SURFACE_LAYERS.includes(layer)) {
    return null
  }

  return {
    layer,
    code: typeof raw.code === 'string' && raw.code ? raw.code : 'unknown',
    retryable: raw.retryable !== false,
    ...(typeof raw.provider === 'string' && raw.provider ? { provider: raw.provider } : {}),
    ...(typeof raw.model === 'string' && raw.model ? { model: raw.model } : {}),
    ...(raw.auth_kind === 'oauth' || raw.auth_kind === 'api_key' ? { authKind: raw.auth_kind } : {}),
    ...(typeof raw.provider_label === 'string' && raw.provider_label ? { providerLabel: raw.provider_label } : {}),
    ...(typeof raw.api_key_env === 'string' && raw.api_key_env ? { apiKeyEnv: raw.api_key_env } : {}),
    ...(typeof raw.message === 'string' && raw.message.trim() ? { message: raw.message.trim() } : {})
  }
}

/** True when the Nous free tier refused or could not serve the turn: the way
 *  forward is the free sign-in (or another provider), never an OAuth re-login. */
export function isFreeTierSurface(surface: ErrorSurface | null | undefined): boolean {
  return typeof surface?.code === 'string' && surface.code.startsWith('free_tier_')
}

/** True when the failed turn's provider rejected an OAuth grant — the
 *  one-click recovery is re-running that provider's sign-in, not editing keys. */
export function isOAuthReauthSurface(surface: ErrorSurface | null | undefined): surface is ErrorSurface & {
  authKind: 'oauth'
  provider: string
} {
  return surface?.layer === 'auth' && surface.authKind === 'oauth' && Boolean(surface.provider)
}

/** True when the failed turn's provider rejected a saved API key — the fix is
 *  replacing the key in Settings → Keys, then retrying. */
export function isApiKeyRejectedSurface(surface: ErrorSurface | null | undefined): surface is ErrorSurface & {
  authKind: 'api_key'
} {
  return surface?.layer === 'auth' && surface.authKind === 'api_key'
}

/** Which copy entry the error card (and the matching toast) should read:
 *  the code table when the code is one we have words for, else the layer
 *  table, else `generic` (no descriptor — older backend). One resolver so the
 *  inline card and the global toast never disagree about the same failure. */
export type ErrorCardKey = { code: ErrorCodeKey } | { layer: 'generic' | ErrorSurfaceLayer }

export function errorCardKey(surface: ErrorSurface | null | undefined): ErrorCardKey {
  if (!surface) {
    return { layer: 'generic' }
  }

  return (ERROR_CODE_KEYS as readonly string[]).includes(surface.code)
    ? { code: surface.code as ErrorCodeKey }
    : { layer: surface.layer }
}

/** Which recovery buttons the error card offers for a failure. Every flag
 *  maps to an EXISTING handler in the app; the card only decides visibility. */
export interface ErrorRecoveryPlan {
  /** assistant-ui reload of the failed turn. */
  retry: boolean
  /** Open the model picker (model_not_found). */
  chooseModel: boolean
  /** Run /compress on this session (context too long). */
  compress: boolean
  /** requestFreshSession — when this session cannot continue as-is. */
  startNewSession: boolean
  /** Open the preceding user message in the edit composer (safety refusal). */
  editMessage: boolean
  /** Reveal the Hermes data folder so the user can free space (disk_full). */
  openHermesFolder: boolean
  /** Settings → Keys deep link (auth, api_key). */
  updateApiKey: boolean
  /** Re-run the provider's OAuth sign-in (auth, oauth). */
  signInAgain: boolean
  /** Open the free-tier sign-in dialog (free_tier_* codes): signing in is free and lifts the refusal. */
  signInFreeTier: boolean
  /** Settings → Models deep link. */
  switchProvider: boolean
}

// Layers where the fix is provider/endpoint/auth config, not a retry.
const SWITCH_PROVIDER_LAYERS: readonly ErrorSurfaceLayer[] = ['auth', 'billing', 'endpoint', 'provider']

// Per-code overrides on top of the layer defaults. `retry: false` here means
// the classifier may call the failure retryable at the transport level, but
// an unchanged retry is known to reproduce it for the user (too long a
// conversation, a blocked prompt, a chat another surface owns).
const CODE_PLANS: Partial<Record<ErrorCodeKey, Partial<ErrorRecoveryPlan>>> = {
  SESSION_NOT_OWNED: { retry: false, startNewSession: true },
  content_policy_blocked: { editMessage: true, retry: false },
  context_overflow: { compress: true, retry: false, startNewSession: true },
  disk_full: { openHermesFolder: true, retry: true },
  loop_error: { startNewSession: true },
  model_not_found: { chooseModel: true, retry: false },
  payload_too_large: { compress: true, retry: false, startNewSession: true }
}

export function errorRecoveryPlan(surface: ErrorSurface | null | undefined): ErrorRecoveryPlan {
  const oauthReauth = isOAuthReauthSurface(surface)
  const apiKeyRejected = isApiKeyRejectedSurface(surface)

  const base: ErrorRecoveryPlan = {
    chooseModel: false,
    compress: false,
    editMessage: false,
    openHermesFolder: false,
    // Retry re-runs the failed prompt in place. Suppressed when the classifier
    // says the failure is deterministic — except for a credential rejection,
    // where fixing the credential changes the outcome and Retry is the
    // natural second click.
    retry: !surface || surface.retryable || oauthReauth || apiKeyRejected,
    signInAgain: oauthReauth,
    signInFreeTier: isFreeTierSurface(surface),
    startNewSession: false,
    switchProvider: surface != null && SWITCH_PROVIDER_LAYERS.includes(surface.layer),
    updateApiKey: apiKeyRejected
  }

  const key = errorCardKey(surface)

  return { ...base, ...('code' in key ? CODE_PLANS[key.code] : undefined) }
}

/** Plain-text error-details blob for the error card's "Copy error details". */
export function formatErrorDiagnostics(input: {
  appVersion?: string
  errorText: string
  model?: string
  provider?: string
  surface?: ErrorSurface | null
}): string {
  // The descriptor's identity (captured when the turn failed) beats the
  // caller-supplied fallback (typically the foreground composer's atoms).
  const provider = input.surface?.provider || input.provider
  const model = input.surface?.model || input.model

  const lines = [
    '── Hermes error details ──',
    `time: ${new Date().toISOString()}`,
    input.surface ? `layer: ${input.surface.layer}` : null,
    input.surface ? `code: ${input.surface.code}` : null,
    input.surface ? `retryable: ${input.surface.retryable}` : null,
    provider ? `provider: ${provider}` : null,
    model ? `model: ${model}` : null,
    input.appVersion ? `app: ${input.appVersion}` : null,
    `error: ${input.errorText}`
  ]

  return lines.filter((line): line is string => Boolean(line)).join('\n')
}
