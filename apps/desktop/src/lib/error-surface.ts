// Structured turn-error descriptor forwarded by the gateway (see
// agent/error_surface.py). Names WHICH layer of the stack failed so the error
// card can say "Provider error" / "Gateway error" and offer layer-appropriate
// recovery actions, instead of toasting an opaque string.
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
}

/** Validate a wire payload into an ErrorSurface, or null when absent/garbled. */
export function parseErrorSurface(value: unknown): ErrorSurface | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as {
    auth_kind?: unknown
    code?: unknown
    layer?: unknown
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
    ...(typeof raw.provider_label === 'string' && raw.provider_label ? { providerLabel: raw.provider_label } : {})
  }
}

/** True when the failed turn's provider rejected an OAuth grant — the
 *  one-click recovery is re-running that provider's sign-in, not editing keys. */
export function isOAuthReauthSurface(surface: ErrorSurface | null | undefined): surface is ErrorSurface & {
  authKind: 'oauth'
  provider: string
} {
  return surface?.layer === 'auth' && surface.authKind === 'oauth' && Boolean(surface.provider)
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
