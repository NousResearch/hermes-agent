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
}

/** Validate a wire payload into an ErrorSurface, or null when absent/garbled. */
export function parseErrorSurface(value: unknown): ErrorSurface | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as { code?: unknown; layer?: unknown; retryable?: unknown }
  const layer = typeof raw.layer === 'string' ? (raw.layer as ErrorSurfaceLayer) : null

  if (!layer || !ERROR_SURFACE_LAYERS.includes(layer)) {
    return null
  }

  return {
    layer,
    code: typeof raw.code === 'string' && raw.code ? raw.code : 'unknown',
    retryable: raw.retryable !== false
  }
}

/** Plain-text error-details blob for the error card's "Copy error details". */
export function formatErrorDiagnostics(input: {
  appVersion?: string
  errorText: string
  model?: string
  provider?: string
  surface?: ErrorSurface | null
}): string {
  const lines = [
    '── Hermes error diagnostics ──',
    `time: ${new Date().toISOString()}`,
    input.surface ? `layer: ${input.surface.layer}` : null,
    input.surface ? `code: ${input.surface.code}` : null,
    input.surface ? `retryable: ${input.surface.retryable}` : null,
    input.provider ? `provider: ${input.provider}` : null,
    input.model ? `model: ${input.model}` : null,
    input.appVersion ? `app: ${input.appVersion}` : null,
    `error: ${input.errorText}`
  ]

  return lines.filter((line): line is string => Boolean(line)).join('\n')
}
