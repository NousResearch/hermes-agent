/**
 * The two Hermes Cloud auth verdicts every cloud surface must agree on:
 *
 *   - needsCloudLogin: there is no usable desktop session with the portal;
 *     the user has to sign in again in the browser.
 *   - cloudAgentAccessLost: the portal session is fine, but it will not mint
 *     a token for THIS agent (§5 invalid_target / a persisting invalid_grant)
 *     — the user lost access; retrying cannot help.
 *
 * Both are tagged Errors so discovery, per-agent sign-in and the background
 * re-exchange classify them the same way (and never loop on them).
 */

export const CLOUD_NOT_SIGNED_IN_MESSAGE =
  'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.'

export const CLOUD_SESSION_EXPIRED_MESSAGE =
  'Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.'

export const CLOUD_AGENT_ACCESS_LOST_MESSAGE =
  'You no longer have access to this Hermes Cloud agent. Ask an organization admin for access, or sign in to Hermes Cloud again and choose another agent.'

/**
 * §5 is rate limited per desktop session (429 `slow_down` + Retry-After).
 * A rate-limited exchange is transient: never an auth verdict, never retried
 * in a tight loop.
 */
export const CLOUD_EXCHANGE_RATE_LIMITED_MESSAGE = (retryAfterSeconds: number) =>
  `Hermes Cloud is rate-limiting agent sign-ins. Try again in ${Math.max(1, Math.ceil(retryAfterSeconds))} seconds.`

/**
 * Portal discovery could not confirm which agent a dashboard URL belongs to
 * (network / portal failure). Transient: never a sign-out or access-lost
 * verdict, and nothing is exchanged from an unconfirmed binding meanwhile.
 */
export const CLOUD_DISCOVERY_UNAVAILABLE_MESSAGE =
  'Could not reach Hermes Cloud to confirm this agent. Try again in a moment.'

export type CloudLoginRequiredError = Error & { needsCloudLogin: true; cause?: unknown }
export type CloudAgentAccessLostError = Error & { cloudAgentAccessLost: true; cause?: unknown }

export function cloudLoginRequiredError(message: string, cause?: unknown): CloudLoginRequiredError {
  const error = new Error(message) as CloudLoginRequiredError
  error.needsCloudLogin = true

  if (cause !== undefined) {
    error.cause = cause
  }

  return error
}

export function cloudAgentAccessLostError(cause?: unknown): CloudAgentAccessLostError {
  const error = new Error(CLOUD_AGENT_ACCESS_LOST_MESSAGE) as CloudAgentAccessLostError
  error.cloudAgentAccessLost = true

  if (cause !== undefined) {
    error.cause = cause
  }

  return error
}

export type CloudExchangeRateLimitedError = Error & {
  cloudRateLimited: true
  statusCode: 429
  retryAfterSeconds: number
  cause?: unknown
}

export function cloudExchangeRateLimitedError(
  retryAfterSeconds: number,
  cause?: unknown
): CloudExchangeRateLimitedError {
  const error = new Error(CLOUD_EXCHANGE_RATE_LIMITED_MESSAGE(retryAfterSeconds)) as CloudExchangeRateLimitedError
  error.cloudRateLimited = true
  // Same shape as an HTTP 429 so status-based classifiers agree.
  error.statusCode = 429
  error.retryAfterSeconds = retryAfterSeconds

  if (cause !== undefined) {
    error.cause = cause
  }

  return error
}

/** A 429 (portal §5 rate limit, or our own backoff gate). Transient, never auth loss. */
export function isCloudRateLimited(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const tagged = error as { cloudRateLimited?: unknown; statusCode?: unknown }

  return tagged.cloudRateLimited === true || Number(tagged.statusCode) === 429
}

export function isCloudLoginRequired(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && (error as { needsCloudLogin?: unknown }).needsCloudLogin === true
  )
}

/**
 * Renderer-side check. Electron's ipcRenderer.invoke rejects with a plain
 * Error whose message wraps the main-process message and drops custom
 * properties, so the stable needsCloudLogin messages are the cross-process
 * discriminator (same approach as the pool slot-timeout phrase).
 */
export function isCloudLoginRequiredErrorLike(error: unknown): boolean {
  if (isCloudLoginRequired(error)) {
    return true
  }

  const message = error instanceof Error ? error.message : typeof error === 'string' ? error : ''

  return message.includes(CLOUD_NOT_SIGNED_IN_MESSAGE) || message.includes(CLOUD_SESSION_EXPIRED_MESSAGE)
}

export function isCloudAgentAccessLost(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && (error as { cloudAgentAccessLost?: unknown }).cloudAgentAccessLost === true
  )
}

export type CloudDiscoveryUnavailableError = Error & { cloudDiscoveryUnavailable: true; cause?: unknown }

export function cloudDiscoveryUnavailableError(cause?: unknown): CloudDiscoveryUnavailableError {
  const error = new Error(CLOUD_DISCOVERY_UNAVAILABLE_MESSAGE) as CloudDiscoveryUnavailableError
  error.cloudDiscoveryUnavailable = true

  if (cause !== undefined) {
    error.cause = cause
  }

  return error
}

export function isCloudDiscoveryUnavailable(error: unknown): boolean {
  return Boolean(
    error &&
    typeof error === 'object' &&
    (error as { cloudDiscoveryUnavailable?: unknown }).cloudDiscoveryUnavailable === true
  )
}
