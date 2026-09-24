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

export function isCloudLoginRequired(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && (error as { needsCloudLogin?: unknown }).needsCloudLogin === true
  )
}

export function isCloudAgentAccessLost(error: unknown): boolean {
  return Boolean(
    error && typeof error === 'object' && (error as { cloudAgentAccessLost?: unknown }).cloudAgentAccessLost === true
  )
}
