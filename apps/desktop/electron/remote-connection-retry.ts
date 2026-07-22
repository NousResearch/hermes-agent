import { isGatewayAuthRejection } from './connection-config'

interface RemoteConnectionRetryOptions {
  initialDelayMs?: number
  maxAttempts?: number
  maxDelayMs?: number
  onRetry?: (error: unknown, attempt: number, delayMs: number) => void
  sleep?: (delayMs: number) => Promise<void>
}

function requiresOauthLogin(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }

  const candidate = error as { cause?: unknown; needsOauthLogin?: unknown }

  return candidate.needsOauthLogin === true || isGatewayAuthRejection(error) || isGatewayAuthRejection(candidate.cause)
}

const RETRYABLE_NETWORK_CODES = new Set([
  'EAI_AGAIN',
  'ECONNREFUSED',
  'ECONNRESET',
  'EHOSTUNREACH',
  'ENETDOWN',
  'ENETUNREACH',
  'ENOTFOUND',
  'ETIMEDOUT',
  'ERR_INTERNET_DISCONNECTED',
  'UND_ERR_CONNECT_TIMEOUT',
  'UND_ERR_SOCKET'
])

/** True only for failures that can plausibly clear without changing config. */
function isRetryableRemoteConnectionError(error: unknown, seen = new Set<object>()): boolean {
  if (!error || typeof error !== 'object' || requiresOauthLogin(error) || seen.has(error)) {
    return false
  }

  seen.add(error)

  const candidate = error as {
    cause?: unknown
    code?: unknown
    kind?: unknown
    message?: unknown
    statusCode?: unknown
  }

  const statusCode = Number(candidate.statusCode)

  if (statusCode === 408 || statusCode === 425 || statusCode === 429 || (statusCode >= 500 && statusCode <= 599)) {
    return true
  }

  if (candidate.kind === 'timeout' || candidate.kind === 'unreachable') {
    return true
  }

  if (typeof candidate.code === 'string' && RETRYABLE_NETWORK_CODES.has(candidate.code.toUpperCase())) {
    return true
  }

  if (candidate.cause && isRetryableRemoteConnectionError(candidate.cause, seen)) {
    return true
  }

  // Some timeout wrappers (including the desktop's bounded ticket request)
  // carry no structured code. Keep this narrow: configuration messages such as
  // invalid URLs, missing tokens, invalid SSH settings, and auth failures do not
  // match and therefore fail immediately.
  return typeof candidate.message === 'string' && /(?:timed? out|timeout)/i.test(candidate.message)
}

/**
 * A remote restart can outlive one 8-second ticket/status request. Retry only
 * ordinary transport/server failures; a positively classified 401/403 must
 * immediately surface the sign-in flow instead of being hidden by backoff.
 *
 * Calling `resolve` again is intentional: OAuth WebSocket tickets are
 * single-use, so every attempt must mint a fresh ticket rather than reuse one
 * captured by an earlier attempt.
 */
async function resolveRemoteConnectionWithRetry<T>(
  resolve: () => Promise<T>,
  {
    initialDelayMs = 500,
    maxAttempts = 5,
    maxDelayMs = 4_000,
    onRetry,
    sleep = delayMs => new Promise<void>(done => setTimeout(done, delayMs))
  }: RemoteConnectionRetryOptions = {}
): Promise<T> {
  const attempts = Math.max(1, Math.floor(maxAttempts))
  let delayMs = Math.max(0, initialDelayMs)

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await resolve()
    } catch (error) {
      if (attempt >= attempts || !isRetryableRemoteConnectionError(error)) {
        throw error
      }

      onRetry?.(error, attempt, delayMs)
      await sleep(delayMs)
      delayMs = Math.min(Math.max(delayMs * 2, initialDelayMs), maxDelayMs)
    }
  }

  throw new Error('Remote connection retry loop exhausted unexpectedly')
}

export { isRetryableRemoteConnectionError, requiresOauthLogin, resolveRemoteConnectionWithRetry }
export type { RemoteConnectionRetryOptions }
