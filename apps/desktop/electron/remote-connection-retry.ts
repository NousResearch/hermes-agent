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
      if (attempt >= attempts || requiresOauthLogin(error)) {
        throw error
      }

      onRetry?.(error, attempt, delayMs)
      await sleep(delayMs)
      delayMs = Math.min(Math.max(delayMs * 2, initialDelayMs), maxDelayMs)
    }
  }

  throw new Error('Remote connection retry loop exhausted unexpectedly')
}

export { requiresOauthLogin, resolveRemoteConnectionWithRetry }
export type { RemoteConnectionRetryOptions }
