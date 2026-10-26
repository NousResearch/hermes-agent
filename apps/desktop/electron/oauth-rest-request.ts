import { AsyncLocalStorage } from 'node:async_hooks'

import { readJsonErrorBody, readStatusCode } from './api-transport'
import { isGatewayAuthRejection } from './connection-config'
import { type NativeAccessTokenOptions, NativeAuthChangedError } from './native-access-token'
import { shouldRotateNativeTokenAfterRejection } from './native-auth-decisions'

export interface OauthRestRequestDeps<T> {
  ensureNativeAccessToken: (baseUrl: string, options?: NativeAccessTokenOptions) => Promise<string | null>
  requestWithBearer: (accessToken: string) => Promise<T>
  requestWithCookie: () => Promise<T>
}

async function cookieFallback<T>(request: () => Promise<T>, nativeError: unknown): Promise<T> {
  // An old request must not cross into a newly selected identity's cookie jar.
  if (nativeError instanceof NativeAuthChangedError) {
    throw nativeError
  }

  try {
    return await request()
  } catch (cookieError) {
    if (isGatewayAuthRejection(cookieError)) {
      throw nativeError
    }

    throw cookieError
  }
}

/** A confirmed bearer 401 is an app-token failure, not a server OAuth session. */
function markStaleAppToken(error: unknown): void {
  if (error && typeof error === 'object') {
    ;(error as { appTokenRejected?: boolean }).appTokenRejected = true
  }
}

/**
 * Native-first OAuth selection for REST, readiness, downloads and media.
 * Cookie coexistence can recover a failed refresh, but an empty cookie jar
 * cannot turn that transport failure into a terminal sign-in verdict.
 * Never replay a submitted request: it may be a non-idempotent mutation.
 */
export async function requestWithOauthFallback<T>(baseUrl: string, deps: OauthRestRequestDeps<T>): Promise<T> {
  let nativeAccessToken: string | null

  try {
    nativeAccessToken = await deps.ensureNativeAccessToken(baseUrl)
  } catch (error) {
    return cookieFallback(deps.requestWithCookie, error)
  }

  return nativeAccessToken ? deps.requestWithBearer(nativeAccessToken) : deps.requestWithCookie()
}

export interface MintGatewayWsTicketDeps {
  ensureNativeAccessToken: OauthRestRequestDeps<unknown>['ensureNativeAccessToken']
  fetchJson: (url: string, token: string | null, options: any) => Promise<any>
  fetchJsonViaOauthSession: (url: string, options: any) => Promise<any>
}

// Roster polling reads saved gateways in the background. Its auth failures
// belong on the source row, not in a login window over the active workspace.
const interactiveLoginAllowed = new AsyncLocalStorage<boolean>()

export function withoutInteractiveOauthLogin<T>(work: () => Promise<T>): Promise<T> {
  return interactiveLoginAllowed.run(false, work)
}

export function canShowInteractiveOauthLogin(): boolean {
  return interactiveLoginAllowed.getStore() !== false
}

export async function retryCookie401WithLogin<T>(
  error: unknown,
  options: { method?: unknown; replayOn401?: unknown },
  actions: { clearCookies: () => void; login: () => Promise<unknown>; retry: () => Promise<T> }
): Promise<T> {
  if (!canShowInteractiveOauthLogin() || !shouldReplayAfterCookie401(error, options)) {
    throw error
  }

  actions.clearCookies()

  try {
    await actions.login()
  } catch {
    throw error
  }

  return actions.retry()
}

/**
 * Whether a cookie-mode 401 may be answered by ONE silent re-login and a
 * single resubmission of the same request (#61457). The no-replay rule above
 * stands for arbitrary mutations; a replay needs BOTH:
 *   - pre-effect evidence: the 401 body carries the dashboard auth gate's
 *     structured `{error: unauthenticated|session_expired, reason}` shape,
 *     which the middleware emits before any route handler runs; and
 *   - a replay-safe operation: an idempotent method (GET/HEAD), or the caller
 *     vouching for the operation with `replayOn401: true` (ticket minting).
 */
export function shouldReplayAfterCookie401(
  error: unknown,
  options: { method?: unknown; replayOn401?: unknown } = {}
): boolean {
  if (readStatusCode(error) !== 401) {
    return false
  }

  const body = readJsonErrorBody(error)

  const gateRefusal =
    body !== null &&
    (body.error === 'unauthenticated' || body.error === 'session_expired') &&
    typeof body.reason === 'string' &&
    body.reason.length > 0

  if (!gateRefusal) {
    return false
  }

  const method = String(options.method || 'GET').toUpperCase()

  return method === 'GET' || method === 'HEAD' || options.replayOn401 === true
}

/** Ticket minting is replay-safe, unlike arbitrary REST mutations. */
export async function mintGatewayWsTicket(
  baseUrl: string,
  deps: MintGatewayWsTicketDeps,
  headers: Record<string, string> = {}
): Promise<string> {
  const url = `${baseUrl}/api/auth/ws-ticket`
  // replayOn401: a ticket that was never issued has no effect to double.
  const options = { method: 'POST', timeoutMs: 8_000, headers, replayOn401: true }

  const ticketFrom = async (request: Promise<any>): Promise<string> => {
    const body = await request

    if (!body?.ticket || typeof body.ticket !== 'string') {
      throw new Error('Gateway did not return a WS ticket.')
    }

    return body.ticket
  }

  const mintWithBearer = (bearer: string) => ticketFrom(deps.fetchJson(url, null, { ...options, bearer }))
  const requestWithCookie = () => ticketFrom(deps.fetchJsonViaOauthSession(url, options))

  return requestWithOauthFallback(baseUrl, {
    ensureNativeAccessToken: deps.ensureNativeAccessToken,
    requestWithCookie,
    requestWithBearer: async nativeAt => {
      try {
        try {
          return await mintWithBearer(nativeAt)
        } catch (error) {
          // Preserve #107990: the gate does not rotate a native bearer itself.
          // Only a structured 401 earns one forced refresh; never a 403.
          if (!shouldRotateNativeTokenAfterRejection(error)) {
            throw error
          }

          const rotatedAt = await deps.ensureNativeAccessToken(baseUrl, {
            forceRefresh: true,
            rejectedAccessToken: nativeAt
          })

          if (!rotatedAt || rotatedAt === nativeAt) {
            throw error
          }

          return await mintWithBearer(rotatedAt)
        }
      } catch (error) {
        if (readStatusCode(error) === 401) {
          markStaleAppToken(error)
        }

        return cookieFallback(requestWithCookie, error)
      }
    }
  })
}
