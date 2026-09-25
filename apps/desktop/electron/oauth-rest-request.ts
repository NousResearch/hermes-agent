import { readStatusCode } from './api-transport'
import { isCloudRateLimited } from './cloud-auth-errors'
import { isGatewayAuthRejection, withTransientRetries } from './connection-config'
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

/** Ticket minting is replay-safe, unlike arbitrary REST mutations. */
export async function mintGatewayWsTicket(
  baseUrl: string,
  deps: MintGatewayWsTicketDeps,
  headers: Record<string, string> = {}
): Promise<string> {
  const url = `${baseUrl}/api/auth/ws-ticket`
  const options = { method: 'POST', timeoutMs: 8_000, headers }

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

/**
 * main.ts's ws-ticket mint: the replay-safe mint above inside the transient
 * retry loop. Transport blips (brief host unreachable, 5xx, timeouts) retry so
 * a 1-3s flap does not become the "couldn't start" lockout. Never retried:
 * an auth rejection (hammers a dead session), a binding change (the caller's
 * next connect uses the fresh binding), and a 429 (Hermes Cloud exchange rate
 * limit / its Retry-After backoff — fail this mint, the next connect retries
 * once the backoff has passed).
 */
export function mintGatewayWsTicketWithRetries(
  baseUrl: string,
  deps: MintGatewayWsTicketDeps,
  headers: Record<string, string> = {},
  retry: { sleep?: (ms: number) => Promise<unknown> } = {}
): Promise<string> {
  return withTransientRetries(() => mintGatewayWsTicket(baseUrl, deps, headers), {
    ...retry,
    isRetryable: (error: unknown) =>
      !(error instanceof NativeAuthChangedError) && !isGatewayAuthRejection(error) && !isCloudRateLimited(error)
  })
}
