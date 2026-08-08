import { isGatewayAuthRejection } from './connection-config'

export interface OauthRestRequestDeps<T> {
  ensureNativeAccessToken: (baseUrl: string) => Promise<string | null>
  requestWithBearer: (accessToken: string) => Promise<T>
  requestWithCookie: () => Promise<T>
}

/**
 * Prefer a native bearer for an OAuth-gated REST request while preserving the
 * cookie session as a coexistence fallback.
 *
 * A native refresh error does not skip the cookie path: a live cookie may
 * still serve the request. If that cookie path rejects authentication too,
 * surface the original native error so a refresh timeout is not mislabeled as
 * an expired login merely because the cookie jar is empty.
 */
export async function requestWithOauthFallback<T>(
  baseUrl: string,
  deps: OauthRestRequestDeps<T>
): Promise<T> {
  let nativeAccessToken: string | null = null
  let nativeError: unknown
  let nativeFailed = false

  try {
    nativeAccessToken = await deps.ensureNativeAccessToken(baseUrl)
  } catch (error) {
    nativeError = error
    nativeFailed = true
  }

  if (nativeAccessToken) {
    return deps.requestWithBearer(nativeAccessToken)
  }

  try {
    return await deps.requestWithCookie()
  } catch (cookieError) {
    if (nativeFailed && isGatewayAuthRejection(cookieError)) {
      throw nativeError
    }

    throw cookieError
  }
}
