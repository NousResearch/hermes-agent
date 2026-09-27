import type { NativeTokenSet } from './native-oauth'

export interface NativeAccessTokenOptions {
  forceRefresh?: boolean
  /** A late 401 must reuse a concurrent rotation rather than rotate its winner again. */
  rejectedAccessToken?: string
}

export interface NativeAccessTokenCoordinatorDeps {
  clearTokens: (baseUrl: string) => void
  isRefreshAuthRejection: (error: unknown) => boolean
  loadTokens: (baseUrl: string) => NativeTokenSet | null
  normalizeBaseUrl: (baseUrl: string) => string
  nowSeconds?: () => number
  refreshTokens: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
  storeTokens: (baseUrl: string, tokens: NativeTokenSet) => void
  tokenNeedsRefresh: (tokens: NativeTokenSet, nowSeconds: number) => boolean
}

export class NativeAuthChangedError extends Error {
  constructor() {
    super('Authentication changed while the request was in progress. Try again.')
  }
}

/**
 * One owner for refresh flights and explicit native-token mutations. Per-host
 * token epochs fence refreshes; login epochs order pending browser flows without
 * discarding a valid rotation if a login is abandoned. Other hosts are independent.
 */
export function createNativeAccessTokenCoordinator(deps: NativeAccessTokenCoordinatorDeps) {
  const refreshFlights = new Map<string, Promise<string | null>>()
  const authEpochs = new Map<string, number>()
  const loginEpochs = new Map<string, number>()
  const epochFor = (baseUrl: string) => authEpochs.get(baseUrl) ?? 0

  function beginLogin(rawBaseUrl: string): () => boolean {
    const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
    const epoch = (loginEpochs.get(baseUrl) ?? 0) + 1
    loginEpochs.set(baseUrl, epoch)

    return () => loginEpochs.get(baseUrl) === epoch
  }

  function beginExplicitAuthChange(baseUrl: string): void {
    beginLogin(baseUrl)
    authEpochs.set(baseUrl, epochFor(baseUrl) + 1)
    refreshFlights.delete(baseUrl)
  }

  async function ensure(rawBaseUrl: string, options: NativeAccessTokenOptions = {}): Promise<string | null> {
    const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
    const existingFlight = refreshFlights.get(baseUrl)

    if (existingFlight) {
      return existingFlight
    }

    const tokens = deps.loadTokens(baseUrl)

    if (!tokens) {
      return null
    }

    const nowSeconds = deps.nowSeconds?.() ?? Math.floor(Date.now() / 1_000)
    const rejectedCurrentToken = !options.rejectedAccessToken || options.rejectedAccessToken === tokens.accessToken

    if (!(options.forceRefresh && rejectedCurrentToken) && !deps.tokenNeedsRefresh(tokens, nowSeconds)) {
      return tokens.accessToken
    }

    if (!tokens.refreshToken) {
      deps.clearTokens(baseUrl)

      return null
    }

    const flightEpoch = epochFor(baseUrl)

    const assertCurrent = () => {
      if (epochFor(baseUrl) !== flightEpoch) {
        throw new NativeAuthChangedError()
      }
    }

    const refreshFlight = (async (): Promise<string | null> => {
      let rotated: NativeTokenSet

      try {
        rotated = await deps.refreshTokens(baseUrl, tokens)
      } catch (error) {
        assertCurrent()

        if (deps.isRefreshAuthRejection(error)) {
          deps.clearTokens(baseUrl)

          return null
        }

        throw error
      }

      assertCurrent()
      deps.storeTokens(baseUrl, rotated)

      return rotated.accessToken
    })()

    refreshFlights.set(baseUrl, refreshFlight)

    try {
      return await refreshFlight
    } finally {
      if (refreshFlights.get(baseUrl) === refreshFlight) {
        refreshFlights.delete(baseUrl)
      }
    }
  }

  return {
    ensure,
    beginLogin,
    storeTokens: (rawBaseUrl: string, tokens: NativeTokenSet) => {
      const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
      beginExplicitAuthChange(baseUrl)
      deps.storeTokens(baseUrl, tokens)
    },
    clearTokens: (rawBaseUrl: string) => {
      const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
      beginExplicitAuthChange(baseUrl)
      deps.clearTokens(baseUrl)
    }
  }
}
