import type { NativeTokenSet } from './native-oauth'

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

export interface NativeAccessTokenCoordinator {
  ensure: (baseUrl: string) => Promise<string | null>
  invalidateExplicitAuthChange: (rawBaseUrl: string) => void
}

/**
 * Coordinate native-token refreshes for the Electron main process.
 *
 * One module-level coordinator instance owns one flight map and scoped auth epochs.
 * Equivalent gateway URLs share a flight after normalization. Explicit login
 * and logout advance the relevant epoch so a response that started under older auth
 * state can neither overwrite a new login nor resurrect a logout.
 */
export function createNativeAccessTokenCoordinator(
  deps: NativeAccessTokenCoordinatorDeps
): NativeAccessTokenCoordinator {
  const refreshFlights = new Map<string, Promise<string | null>>()
  const authEpochs = new Map<string, number>()
  const epochFor = (baseUrl: string) => authEpochs.get(baseUrl) ?? 0

  async function ensure(rawBaseUrl: string): Promise<string | null> {
    const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
    const tokens = deps.loadTokens(baseUrl)

    if (!tokens) {
      return null
    }

    const nowSeconds = deps.nowSeconds?.() ?? Math.floor(Date.now() / 1_000)

    if (!deps.tokenNeedsRefresh(tokens, nowSeconds)) {
      return tokens.accessToken
    }

    if (!tokens.refreshToken) {
      deps.clearTokens(baseUrl)

      return null
    }

    const existingFlight = refreshFlights.get(baseUrl)

    if (existingFlight) {
      return existingFlight
    }

    const flightEpoch = epochFor(baseUrl)
    const sentRefreshToken = tokens.refreshToken

    const refreshFlight = (async (): Promise<string | null> => {
      try {
        const rotated = await deps.refreshTokens(baseUrl, tokens)

        if (epochFor(baseUrl) !== flightEpoch) {
          return null
        }

        deps.storeTokens(baseUrl, rotated)

        return rotated.accessToken
      } catch (error) {
        if (epochFor(baseUrl) !== flightEpoch) {
          return null
        }

        if (deps.isRefreshAuthRejection(error)) {
          const stored = deps.loadTokens(baseUrl)

          if (stored?.refreshToken === sentRefreshToken) {
            deps.clearTokens(baseUrl)
          }

          return null
        }

        throw error
      }
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
    invalidateExplicitAuthChange: rawBaseUrl => {
      const baseUrl = deps.normalizeBaseUrl(rawBaseUrl)
      authEpochs.set(baseUrl, (authEpochs.get(baseUrl) ?? 0) + 1)
    }
  }
}
