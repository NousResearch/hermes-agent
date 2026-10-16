import { isNousCloudAgentUrl } from './backend-health'

interface CloudRecoveryDeps {
  hasNativeSession: (baseUrl: string) => boolean
  restoreCookieSession: (baseUrl: string) => Promise<boolean>
  onRecovered?: (baseUrl: string) => void
  now?: () => number
}

/** Coalesce the cookie refresh, never the single-use ticket minted afterward. */
export function createCloudSessionRecovery(deps: CloudRecoveryDeps) {
  const inflight = new Map<string, Promise<boolean>>()
  const failedAt = new Map<string, number>()
  const now = deps.now ?? Date.now

  return async <T>(baseUrl: string, mint: () => Promise<T>): Promise<T> => {
    const hadNativeSession = deps.hasNativeSession(baseUrl)

    try {
      return await mint()
    } catch (error) {
      if (
        !isNousCloudAgentUrl(baseUrl) ||
        hadNativeSession ||
        deps.hasNativeSession(baseUrl) ||
        (error as { statusCode?: number })?.statusCode !== 401
      ) {
        throw error
      }

      let recovery = inflight.get(baseUrl)

      if (!recovery) {
        const failed = failedAt.get(baseUrl)

        if (failed !== undefined && now() - failed < 60_000) {
          throw error
        }

        recovery = Promise.resolve()
          .then(() => deps.restoreCookieSession(baseUrl))
          .catch(() => false)
          .then(restored => {
            if (!restored) {
              failedAt.set(baseUrl, now())
            }

            return restored
          })
          .finally(() => inflight.delete(baseUrl))
        inflight.set(baseUrl, recovery)
      }

      if (!(await recovery)) {
        throw error
      }

      // Native login may have changed while the background cookie flow ran.
      // Never turn that transition into a cross-identity fallback.
      if (deps.hasNativeSession(baseUrl)) {
        throw error
      }

      try {
        const ticket = await mint()
        failedAt.delete(baseUrl)
        deps.onRecovered?.(baseUrl)

        return ticket
      } catch (retryError) {
        failedAt.set(baseUrl, now())
        throw retryError
      }
    }
  }
}
