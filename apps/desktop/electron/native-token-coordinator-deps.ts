/**
 * native-token-coordinator-deps.ts
 *
 * The one wiring of the native access-token coordinator that main.ts uses,
 * kept out of main.ts so the real policy is unit-tested rather than
 * re-implemented in tests.
 *
 * The refresh strategy is per connection, and the decision is made ONLY from
 * main-process state populated by portal discovery — the cloud agent registry
 * (dashboard URL → agent id, written solely from portal `/api/agents`
 * results). Token-set fields are never consulted: a remote gateway writes its
 * own token response, so trusting its `provider` / `user_id` would let any
 * gateway make the desktop exchange the user's portal session for a real
 * Hermes Cloud agent bearer and hand it to that gateway.
 *
 *   - registry-known URL → re-exchange the portal session for that agent
 *     (audience from the registry), including a lazy first exchange when a
 *     saved cloud connection has no stored bearer (after sign-out + sign-in);
 *   - anything else      → the gateway's own `/auth/native/refresh`.
 */

import { readStatusCode } from './api-transport'
import { isCloudAgentAccessLost, isCloudLoginRequired } from './cloud-auth-errors'
import type { NativeAccessTokenCoordinatorDeps } from './native-access-token'
import type { NativeTokenSet } from './native-oauth'

export interface DesktopNativeTokenDeps {
  loadTokens: (baseUrl: string) => NativeTokenSet | null
  storeTokens: (baseUrl: string, tokens: NativeTokenSet) => void
  clearTokens: (baseUrl: string) => void
  normalizeBaseUrl: (baseUrl: string) => string
  nowSeconds?: () => number
  tokenNeedsRefresh: (tokens: NativeTokenSet, nowSeconds: number) => boolean
  /** POST {gateway}/auth/native/refresh → parsed token set. */
  refreshGatewayTokens: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
  /** Registry lookup: the agent id portal discovery returned for this URL. */
  cloudAgentIdFor: (baseUrl: string) => null | string
  /** Portal §5 exchange for one agent. */
  exchangeForAgent: (agentId: string) => Promise<NativeTokenSet>
  hasLivePortalSession: () => boolean
}

/**
 * Refresh/exchange verdicts that mean "this connection is signed out" (clear
 * the stored set), as opposed to a transient failure that keeps it: a gateway
 * 401, a lost portal session, or lost access to the agent.
 */
export function isNativeRefreshAuthRejection(error: unknown): boolean {
  return readStatusCode(error) === 401 || isCloudLoginRequired(error) || isCloudAgentAccessLost(error)
}

export function createNativeTokenCoordinatorDeps(deps: DesktopNativeTokenDeps): NativeAccessTokenCoordinatorDeps {
  const cloudAgentId = (baseUrl: string) => deps.cloudAgentIdFor(baseUrl)

  return {
    loadTokens: deps.loadTokens,
    storeTokens: deps.storeTokens,
    clearTokens: deps.clearTokens,
    normalizeBaseUrl: deps.normalizeBaseUrl,
    nowSeconds: deps.nowSeconds,
    tokenNeedsRefresh: deps.tokenNeedsRefresh,
    isRefreshAuthRejection: isNativeRefreshAuthRejection,
    // A cloud agent bearer has no refresh token but renews by re-exchange.
    canRefresh: (tokens, baseUrl) => Boolean(cloudAgentId(baseUrl)) || Boolean(tokens.refreshToken),
    refreshTokens: (baseUrl, tokens) => {
      const agentId = cloudAgentId(baseUrl)

      return agentId ? deps.exchangeForAgent(agentId) : deps.refreshGatewayTokens(baseUrl, tokens)
    },
    bootstrapTokens: async baseUrl => {
      const agentId = cloudAgentId(baseUrl)

      return agentId && deps.hasLivePortalSession() ? deps.exchangeForAgent(agentId) : null
    }
  }
}
