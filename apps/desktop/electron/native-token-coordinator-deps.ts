/**
 * native-token-coordinator-deps.ts
 *
 * The one wiring of the native access-token coordinator that main.ts uses,
 * kept out of main.ts so the real policy is unit-tested rather than
 * re-implemented in tests.
 *
 * The refresh strategy is per connection, and the decision is made ONLY from
 * main-process state populated by portal discovery. Token-set fields are
 * never consulted: a remote gateway writes its own token response, so
 * trusting its `provider` / `user_id` would let any gateway make the desktop
 * exchange the user's portal session for a real Hermes Cloud agent bearer and
 * hand it to that gateway.
 *
 *   - ROUTING: a URL with a cloud agent registry row renews by re-exchanging
 *     the portal session (never `/auth/native/refresh`); anything else uses
 *     the gateway's own `/auth/native/refresh`;
 *   - AUDIENCE: every exchange — expired/forced re-exchange and the lazy
 *     first exchange for a saved connection with nothing stored — takes the
 *     agent id from `confirmedCloudAgentId`, i.e. a binding portal discovery
 *     confirmed within the last few minutes (or a live discovery run now).
 *     A persisted registry row alone never authorizes an exchange.
 *
 * Transient failures (a §5 rate limit, or discovery being unreachable) keep
 * the stored set, and a still-valid bearer keeps being served until it
 * actually expires. No new bearer is minted from an unconfirmed binding.
 */

import { readStatusCode } from './api-transport'
import {
  cloudAgentAccessLostError,
  cloudDiscoveryUnavailableError,
  isCloudAgentAccessLost,
  isCloudDiscoveryUnavailable,
  isCloudLoginRequired,
  isCloudRateLimited
} from './cloud-auth-errors'
import { type NativeAccessTokenCoordinatorDeps, NativeAuthChangedError } from './native-access-token'
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
  /** Routing hint: portal discovery recorded this URL as a cloud agent dashboard. */
  isCloudAgentUrl: (baseUrl: string) => boolean
  /**
   * The exchange audience: the agent id a RECENT portal discovery confirmed
   * for this URL (running one if needed), or null. Rejects with a transient
   * cloudDiscoveryUnavailable error when the portal cannot be reached.
   */
  confirmedCloudAgentId: (baseUrl: string) => Promise<null | string>
  /**
   * Whether the latest portal snapshot still binds this URL to this agent.
   * Checked after every exchange: a discovery landing mid-flight can re-bind
   * or drop the URL, and the bearer must then not be stored for it.
   */
  isCloudBindingCurrent: (baseUrl: string, agentId: string) => boolean
  /** Portal §5 exchange for one agent. */
  exchangeForAgent: (agentId: string) => Promise<NativeTokenSet>
  hasLivePortalSession: () => boolean
  /** Whether a SAVED connection (config or registry) names this URL with mode 'cloud'. */
  isSavedCloudConnection?: (baseUrl: string) => boolean
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
  const nowSeconds = () => deps.nowSeconds?.() ?? Math.floor(Date.now() / 1_000)

  async function exchangeForCurrentBinding(baseUrl: string, agentId: string): Promise<NativeTokenSet> {
    const tokens = await deps.exchangeForAgent(agentId)

    if (!deps.isCloudBindingCurrent(baseUrl, agentId)) {
      // Re-bound or dropped mid-flight: discard; the retry uses the fresh binding.
      throw new NativeAuthChangedError()
    }

    return tokens
  }

  async function reExchange(baseUrl: string, tokens: NativeTokenSet, forced: boolean): Promise<NativeTokenSet> {
    try {
      const agentId = await deps.confirmedCloudAgentId(baseUrl)

      if (!agentId) {
        // Still routed as cloud = the throttle blocked a confirmation
        // (transient). Otherwise the portal no longer lists this URL.
        throw deps.isCloudAgentUrl(baseUrl) ? cloudDiscoveryUnavailableError() : cloudAgentAccessLostError()
      }

      // The confirming discovery re-bound this URL and cleared its bearer
      // (epoch-fenced, so this flight could not store anyway): don't spend
      // a rate-limited exchange; the retry bootstraps from the fresh binding.
      if (deps.loadTokens(baseUrl)?.accessToken !== tokens.accessToken) {
        throw new NativeAuthChangedError()
      }

      return await exchangeForCurrentBinding(baseUrl, agentId)
    } catch (error) {
      // Transient (rate limited, or the binding could not be confirmed):
      // keep serving the current bearer while it is still valid (unless it
      // was just rejected). Otherwise the error propagates as a transient
      // failure and the stored set is kept.
      if (
        (isCloudRateLimited(error) || isCloudDiscoveryUnavailable(error)) &&
        !forced &&
        tokens.expiresAt > nowSeconds()
      ) {
        return tokens
      }

      throw error
    }
  }

  return {
    loadTokens: deps.loadTokens,
    storeTokens: deps.storeTokens,
    clearTokens: deps.clearTokens,
    normalizeBaseUrl: deps.normalizeBaseUrl,
    nowSeconds: deps.nowSeconds,
    tokenNeedsRefresh: deps.tokenNeedsRefresh,
    isRefreshAuthRejection: isNativeRefreshAuthRejection,
    // A cloud agent bearer has no refresh token but renews by re-exchange.
    canRefresh: (tokens, baseUrl) => deps.isCloudAgentUrl(baseUrl) || Boolean(tokens.refreshToken),
    refreshTokens: (baseUrl, tokens, context) =>
      deps.isCloudAgentUrl(baseUrl)
        ? reExchange(baseUrl, tokens, Boolean(context?.forced))
        : deps.refreshGatewayTokens(baseUrl, tokens),
    bootstrapTokens: async baseUrl => {
      if (!deps.hasLivePortalSession()) {
        return null
      }

      if (!deps.isCloudAgentUrl(baseUrl) && !deps.isSavedCloudConnection?.(baseUrl)) {
        return null
      }

      const agentId = await deps.confirmedCloudAgentId(baseUrl)

      return agentId ? exchangeForCurrentBinding(baseUrl, agentId) : null
    }
  }
}
