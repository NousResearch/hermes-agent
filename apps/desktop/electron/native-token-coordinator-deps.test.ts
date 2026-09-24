/**
 * The production coordinator wiring (createNativeTokenCoordinatorDeps) on its
 * own: the refresh strategy is decided by the portal-discovery registry only,
 * a §5 rate limit is transient, and a saved cloud connection the registry
 * lost earns one throttled rediscovery.
 */

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { cloudExchangeRateLimitedError } from './cloud-auth-errors'
import { normalizeRemoteBaseUrl } from './connection-config'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import { type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'
import { createNativeTokenCoordinatorDeps, type DesktopNativeTokenDeps } from './native-token-coordinator-deps'
import { mintGatewayWsTicket } from './oauth-rest-request'

const AGENT_URL = 'https://agent-1.cloud.example.test'
const OTHER_URL = 'https://gw.example.test'

const agentSet = (overrides: Partial<NativeTokenSet> = {}): NativeTokenSet => ({
  accessToken: 'AGENT-AT-1',
  refreshToken: '',
  expiresAt: 10_000,
  provider: 'hermes-cloud-agent',
  userId: 'agt_1',
  ...overrides
})

function wiring(overrides: Partial<DesktopNativeTokenDeps> & { registry?: Record<string, string> } = {}) {
  const store = new Map<string, NativeTokenSet>()
  const registry = new Map(Object.entries(overrides.registry ?? {}))
  const calls = { exchanged: [] as string[], gatewayRefreshes: [] as string[], cleared: [] as string[] }

  const deps = createNativeTokenCoordinatorDeps({
    loadTokens: url => store.get(url) ?? null,
    storeTokens: (url, tokens) => void store.set(url, tokens),
    clearTokens: url => {
      calls.cleared.push(url)
      store.delete(url)
    },
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    nowSeconds: () => 1_000,
    tokenNeedsRefresh,
    refreshGatewayTokens: async url => {
      calls.gatewayRefreshes.push(url)

      throw httpStatusError(401, 'rejected')
    },
    cloudAgentIdFor: url => registry.get(url) ?? null,
    exchangeForAgent: async agentId => {
      calls.exchanged.push(agentId)

      return agentSet({ accessToken: `EXCHANGED-${agentId}`, userId: agentId, expiresAt: 5_000 })
    },
    hasLivePortalSession: () => true,
    ...overrides
  })

  return { deps, coordinator: createNativeAccessTokenCoordinator(deps), store, registry, calls }
}

// --- T1 ---

test('T1: an unregistered set claiming provider hermes-cloud-agent with no refresh token is cleared, never refreshed or exchanged', async () => {
  const { deps, coordinator, store, calls } = wiring({ registry: { [AGENT_URL]: 'agt_1' } })
  const claimed = agentSet({ userId: 'agt_1', expiresAt: 1 })

  // canRefresh looks at the registry only — not at provider/userId/refreshToken shape.
  expect(deps.canRefresh!(claimed, OTHER_URL)).toBe(false)
  expect(deps.canRefresh!(agentSet({ provider: 'anything', userId: '' }), AGENT_URL)).toBe(true)

  store.set(OTHER_URL, claimed)
  await expect(coordinator.ensure(OTHER_URL)).resolves.toBeNull()
  expect(store.has(OTHER_URL)).toBe(false)
  expect(calls.cleared).toEqual([OTHER_URL])
  expect(calls.gatewayRefreshes).toEqual([])
  expect(calls.exchanged).toEqual([])
})

// --- N1: §5 rate limit on the coordinator path ---

function rateLimitedWiring(opts: { expiresAt: number }) {
  let attempts = 0

  const w = wiring({
    registry: { [AGENT_URL]: 'agt_1' },
    exchangeForAgent: async () => {
      attempts++

      throw cloudExchangeRateLimitedError(30)
    }
  })

  w.store.set(AGENT_URL, agentSet({ expiresAt: opts.expiresAt }))

  return { ...w, attempts: () => attempts }
}

test('N1: while rate limited, a near-expiry but still-valid agent bearer keeps being served and is not cleared', async () => {
  // Inside the refresh skew (needs refresh) but not expired yet.
  const w = rateLimitedWiring({ expiresAt: 1_020 })

  await expect(w.coordinator.ensure(AGENT_URL)).resolves.toBe('AGENT-AT-1')
  await expect(w.coordinator.ensure(AGENT_URL)).resolves.toBe('AGENT-AT-1')
  expect(w.store.get(AGENT_URL)?.accessToken).toBe('AGENT-AT-1')
  expect(w.calls.cleared).toEqual([])
})

test('N1: rate limited with an expired bearer fails transiently (429) and keeps the stored set', async () => {
  const w = rateLimitedWiring({ expiresAt: 999 })

  await expect(w.coordinator.ensure(AGENT_URL)).rejects.toMatchObject({ statusCode: 429 })
  expect(w.store.has(AGENT_URL)).toBe(true)
  expect(w.calls.cleared).toEqual([])
})

test('N1: a forced re-exchange after a 401 is not answered with the rejected token; the 429 surfaces, nothing is cleared', async () => {
  const w = rateLimitedWiring({ expiresAt: 5_000 })

  await expect(
    w.coordinator.ensure(AGENT_URL, { forceRefresh: true, rejectedAccessToken: 'AGENT-AT-1' })
  ).rejects.toMatchObject({ statusCode: 429 })
  expect(w.store.has(AGENT_URL)).toBe(true)
})

test('N1: ws-ticket minting surfaces the 429 as a transient failure, never a sign-in verdict', async () => {
  const w = rateLimitedWiring({ expiresAt: 5_000 })

  const error = await mintGatewayWsTicket(AGENT_URL, {
    ensureNativeAccessToken: w.coordinator.ensure,
    fetchJson: async () => {
      throw httpStatusError(401, JSON.stringify({ detail: 'expired' }))
    },
    fetchJsonViaOauthSession: async () => {
      throw httpStatusError(401, 'no cookie')
    }
  }).catch(e => e)

  expect(error).toMatchObject({ statusCode: 429 })
  expect(error.needsOauthLogin).toBeUndefined()
  expect(w.store.has(AGENT_URL)).toBe(true)
})

// --- N3: saved cloud connection the registry lost ---

test("N3: a saved cloud connection with no registry entry rediscovers once and exchanges for the portal's answer", async () => {
  const rediscovered: string[] = []

  const w = wiring({
    isSavedCloudConnection: url => url === AGENT_URL,
    rediscoverCloudAgentId: async url => {
      rediscovered.push(url)

      return 'agt_portal'
    }
  })

  await expect(w.coordinator.ensure(AGENT_URL)).resolves.toBe('EXCHANGED-agt_portal')
  expect(rediscovered).toEqual([AGENT_URL])
  expect(w.calls.exchanged).toEqual(['agt_portal'])
})

test('N3: no rediscovery for an unsaved URL, without a live portal session, or when the portal does not list it', async () => {
  let rediscoveries = 0

  const rediscoverCloudAgentId = async () => {
    rediscoveries++

    return null
  }

  const unsaved = wiring({ isSavedCloudConnection: () => false, rediscoverCloudAgentId })
  await expect(unsaved.coordinator.ensure(AGENT_URL)).resolves.toBeNull()

  const signedOut = wiring({
    isSavedCloudConnection: () => true,
    rediscoverCloudAgentId,
    hasLivePortalSession: () => false
  })

  await expect(signedOut.coordinator.ensure(AGENT_URL)).resolves.toBeNull()
  expect(rediscoveries).toBe(0)

  const unlisted = wiring({ isSavedCloudConnection: () => true, rediscoverCloudAgentId })
  await expect(unlisted.coordinator.ensure(AGENT_URL)).resolves.toBeNull()
  expect(rediscoveries).toBe(1)
  expect(unlisted.calls.exchanged).toEqual([])
})
