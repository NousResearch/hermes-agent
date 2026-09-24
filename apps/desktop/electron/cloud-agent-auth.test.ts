/**
 * Cloud-agent bearer lifecycle: per-agent sign-in via the §5 exchange, the
 * agent-id registry that survives restarts, the per-connection refresher
 * strategy (cloud → re-exchange, never /auth/native/refresh), 401-after-valid,
 * and logout clearing every derived agent token.
 */

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import {
  canRefreshNativeTokenSet,
  createCloudAgentAuth,
  createCloudAgentRegistry,
  createNativeTokenRefresher,
  isCloudAuthLoss
} from './cloud-agent-auth'
import { createNativeAccessTokenCoordinator } from './native-access-token'
import { type NativeTokenSet, tokenNeedsRefresh } from './native-oauth'
import { mintGatewayWsTicket } from './oauth-rest-request'

const AGENT_URL = 'https://agent-1.cloud.example.test'

function memoryIo(initial = '') {
  let text = initial

  return {
    io: {
      readText: () => {
        if (!text) {
          throw new Error('ENOENT')
        }

        return text
      },
      writeText: (next: string) => {
        text = next
      }
    },
    text: () => text
  }
}

const agentTokens = (overrides: Partial<NativeTokenSet> = {}): NativeTokenSet => ({
  accessToken: 'AGENT-AT-1',
  refreshToken: '',
  expiresAt: 10_000,
  provider: 'hermes-cloud-agent',
  userId: 'agt_1',
  ...overrides
})

const gatewayTokens = (overrides: Partial<NativeTokenSet> = {}): NativeTokenSet => ({
  accessToken: 'GW-AT',
  refreshToken: 'GW-RT',
  expiresAt: 1,
  provider: 'nous',
  userId: 'u1',
  ...overrides
})

test('refresher: a cloud-agent token re-exchanges; it never calls the gateway /auth/native/refresh', async () => {
  const calls: string[] = []

  const refresh = createNativeTokenRefresher({
    refreshGatewayTokens: async baseUrl => {
      calls.push(`gateway:${baseUrl}`)

      return gatewayTokens({ accessToken: 'GW-AT-2' })
    },
    reexchangeCloudAgent: async (baseUrl, tokens) => {
      calls.push(`exchange:${baseUrl}:${tokens.userId}`)

      return agentTokens({ accessToken: 'AGENT-AT-2' })
    }
  })

  await expect(refresh(AGENT_URL, agentTokens())).resolves.toMatchObject({ accessToken: 'AGENT-AT-2' })
  await expect(refresh('https://gw.example.test', gatewayTokens())).resolves.toMatchObject({ accessToken: 'GW-AT-2' })
  expect(calls).toEqual([`exchange:${AGENT_URL}:agt_1`, 'gateway:https://gw.example.test'])
})

test('a cloud-agent token without a refresh token is still refreshable; a gateway one is not', () => {
  expect(canRefreshNativeTokenSet(agentTokens())).toBe(true)
  expect(canRefreshNativeTokenSet(gatewayTokens({ refreshToken: '' }))).toBe(false)
  expect(canRefreshNativeTokenSet(gatewayTokens())).toBe(true)
})

function makeCoordinator(opts: {
  reexchange: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
  now?: number
}) {
  const map = new Map<string, NativeTokenSet>()
  let gatewayRefreshes = 0

  const coordinator = createNativeAccessTokenCoordinator({
    canRefresh: canRefreshNativeTokenSet,
    clearTokens: url => void map.delete(url),
    isRefreshAuthRejection: error => isCloudAuthLoss(error),
    loadTokens: url => map.get(url) ?? null,
    normalizeBaseUrl: url => url.replace(/\/+$/, ''),
    nowSeconds: () => opts.now ?? 1_000,
    refreshTokens: createNativeTokenRefresher({
      refreshGatewayTokens: async () => {
        gatewayRefreshes++

        return gatewayTokens()
      },
      reexchangeCloudAgent: opts.reexchange
    }),
    storeTokens: (url, tokens) => void map.set(url, tokens),
    tokenNeedsRefresh
  })

  return { coordinator, map, gatewayRefreshes: () => gatewayRefreshes }
}

test('coordinator: an expiring cloud-agent token is re-exchanged (not cleared for lacking a refresh token)', async () => {
  let exchanges = 0

  const { coordinator, map, gatewayRefreshes } = makeCoordinator({
    reexchange: async () => {
      exchanges++

      return agentTokens({ accessToken: 'AGENT-AT-2', expiresAt: 5_000 })
    }
  })

  map.set(AGENT_URL, agentTokens({ expiresAt: 1_010 }))

  await expect(coordinator.ensure(AGENT_URL)).resolves.toBe('AGENT-AT-2')
  expect(exchanges).toBe(1)
  expect(gatewayRefreshes()).toBe(0)
  expect(map.get(AGENT_URL)?.accessToken).toBe('AGENT-AT-2')
})

test('coordinator: losing the portal session or agent access during re-exchange signs the connection out', async () => {
  for (const failure of [
    Object.assign(new Error('signed out'), { needsCloudLogin: true }),
    Object.assign(new Error('lost'), { cloudAgentAccessLost: true })
  ]) {
    const { coordinator, map } = makeCoordinator({
      reexchange: async () => {
        throw failure
      }
    })

    map.set(AGENT_URL, agentTokens({ expiresAt: 1 }))
    await expect(coordinator.ensure(AGENT_URL)).resolves.toBeNull()
    expect(map.has(AGENT_URL)).toBe(false)
  }
})

test('coordinator: a transient re-exchange failure keeps the agent token for the next attempt', async () => {
  const { coordinator, map } = makeCoordinator({
    reexchange: async () => {
      throw httpStatusError(503, 'down')
    }
  })

  map.set(AGENT_URL, agentTokens({ expiresAt: 1 }))
  await expect(coordinator.ensure(AGENT_URL)).rejects.toMatchObject({ statusCode: 503 })
  expect(map.has(AGENT_URL)).toBe(true)
})

test('ws-ticket: a 401 on a still-valid agent token earns exactly one forced re-exchange', async () => {
  let exchanges = 0

  const { coordinator, map } = makeCoordinator({
    reexchange: async () => {
      exchanges++

      return agentTokens({ accessToken: `AGENT-AT-${exchanges + 1}` })
    }
  })

  map.set(AGENT_URL, agentTokens())
  const bearers: string[] = []

  const deps = {
    ensureNativeAccessToken: coordinator.ensure,
    fetchJson: async (_url: string, _token: null | string, options: any) => {
      bearers.push(options.bearer)

      if (options.bearer === 'AGENT-AT-1') {
        throw httpStatusError(401, JSON.stringify({ detail: 'expired' }))
      }

      return { ticket: 'T-1' }
    },
    fetchJsonViaOauthSession: async () => {
      throw httpStatusError(401, 'no cookie')
    }
  }

  await expect(mintGatewayWsTicket(AGENT_URL, deps)).resolves.toBe('T-1')
  expect(bearers).toEqual(['AGENT-AT-1', 'AGENT-AT-2'])
  expect(exchanges).toBe(1)

  // A second 401 on the fresh token is a confirmed rejection: no further loop.
  map.set(AGENT_URL, agentTokens({ accessToken: 'AGENT-AT-1' }))
  bearers.length = 0
  exchanges = 0

  deps.fetchJson = async (_url, _token, options) => {
    bearers.push(options.bearer)
    throw httpStatusError(401, JSON.stringify({ detail: 'nope' }))
  }

  await expect(mintGatewayWsTicket(AGENT_URL, deps)).rejects.toMatchObject({ statusCode: 401 })
  expect(bearers).toEqual(['AGENT-AT-1', 'AGENT-AT-2'])
  expect(exchanges).toBe(1)
})

test('registry: agent ids persist per normalized dashboard URL across restarts', () => {
  const disk = memoryIo()
  const first = createCloudAgentRegistry(disk.io, url => url.replace(/\/+$/, ''))

  first.remember(`${AGENT_URL}/`, 'agt_1')
  first.remember('https://agent-2.cloud.example.test', 'agt_2')

  const restarted = createCloudAgentRegistry(disk.io, url => url.replace(/\/+$/, ''))
  expect(restarted.agentIdFor(AGENT_URL)).toBe('agt_1')
  expect(restarted.urls().sort()).toEqual([AGENT_URL, 'https://agent-2.cloud.example.test'])
  expect(createCloudAgentRegistry(memoryIo('not json').io, u => u).agentIdFor(AGENT_URL)).toBeNull()
})

function makeAuth(
  opts: { discover?: () => Promise<any[]>; exchange?: (agentId: string) => Promise<NativeTokenSet> } = {}
) {
  const disk = memoryIo()
  const registry = createCloudAgentRegistry(disk.io, url => url.replace(/\/+$/, ''))
  const stored = new Map<string, NativeTokenSet>()
  const cleared: string[] = []
  const exchanged: string[] = []
  let portalCleared = false

  const auth = createCloudAgentAuth({
    normalizeBaseUrl: url => url.replace(/\/+$/, ''),
    registry,
    exchangeForAgent:
      opts.exchange ??
      (async agentId => {
        exchanged.push(agentId)

        return agentTokens({ userId: agentId })
      }),
    storeAgentTokens: (url, tokens) => void stored.set(url, tokens),
    clearAgentTokens: url => {
      cleared.push(url)
      stored.delete(url)
    },
    clearPortalSession: () => {
      portalCleared = true
    },
    discoverAgents: opts.discover
  })

  return { auth, registry, stored, cleared, exchanged, portalCleared: () => portalCleared, disk }
}

test('agent sign-in resolves the agent id from the last discovery and stores the exchanged bearer', async () => {
  const { auth, stored, exchanged, registry } = makeAuth()

  auth.rememberDiscovered([
    { id: 'agt_1', dashboardUrl: `${AGENT_URL}/` },
    { id: 'agt_x', dashboardUrl: null }
  ])

  await expect(auth.signIn(AGENT_URL)).resolves.toEqual({ baseUrl: AGENT_URL, connected: true })
  expect(exchanged).toEqual(['agt_1'])
  expect(stored.get(AGENT_URL)).toMatchObject({
    accessToken: 'AGENT-AT-1',
    provider: 'hermes-cloud-agent',
    userId: 'agt_1'
  })
  expect(registry.agentIdFor(AGENT_URL)).toBe('agt_1')
})

test('agent sign-in prefers an explicit agent id and persists it for reconnect after restart', async () => {
  const { auth, exchanged, disk } = makeAuth()

  await auth.signIn(AGENT_URL, 'agt_explicit')
  expect(exchanged).toEqual(['agt_explicit'])
  expect(createCloudAgentRegistry(disk.io, u => u).agentIdFor(AGENT_URL)).toBe('agt_explicit')
})

test('agent sign-in for an unknown URL falls back to one live discovery, then fails clearly', async () => {
  let discoveries = 0

  const { auth, exchanged } = makeAuth({
    discover: async () => {
      discoveries++

      return [{ id: 'agt_7', dashboardUrl: AGENT_URL }]
    }
  })

  await auth.signIn(AGENT_URL)
  expect(exchanged).toEqual(['agt_7'])
  expect(discoveries).toBe(1)

  await expect(auth.signIn('https://unknown.cloud.example.test')).rejects.toThrow(/could not find this agent/i)
})

test('re-exchange uses the agent id carried by the stored token set', async () => {
  const { auth, exchanged } = makeAuth()

  await expect(auth.reexchange(AGENT_URL, agentTokens({ userId: 'agt_9' }))).resolves.toMatchObject({ userId: 'agt_9' })
  expect(exchanged).toEqual(['agt_9'])
})

test('logout clears the portal session and every derived agent token', async () => {
  const { auth, cleared, stored, portalCleared } = makeAuth()

  await auth.signIn(AGENT_URL, 'agt_1')
  await auth.signIn('https://agent-2.cloud.example.test', 'agt_2')
  auth.logout()

  expect(portalCleared()).toBe(true)
  expect(cleared.sort()).toEqual([AGENT_URL, 'https://agent-2.cloud.example.test'])
  expect(stored.size).toBe(0)
})

test('isCloudAuthLoss recognises signed-out and access-lost errors only', () => {
  expect(isCloudAuthLoss(Object.assign(new Error('x'), { needsCloudLogin: true }))).toBe(true)
  expect(isCloudAuthLoss(Object.assign(new Error('x'), { cloudAgentAccessLost: true }))).toBe(true)
  expect(isCloudAuthLoss(httpStatusError(401, 'x'))).toBe(false)
  expect(isCloudAuthLoss(new Error('network'))).toBe(false)
})
