/**
 * Cloud-agent bearer lifecycle: per-agent sign-in via the §5 exchange, the
 * portal-discovery-only agent registry (reconciled as a snapshot; a row is a
 * routing hint, and only a recently confirmed binding is an exchange
 * audience), the real coordinator
 * wiring main.ts uses (createNativeTokenCoordinatorDeps), lazy re-exchange
 * for saved connections, and logout / org-switch cleanup.
 */

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import {
  CLOUD_AGENT_NOT_FOUND_MESSAGE,
  cloudDashboardUrlAllowed,
  createCloudAgentAuth,
  createCloudAgentRegistry
} from './cloud-agent-auth'
import { isCloudDiscoveryUnavailable } from './cloud-auth-errors'
import { normalizeRemoteBaseUrl } from './connection-config'
import { createNativeAccessTokenCoordinator, NativeAuthChangedError } from './native-access-token'
import { type NativeTokenSet, parseTokenResponse, tokenNeedsRefresh } from './native-oauth'
import { createNativeTokenCoordinatorDeps, isNativeRefreshAuthRejection } from './native-token-coordinator-deps'
import { mintGatewayWsTicket } from './oauth-rest-request'

const AGENT_URL = 'https://agent-1.cloud.example.test'
const EVIL_URL = 'https://evil.example'

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

/**
 * The production coordinator wiring (the same factory main.ts calls), over
 * an in-memory token store and a registry fed only by a portal discovery
 * that just returned `discovered`.
 */
function makeWiring(
  opts: {
    now?: number
    portalLive?: boolean
    discovered?: Record<string, string>
    exchange?: (agentId: string) => Promise<NativeTokenSet>
    refreshGateway?: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
  } = {}
) {
  const store = new Map<string, NativeTokenSet>()
  const registry = createCloudAgentRegistry(memoryIo().io, normalizeRemoteBaseUrl)
  const exchanged: string[] = []
  const gatewayRefreshes: string[] = []
  const rows = Object.entries(opts.discovered ?? {}).map(([dashboardUrl, id]) => ({ id, dashboardUrl }))

  const exchangeForAgent = async (agentId: string) => {
    exchanged.push(agentId)

    return opts.exchange
      ? opts.exchange(agentId)
      : agentTokens({ accessToken: `REAL-AGENT-TOKEN-${agentId}-${exchanged.length}`, userId: agentId })
  }

  const auth = createCloudAgentAuth({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    registry,
    exchangeForAgent,
    storeAgentTokens: (url, tokens) => void store.set(url, tokens),
    clearAgentTokens: url => void store.delete(url),
    listStoredTokenUrls: () => [...store.keys()],
    loadStoredTokens: url => store.get(url) ?? null,
    clearPortalSession: () => undefined,
    discoverAgents: async () => rows,
    nowMs: () => (opts.now ?? 1_000) * 1_000
  })

  // The portal just listed these: fresh, confirmed bindings.
  auth.reconcileDiscovered(rows)

  const coordinator = createNativeAccessTokenCoordinator(
    createNativeTokenCoordinatorDeps({
      loadTokens: url => store.get(url) ?? null,
      storeTokens: (url, tokens) => void store.set(url, tokens),
      clearTokens: url => void store.delete(url),
      normalizeBaseUrl: normalizeRemoteBaseUrl,
      nowSeconds: () => opts.now ?? 1_000,
      tokenNeedsRefresh,
      refreshGatewayTokens: async (baseUrl, tokens) => {
        gatewayRefreshes.push(baseUrl)

        if (opts.refreshGateway) {
          return opts.refreshGateway(baseUrl, tokens)
        }

        throw httpStatusError(401, 'gateway refresh rejected')
      },
      isCloudAgentUrl: url => registry.agentIdFor(url) !== null,
      confirmedCloudAgentId: url => auth.confirmedAgentIdFor(url),
      isCloudBindingCurrent: (url, id) => auth.isBindingCurrent(url, id),
      exchangeForAgent,
      hasLivePortalSession: () => opts.portalLive ?? true
    })
  )

  return { coordinator, store, registry, exchanged, gatewayRefreshes }
}

// --- B1: a remote gateway can never make the desktop mint a cloud bearer ---

test('B1 regression (reviewer PoC): a gateway claiming provider=hermes-cloud-agent gets no exchange and no bearer', async () => {
  const { coordinator, store, exchanged } = makeWiring({ discovered: { [AGENT_URL]: 'victim-agent' } })

  // A malicious third-party gateway's /auth/native/token response.
  store.set(
    EVIL_URL,
    parseTokenResponse({ access_token: 'evil', expires_at: 0, provider: 'hermes-cloud-agent', user_id: 'victim-agent' })
  )

  const bearer = await coordinator.ensure(EVIL_URL)

  expect(exchanged).toEqual([])
  expect(bearer).toBeNull()
  expect(store.has(EVIL_URL)).toBe(false)
})

test('B1: even a stored set already tagged hermes-cloud-agent is not cloud unless portal discovery says so', async () => {
  const { coordinator, store, exchanged, gatewayRefreshes } = makeWiring({
    discovered: { [AGENT_URL]: 'victim-agent' },
    refreshGateway: async () => gatewayTokens({ accessToken: 'GW-AT-2', expiresAt: 9_000 })
  })

  // Written by an older build (before reserved providers were stripped).
  store.set(EVIL_URL, agentTokens({ userId: 'victim-agent', refreshToken: 'evil-rt', expiresAt: 1 }))

  await expect(coordinator.ensure(EVIL_URL)).resolves.toBe('GW-AT-2')
  expect(exchanged).toEqual([])
  expect(gatewayRefreshes).toEqual([EVIL_URL])
  // No bearer for a victim agent is ever handed to the gateway's own URL.
  await expect(coordinator.ensure(EVIL_URL, { forceRefresh: true })).resolves.not.toMatch(/REAL-AGENT-TOKEN/)
})

test('B1: the exchange audience comes from the registry; a disagreeing token-set userId is ignored', async () => {
  const { coordinator, store, exchanged } = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })

  store.set(AGENT_URL, agentTokens({ userId: 'victim-agent', expiresAt: 1 }))

  await expect(coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-1')
  expect(exchanged).toEqual(['agt_1'])
})

test('a registry-known cloud URL re-exchanges on expiry (never /auth/native/refresh), even without a refresh token', async () => {
  const { coordinator, store, exchanged, gatewayRefreshes } = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })

  store.set(AGENT_URL, agentTokens({ expiresAt: 1_010 }))

  await expect(coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-1')
  expect(exchanged).toEqual(['agt_1'])
  expect(gatewayRefreshes).toEqual([])
  expect(store.get(AGENT_URL)?.accessToken).toBe('REAL-AGENT-TOKEN-agt_1-1')
})

test('a gateway token set rotates through the gateway refresher; with no refresh token it is signed out', async () => {
  const { coordinator, store, gatewayRefreshes } = makeWiring({
    refreshGateway: async () => gatewayTokens({ accessToken: 'GW-AT-2', expiresAt: 9_000 })
  })

  store.set('https://gw.example.test', gatewayTokens())
  await expect(coordinator.ensure('https://gw.example.test')).resolves.toBe('GW-AT-2')
  expect(gatewayRefreshes).toEqual(['https://gw.example.test'])

  store.set('https://gw2.example.test', gatewayTokens({ refreshToken: '' }))
  await expect(coordinator.ensure('https://gw2.example.test')).resolves.toBeNull()
  expect(store.has('https://gw2.example.test')).toBe(false)
})

// --- real isRefreshAuthRejection wiring ---

test('isNativeRefreshAuthRejection: 401, signed-out and access-lost clear; transient failures do not', () => {
  expect(isNativeRefreshAuthRejection(httpStatusError(401, 'x'))).toBe(true)
  expect(isNativeRefreshAuthRejection(Object.assign(new Error('x'), { needsCloudLogin: true }))).toBe(true)
  expect(isNativeRefreshAuthRejection(Object.assign(new Error('x'), { cloudAgentAccessLost: true }))).toBe(true)
  expect(isNativeRefreshAuthRejection(httpStatusError(503, 'x'))).toBe(false)
  expect(isNativeRefreshAuthRejection(new Error('network'))).toBe(false)
})

test('losing the portal session or agent access during re-exchange signs the connection out', async () => {
  for (const failure of [
    Object.assign(new Error('signed out'), { needsCloudLogin: true }),
    Object.assign(new Error('lost'), { cloudAgentAccessLost: true })
  ]) {
    const { coordinator, store } = makeWiring({
      discovered: { [AGENT_URL]: 'agt_1' },
      exchange: async () => {
        throw failure
      }
    })

    store.set(AGENT_URL, agentTokens({ expiresAt: 1 }))
    await expect(coordinator.ensure(AGENT_URL)).resolves.toBeNull()
    expect(store.has(AGENT_URL)).toBe(false)
  }
})

test('a transient re-exchange failure keeps the agent token for the next attempt', async () => {
  const { coordinator, store } = makeWiring({
    discovered: { [AGENT_URL]: 'agt_1' },
    exchange: async () => {
      throw httpStatusError(503, 'down')
    }
  })

  store.set(AGENT_URL, agentTokens({ expiresAt: 1 }))
  await expect(coordinator.ensure(AGENT_URL)).rejects.toMatchObject({ statusCode: 503 })
  expect(store.has(AGENT_URL)).toBe(true)
})

// --- M2: saved cloud connections recover after sign-out + sign-in ---

test('M2: a registry-known cloud URL with no stored bearer lazily exchanges while the portal session is live', async () => {
  const { coordinator, store, exchanged } = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })

  await expect(coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-1')
  expect(exchanged).toEqual(['agt_1'])
  expect(store.get(AGENT_URL)?.accessToken).toBe('REAL-AGENT-TOKEN-agt_1-1')
  // Stored: the next call is served without another exchange.
  await expect(coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-1')
  expect(exchanged).toEqual(['agt_1'])
})

test('M2: no lazy exchange without a live portal session, or for a URL portal discovery never returned', async () => {
  const signedOut = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' }, portalLive: false })
  await expect(signedOut.coordinator.ensure(AGENT_URL)).resolves.toBeNull()
  expect(signedOut.exchanged).toEqual([])

  const unknown = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })
  await expect(unknown.coordinator.ensure(EVIL_URL)).resolves.toBeNull()
  expect(unknown.exchanged).toEqual([])
})

test('M2: concurrent callers share one lazy exchange; a lost-access verdict is "not signed in", not a crash', async () => {
  const ok = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })
  await expect(Promise.all([ok.coordinator.ensure(AGENT_URL), ok.coordinator.ensure(AGENT_URL)])).resolves.toEqual([
    'REAL-AGENT-TOKEN-agt_1-1',
    'REAL-AGENT-TOKEN-agt_1-1'
  ])
  expect(ok.exchanged).toEqual(['agt_1'])

  const lost = makeWiring({
    discovered: { [AGENT_URL]: 'agt_1' },
    exchange: async () => {
      throw Object.assign(new Error('lost'), { cloudAgentAccessLost: true })
    }
  })

  await expect(lost.coordinator.ensure(AGENT_URL)).resolves.toBeNull()
  expect(lost.store.has(AGENT_URL)).toBe(false)
})

test('ws-ticket: a 401 on a still-valid agent token earns exactly one forced re-exchange', async () => {
  const { coordinator, store, exchanged } = makeWiring({ discovered: { [AGENT_URL]: 'agt_1' } })

  store.set(AGENT_URL, agentTokens())
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
  expect(bearers).toEqual(['AGENT-AT-1', 'REAL-AGENT-TOKEN-agt_1-1'])
  expect(exchanged).toEqual(['agt_1'])
})

// --- registry ---

test('registry: bindings persist per normalized dashboard URL across restarts; replaceAll/forget/clear persist too', () => {
  const disk = memoryIo()
  const first = createCloudAgentRegistry(disk.io, url => url.replace(/\/+$/, ''))

  expect(
    first.replaceAll({
      [AGENT_URL]: { agentId: 'agt_1', confirmedAt: 5 },
      'https://agent-2.cloud.example.test': { agentId: 'agt_2', confirmedAt: 5 }
    })
  ).toEqual([])

  const restarted = createCloudAgentRegistry(disk.io, url => url.replace(/\/+$/, ''))
  expect(restarted.agentIdFor(`${AGENT_URL}/`)).toBe('agt_1')
  expect(restarted.bindingFor(AGENT_URL)).toEqual({ agentId: 'agt_1', confirmedAt: 5 })
  expect(restarted.urls().sort()).toEqual([AGENT_URL, 'https://agent-2.cloud.example.test'])
  expect(JSON.parse(disk.text())).toMatchObject({ version: 2 })
  expect(createCloudAgentRegistry(memoryIo('not json').io, u => u).agentIdFor(AGENT_URL)).toBeNull()

  // A snapshot reports what it dropped or re-bound.
  expect(
    restarted.replaceAll({
      [AGENT_URL]: { agentId: 'agt_1b', confirmedAt: 6 },
      'https://agent-3.cloud.example.test': { agentId: 'agt_3', confirmedAt: 6 }
    })
  ).toEqual([AGENT_URL, 'https://agent-2.cloud.example.test'])

  restarted.forget(AGENT_URL)
  expect(createCloudAgentRegistry(disk.io, u => u).urls()).toEqual(['https://agent-3.cloud.example.test'])
  restarted.clear()
  expect(createCloudAgentRegistry(disk.io, u => u).urls()).toEqual([])
})

test('registry: earlier on-disk formats read as unconfirmed rows', () => {
  for (const legacy of [
    { [AGENT_URL]: 'agt_old' },
    { orgId: 'org_a', agents: { [AGENT_URL]: 'agt_old' } },
    // Unversioned object rows are not trusted as confirmed either.
    { orgId: 'org_a', agents: { [AGENT_URL]: { agentId: 'agt_old', confirmedAt: 123 } } }
  ]) {
    const registry = createCloudAgentRegistry(memoryIo(JSON.stringify(legacy)).io, u => u)
    expect(registry.bindingFor(AGENT_URL)).toEqual({ agentId: 'agt_old', confirmedAt: 0 })
  }
})

test('dashboard URL policy: https only, plus plain http on loopback for the local portal stand-in', () => {
  expect(cloudDashboardUrlAllowed('https://agent.example.test')).toBe(true)
  expect(cloudDashboardUrlAllowed('http://127.0.0.1:9119')).toBe(true)
  expect(cloudDashboardUrlAllowed('http://localhost:9119')).toBe(true)
  expect(cloudDashboardUrlAllowed('http://agent.example.test')).toBe(false)
  expect(cloudDashboardUrlAllowed('file:///etc/passwd')).toBe(false)
  expect(cloudDashboardUrlAllowed('not a url')).toBe(false)
  expect(cloudDashboardUrlAllowed('')).toBe(false)
})

// --- agent auth ---

function makeAuth(
  opts: {
    discover?: () => Promise<any[]>
    exchange?: (agentId: string) => Promise<NativeTokenSet>
    stored?: Record<string, NativeTokenSet>
    disk?: ReturnType<typeof memoryIo>
    nowMs?: () => number
  } = {}
) {
  const disk = opts.disk ?? memoryIo()
  const registry = createCloudAgentRegistry(disk.io, normalizeRemoteBaseUrl)
  const stored = new Map<string, NativeTokenSet>(Object.entries(opts.stored ?? {}))
  const cleared: string[] = []
  const exchanged: string[] = []
  const logs: string[] = []
  let portalCleared = false

  const auth = createCloudAgentAuth({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
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
    listStoredTokenUrls: () => [...stored.keys()],
    loadStoredTokens: url => stored.get(url) ?? null,
    clearPortalSession: () => {
      portalCleared = true
    },
    discoverAgents: opts.discover,
    log: line => logs.push(line),
    nowMs: opts.nowMs
  })

  return { auth, registry, stored, cleared, exchanged, logs, portalCleared: () => portalCleared, disk }
}

test('agent sign-in resolves the agent id from the last discovery and stores the exchanged bearer', async () => {
  const { auth, stored, exchanged, registry } = makeAuth()

  auth.reconcileDiscovered([
    { id: 'agt_1', dashboardUrl: `${AGENT_URL}/` },
    { id: 'agt_x', dashboardUrl: null }
  ])

  await expect(auth.signIn(AGENT_URL)).resolves.toEqual({ baseUrl: AGENT_URL, connected: true })
  expect(exchanged).toEqual(['agt_1'])
  expect(stored.get(AGENT_URL)).toMatchObject({ accessToken: 'AGENT-AT-1', userId: 'agt_1' })
  expect(registry.agentIdFor(AGENT_URL)).toBe('agt_1')
})

test('M1: a renderer agent-id hint that matches the registry is used without re-discovery', async () => {
  let discoveries = 0

  const { auth, exchanged } = makeAuth({
    discover: async () => {
      discoveries++

      return []
    }
  })

  auth.reconcileDiscovered([{ id: 'agt_1', dashboardUrl: AGENT_URL }])
  await auth.signIn(AGENT_URL, 'agt_1')
  expect(exchanged).toEqual(['agt_1'])
  expect(discoveries).toBe(0)
})

test('M1: a hint that disagrees with the registry is never trusted: re-discover and use the portal answer', async () => {
  const { auth, exchanged, registry } = makeAuth({
    discover: async () => [{ id: 'agt_1', dashboardUrl: AGENT_URL }]
  })

  await auth.signIn(AGENT_URL, 'victim-agent')
  expect(exchanged).toEqual(['agt_1'])
  expect(registry.agentIdFor(AGENT_URL)).toBe('agt_1')
})

test('M1: a URL that is not a discovered agent dashboard is refused — no exchange, even with a hint', async () => {
  let discoveries = 0

  const { auth, exchanged, stored } = makeAuth({
    discover: async () => {
      discoveries++

      return [{ id: 'agt_1', dashboardUrl: AGENT_URL }]
    }
  })

  await expect(auth.signIn(EVIL_URL, 'agt_1')).rejects.toThrow(/could not find this agent/i)
  await expect(auth.signIn(EVIL_URL)).rejects.toThrow(/could not find this agent/i)
  expect(discoveries).toBe(2)
  expect(exchanged).toEqual([])
  expect(stored.size).toBe(0)
})

test('M1: a non-https dashboard URL is refused before any discovery or exchange', async () => {
  let discoveries = 0

  const { auth, exchanged } = makeAuth({
    discover: async () => {
      discoveries++

      return [{ id: 'agt_1', dashboardUrl: 'http://agent.example.test' }]
    }
  })

  await expect(auth.signIn('http://agent.example.test', 'agt_1')).rejects.toThrow(/https/i)
  expect(discoveries).toBe(0)
  expect(exchanged).toEqual([])
})

test('m4: one malformed discovery row is skipped without failing the others (and logs no tokens)', () => {
  const { auth, registry, logs } = makeAuth()

  expect(() =>
    auth.reconcileDiscovered([
      { id: 'bad', dashboardUrl: 'ftp://nope' },
      { id: 'bad2', dashboardUrl: 'http://plain-http.example.test' },
      { id: 'bad3', dashboardUrl: '::::' },
      { id: 'agt_1', dashboardUrl: AGENT_URL }
    ])
  ).not.toThrow()
  expect(registry.urls()).toEqual([AGENT_URL])
  expect(logs.length).toBe(3)
  expect(logs.join('\n')).not.toMatch(/token|bearer/i)
})

test('m3: logout clears the portal session and every stored cloud agent bearer, registry or not', async () => {
  const { auth, cleared, stored, portalCleared, registry } = makeAuth({
    stored: {
      // A cloud bearer whose best-effort registry write was lost.
      'https://orphan.cloud.example.test': agentTokens({ userId: 'agt_orphan' }),
      // A plain gateway login must survive a Cloud sign-out.
      'https://gw.example.test': gatewayTokens()
    }
  })

  auth.reconcileDiscovered([
    { id: 'agt_1', dashboardUrl: AGENT_URL },
    { id: 'agt_2', dashboardUrl: 'https://agent-2.cloud.example.test' }
  ])
  await auth.signIn(AGENT_URL, 'agt_1')
  await auth.signIn('https://agent-2.cloud.example.test', 'agt_2')
  auth.logout()

  expect(portalCleared()).toBe(true)
  expect([...new Set(cleared)].sort()).toEqual([
    AGENT_URL,
    'https://agent-2.cloud.example.test',
    'https://orphan.cloud.example.test'
  ])
  expect([...stored.keys()]).toEqual(['https://gw.example.test'])
  // The registry survives sign-out so saved connections re-exchange after
  // the next sign-in (M2).
  expect(registry.agentIdFor(AGENT_URL)).toBe('agt_1')
})

test('m1: forgetting the old org drops every cloud agent bearer and the registry, not the portal session', async () => {
  const { auth, stored, registry, portalCleared } = makeAuth({ stored: { 'https://gw.example.test': gatewayTokens() } })

  auth.reconcileDiscovered([{ id: 'agt_1', dashboardUrl: AGENT_URL }])
  await auth.signIn(AGENT_URL, 'agt_1')
  auth.forgetAgents()

  expect(registry.urls()).toEqual([])
  expect([...stored.keys()]).toEqual(['https://gw.example.test'])
  expect(portalCleared()).toBe(false)
})

// --- T2: stale registry vs. a mismatched hint ---

test("T2: a hint that disagrees with a STALE registry entry re-discovers; the portal's new id is exchanged", async () => {
  let discoveries = 0

  const { auth, exchanged, registry } = makeAuth({
    discover: async () => {
      discoveries++

      return [{ id: 'agt_new', dashboardUrl: AGENT_URL }]
    }
  })

  auth.reconcileDiscovered([{ id: 'agt_old', dashboardUrl: AGENT_URL }])
  await auth.signIn(AGENT_URL, 'agt_new')
  expect(discoveries).toBe(1)
  expect(exchanged).toEqual(['agt_new'])
  expect(registry.agentIdFor(AGENT_URL)).toBe('agt_new')
})

// --- N2: org change detected across a sign-out ---

test("N2: sign out of org A, sign in to org B → A's registry entries are dropped and never exchanged", async () => {
  const disk = memoryIo()
  const first = makeAuth({ disk })

  first.auth.adoptSessionOrg('org_a')
  first.auth.reconcileDiscovered([{ id: 'agt_a', dashboardUrl: AGENT_URL }])
  await first.auth.signIn(AGENT_URL, 'agt_a')
  first.auth.logout()
  expect(first.registry.agentIdFor(AGENT_URL)).toBe('agt_a')

  // Restart: the org is persisted with the registry (non-secret).
  const exchanged: string[] = []
  const registry = createCloudAgentRegistry(disk.io, normalizeRemoteBaseUrl)
  expect(registry.orgId()).toBe('org_a')

  const second = createCloudAgentAuth({
    normalizeBaseUrl: normalizeRemoteBaseUrl,
    registry,
    exchangeForAgent: async agentId => {
      exchanged.push(agentId)

      return agentTokens({ userId: agentId })
    },
    storeAgentTokens: () => undefined,
    clearAgentTokens: () => undefined,
    listStoredTokenUrls: () => [],
    loadStoredTokens: () => null,
    clearPortalSession: () => undefined
  })

  second.adoptSessionOrg('org_b')
  expect(registry.urls()).toEqual([])
  expect(registry.orgId()).toBe('org_b')
  await expect(second.signIn(AGENT_URL)).rejects.toThrow(/could not find this agent/i)
  expect(exchanged).toEqual([])
})

test('N2: signing in again to the SAME org keeps the registry; unknown-org entries are dropped for a known org', () => {
  const same = makeAuth()
  same.auth.adoptSessionOrg('org_a')
  same.auth.reconcileDiscovered([{ id: 'agt_a', dashboardUrl: AGENT_URL }])
  same.auth.adoptSessionOrg('org_a')
  expect(same.registry.agentIdFor(AGENT_URL)).toBe('agt_a')

  // A bare map written by an earlier build has no org: not trusted for org_b.
  const legacy = makeAuth({ disk: memoryIo(JSON.stringify({ [AGENT_URL]: 'agt_legacy' })) })
  expect(legacy.registry.agentIdFor(AGENT_URL)).toBe('agt_legacy')
  legacy.auth.adoptSessionOrg('org_b')
  expect(legacy.registry.urls()).toEqual([])
})

test("N2: an org change also clears the old org's stored agent bearers", async () => {
  const { auth, stored } = makeAuth({ stored: { 'https://gw.example.test': gatewayTokens() } })

  auth.adoptSessionOrg('org_a')
  auth.reconcileDiscovered([{ id: 'agt_a', dashboardUrl: AGENT_URL }])
  await auth.signIn(AGENT_URL, 'agt_a')
  auth.adoptSessionOrg('org_b')
  expect([...stored.keys()]).toEqual(['https://gw.example.test'])
})

test('N2: a discovery that started under org A and lands after a sign-in to org B is dropped', async () => {
  let release: (rows: any[]) => void = () => undefined

  const { auth, registry } = makeAuth({
    discover: () => new Promise(resolve => (release = resolve))
  })

  auth.adoptSessionOrg('org_a')
  const pending = auth.confirmedAgentIdFor(AGENT_URL)
  await Promise.resolve()

  auth.adoptSessionOrg('org_b')
  release([{ id: 'agt_a', dashboardUrl: AGENT_URL }])

  await expect(pending).resolves.toBeNull()
  expect(registry.urls()).toEqual([])
})

// --- N3: throttled background rediscovery ---

test('N3: confirmedAgentIdFor shares one discovery between concurrent callers and throttles repeats', async () => {
  let now = 0
  let discoveries = 0
  let listed: Array<{ id: string; dashboardUrl: string }> = []

  const { auth } = makeAuth({
    nowMs: () => now,
    discover: async () => {
      discoveries++

      return listed
    }
  })

  await expect(
    Promise.all([auth.confirmedAgentIdFor(AGENT_URL), auth.confirmedAgentIdFor(AGENT_URL)])
  ).resolves.toEqual([null, null])
  expect(discoveries).toBe(1)

  // Within the throttle window: no second portal round trip.
  listed = [{ id: 'agt_1', dashboardUrl: AGENT_URL }]
  now = 30_000
  await expect(auth.confirmedAgentIdFor(AGENT_URL)).resolves.toBeNull()
  expect(discoveries).toBe(1)

  now = 61_000
  await expect(auth.confirmedAgentIdFor(AGENT_URL)).resolves.toBe('agt_1')
  expect(discoveries).toBe(2)
  // Now known: served from the registry.
  await expect(auth.confirmedAgentIdFor(AGENT_URL)).resolves.toBe('agt_1')
  expect(discoveries).toBe(2)
})

test('N3: a failed discovery fails closed with a transient error (not sign-out); non-https URLs never trigger one', async () => {
  let discoveries = 0

  const { auth, logs, exchanged } = makeAuth({
    discover: async () => {
      discoveries++

      throw Object.assign(new Error('portal unreachable'), { statusCode: 503 })
    }
  })

  await expect(auth.confirmedAgentIdFor('http://plain.example.test')).resolves.toBeNull()
  expect(discoveries).toBe(0)

  const error = await auth.confirmedAgentIdFor(AGENT_URL).catch(e => e)
  expect(isCloudDiscoveryUnavailable(error)).toBe(true)
  expect(isNativeRefreshAuthRejection(error)).toBe(false)
  expect(discoveries).toBe(1)
  expect(logs.join('\n')).toMatch(/agent discovery failed/)

  await expect(auth.signIn(AGENT_URL)).rejects.toMatchObject({ cloudDiscoveryUnavailable: true })
  expect(exchanged).toEqual([])
})

test('N3: an explicit sign-in with no portal session surfaces the sign-in prompt, not a transient error', async () => {
  const { auth, exchanged } = makeAuth({
    discover: async () => {
      throw Object.assign(new Error('signed out'), { needsCloudLogin: true })
    }
  })

  await expect(auth.signIn(AGENT_URL)).rejects.toMatchObject({ needsCloudLogin: true })
  expect(exchanged).toEqual([])
})

// --- P1: a persisted registry row never authorizes an exchange ---

/**
 * main.ts wiring end to end: one persisted registry, the agent auth over it,
 * and the production coordinator deps, on a shared clock. The portal answer
 * (`portal.rows`) can change between calls; `restart()` re-reads the disk.
 */
function makeCloud(opts: { disk?: ReturnType<typeof memoryIo>; rows?: any[]; savedUrls?: string[] } = {}) {
  const disk = opts.disk ?? memoryIo()
  const portal = { rows: opts.rows ?? [], discoveries: 0, fail: null as null | Error }
  const clock = { ms: 1_000_000 }
  const store = new Map<string, NativeTokenSet>()
  const exchanged: string[] = []
  const logs: string[] = []
  const saved = new Set(opts.savedUrls ?? [])

  const build = () => {
    const registry = createCloudAgentRegistry(disk.io, normalizeRemoteBaseUrl)

    const exchangeForAgent = async (agentId: string) => {
      exchanged.push(agentId)

      return agentTokens({
        accessToken: `REAL-AGENT-TOKEN-${agentId}-${exchanged.length}`,
        userId: agentId,
        expiresAt: Math.floor(clock.ms / 1_000) + 900
      })
    }

    const auth = createCloudAgentAuth({
      normalizeBaseUrl: normalizeRemoteBaseUrl,
      registry,
      exchangeForAgent,
      // Like main.ts: explicit bearer mutations go through the coordinator.
      storeAgentTokens: (url, tokens) => coordinator.storeTokens(url, tokens),
      clearAgentTokens: url => coordinator.clearTokens(url),
      listStoredTokenUrls: () => [...store.keys()],
      loadStoredTokens: url => store.get(url) ?? null,
      clearPortalSession: () => undefined,
      discoverAgents: async () => {
        portal.discoveries++

        if (portal.fail) {
          throw portal.fail
        }

        return portal.rows
      },
      log: line => logs.push(line),
      nowMs: () => clock.ms
    })

    const coordinator = createNativeAccessTokenCoordinator(
      createNativeTokenCoordinatorDeps({
        loadTokens: url => store.get(url) ?? null,
        storeTokens: (url, tokens) => void store.set(url, tokens),
        clearTokens: url => void store.delete(url),
        normalizeBaseUrl: normalizeRemoteBaseUrl,
        nowSeconds: () => Math.floor(clock.ms / 1_000),
        tokenNeedsRefresh,
        refreshGatewayTokens: async () => {
          throw httpStatusError(401, 'gateway refresh rejected')
        },
        isCloudAgentUrl: url => registry.agentIdFor(url) !== null,
        confirmedCloudAgentId: url => auth.confirmedAgentIdFor(url),
        isCloudBindingCurrent: (url, id) => auth.isBindingCurrent(url, id),
        exchangeForAgent,
        hasLivePortalSession: () => true,
        isSavedCloudConnection: url => saved.has(url)
      })
    )

    return { registry, auth, coordinator }
  }

  let built = build()

  return {
    disk,
    portal,
    clock,
    store,
    exchanged,
    logs,
    get registry() {
      return built.registry
    },
    get auth() {
      return built.auth
    },
    get coordinator() {
      return built.coordinator
    },
    restart() {
      built = build()
    },
    expireStored(url: string) {
      const current = store.get(url)

      if (current) {
        store.set(url, { ...current, expiresAt: Math.floor(clock.ms / 1_000) - 1 })
      }
    }
  }
}

const FIVE_MIN_MS = 5 * 60_000

test('P1(a): stale persisted U→A, portal now U→B, empty-token saved-connection bootstrap → exchanges B, never A', async () => {
  const cloud = makeCloud({ rows: [{ id: 'agt_old', dashboardUrl: AGENT_URL }], savedUrls: [AGENT_URL] })

  await cloud.auth.signIn(AGENT_URL)
  expect(cloud.exchanged).toEqual(['agt_old'])

  // Signed out (bearers gone), restart, later the portal re-homed the URL.
  cloud.store.clear()
  cloud.restart()
  cloud.clock.ms += FIVE_MIN_MS + 1
  cloud.portal.rows = [{ id: 'agt_new', dashboardUrl: AGENT_URL }]

  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_new-2')
  expect(cloud.exchanged).toEqual(['agt_old', 'agt_new'])
})

test('P1(b): stale persisted U→A, portal now U→B, expired-bearer re-exchange → B, never A', async () => {
  const cloud = makeCloud({ rows: [{ id: 'agt_old', dashboardUrl: AGENT_URL }], savedUrls: [AGENT_URL] })

  await cloud.auth.signIn(AGENT_URL)
  cloud.restart()
  cloud.clock.ms += FIVE_MIN_MS + 1
  cloud.portal.rows = [{ id: 'agt_new', dashboardUrl: AGENT_URL }]
  cloud.expireStored(AGENT_URL)

  // The in-flight refresh started from the old bearer, which the reconciling
  // discovery cleared (epoch-fenced): it fails "auth changed" without an
  // exchange; the retry exchanges from the freshly confirmed binding.
  await expect(cloud.coordinator.ensure(AGENT_URL)).rejects.toBeInstanceOf(NativeAuthChangedError)
  // No exchange at all on the fenced flight — not even for the new agent.
  expect(cloud.exchanged).toEqual(['agt_old'])
  expect(cloud.store.has(AGENT_URL)).toBe(false)
  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_new-2')
  expect(cloud.exchanged).toEqual(['agt_old', 'agt_new'])
  expect(cloud.portal.discoveries).toBe(2)

  // The forced path (a 401 on a still-valid bearer) is confirmed the same way.
  cloud.clock.ms += FIVE_MIN_MS + 1
  cloud.portal.rows = [{ id: 'agt_newer', dashboardUrl: AGENT_URL }]
  const current = cloud.store.get(AGENT_URL)!
  cloud.store.set(AGENT_URL, { ...current, expiresAt: Math.floor(cloud.clock.ms / 1_000) + 900 })

  await expect(
    cloud.coordinator.ensure(AGENT_URL, { forceRefresh: true, rejectedAccessToken: current.accessToken })
  ).rejects.toBeInstanceOf(NativeAuthChangedError)
  expect(cloud.exchanged).toEqual(['agt_old', 'agt_new'])
  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_newer-3')
  expect(cloud.exchanged).toEqual(['agt_old', 'agt_new', 'agt_newer'])
})

test("P1(c): the portal no longer lists U → no exchange, U's agent bearer is cleared, not found", async () => {
  const cloud = makeCloud({ rows: [{ id: 'agt_old', dashboardUrl: AGENT_URL }], savedUrls: [AGENT_URL] })

  await cloud.auth.signIn(AGENT_URL)
  cloud.clock.ms += FIVE_MIN_MS + 1
  cloud.portal.rows = []
  cloud.expireStored(AGENT_URL)

  // The refresh in flight is fenced off by the clear; nothing is exchanged,
  // and the next call finds the connection signed out.
  await expect(cloud.coordinator.ensure(AGENT_URL)).rejects.toBeInstanceOf(NativeAuthChangedError)
  expect(cloud.exchanged).toEqual(['agt_old'])
  expect(cloud.store.has(AGENT_URL)).toBe(false)
  expect(cloud.registry.agentIdFor(AGENT_URL)).toBeNull()
  cloud.clock.ms += 61_000
  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBeNull()
  expect(cloud.exchanged).toEqual(['agt_old'])

  // A still-valid bearer for a URL the portal dropped is cleared by the
  // reconciling discovery itself, and an explicit sign-in is refused.
  const other = makeCloud({ rows: [{ id: 'agt_old', dashboardUrl: AGENT_URL }] })

  await other.auth.signIn(AGENT_URL)
  other.clock.ms += FIVE_MIN_MS + 1
  other.portal.rows = []
  await expect(other.auth.signIn(AGENT_URL)).rejects.toThrow(/could not find this agent/i)
  expect(other.exchanged).toEqual(['agt_old'])
  expect(other.store.has(AGENT_URL)).toBe(false)
})

test('P1(d): portal rows that normalize to the same URL fail closed — no exchange for that URL', async () => {
  const cloud = makeCloud({
    rows: [
      { id: 'agt_a', dashboardUrl: 'https://a.example/' },
      { id: 'agt_b', dashboardUrl: 'https://A.example:443' },
      { id: 'agt_1', dashboardUrl: AGENT_URL }
    ],
    savedUrls: ['https://a.example']
  })

  await expect(cloud.auth.signIn('https://a.example')).rejects.toThrow(/could not find this agent/i)
  await expect(cloud.coordinator.ensure('https://a.example')).resolves.toBeNull()
  expect(cloud.exchanged).toEqual([])
  expect(cloud.registry.agentIdFor('https://a.example')).toBeNull()
  // The unambiguous row is unaffected.
  await cloud.auth.signIn(AGENT_URL)
  expect(cloud.exchanged).toEqual(['agt_1'])
  expect(cloud.logs.join('\n')).not.toMatch(/agt_a|agt_b|token|bearer/i)
})

test('P1(e): a binding confirmed under 5 minutes ago is used without another portal discovery', async () => {
  const cloud = makeCloud({ rows: [{ id: 'agt_1', dashboardUrl: AGENT_URL }], savedUrls: [AGENT_URL] })

  await cloud.auth.signIn(AGENT_URL)
  expect(cloud.portal.discoveries).toBe(1)

  cloud.clock.ms += FIVE_MIN_MS - 60_000
  cloud.expireStored(AGENT_URL)
  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-2')

  cloud.store.clear()
  await expect(cloud.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_1-3')
  await cloud.auth.signIn(AGENT_URL, 'agt_1')
  expect(cloud.portal.discoveries).toBe(1)
  expect(cloud.exchanged).toEqual(['agt_1', 'agt_1', 'agt_1', 'agt_1'])
})

test('P1(f): older registry formats migrate as unconfirmed and force one discovery before any exchange', async () => {
  for (const legacy of [
    JSON.stringify({ [AGENT_URL]: 'agt_old' }),
    JSON.stringify({ orgId: null, agents: { [AGENT_URL]: 'agt_old' } })
  ]) {
    const bootstrap = makeCloud({
      disk: memoryIo(legacy),
      rows: [{ id: 'agt_new', dashboardUrl: AGENT_URL }],
      savedUrls: [AGENT_URL]
    })

    // Still a routing hint...
    expect(bootstrap.registry.agentIdFor(AGENT_URL)).toBe('agt_old')
    // ...but never an audience.
    await expect(bootstrap.coordinator.ensure(AGENT_URL)).resolves.toBe('REAL-AGENT-TOKEN-agt_new-1')
    expect(bootstrap.portal.discoveries).toBe(1)
    expect(bootstrap.exchanged).toEqual(['agt_new'])

    const hinted = makeCloud({ disk: memoryIo(legacy), rows: [{ id: 'agt_new', dashboardUrl: AGENT_URL }] })

    await hinted.auth.signIn(AGENT_URL, 'agt_old')
    expect(hinted.portal.discoveries).toBe(1)
    expect(hinted.exchanged).toEqual(['agt_new'])
  }
})

test('an older discovery that lands after a newer one is dropped: a stale snapshot cannot re-bind a URL', () => {
  const { auth, registry } = makeAuth()

  const older = auth.beginDiscovery()
  const newer = auth.beginDiscovery()

  auth.reconcileDiscovered([{ id: 'agt_new', dashboardUrl: AGENT_URL }], newer)
  auth.reconcileDiscovered([{ id: 'agt_old', dashboardUrl: AGENT_URL }], older)

  expect(registry.bindingFor(AGENT_URL)?.agentId).toBe('agt_new')
})

test('agent sign-in never stores a bearer for a binding re-bound while its exchange was in flight', async () => {
  let auth: ReturnType<typeof makeAuth>['auth']

  const made = makeAuth({
    exchange: async agentId => {
      // A Settings refresh lands mid-flight and moves the URL to another agent.
      auth.reconcileDiscovered([{ id: 'agt_new', dashboardUrl: AGENT_URL }])

      return agentTokens({ userId: agentId })
    }
  })

  auth = made.auth
  auth.reconcileDiscovered([{ id: 'agt_old', dashboardUrl: AGENT_URL }])

  await expect(auth.signIn(AGENT_URL)).rejects.toThrow(CLOUD_AGENT_NOT_FOUND_MESSAGE)
  expect(made.stored.get(AGENT_URL)).toBeUndefined()
})
