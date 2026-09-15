import { describe, expect, it, vi } from 'vitest'

import { createGatewayWsCookieStore, type GatewayCookie } from './gateway-ws-cookie'

const LEGACY = 'persist:hermes-oauth'
const GATEWAY = 'https://gateway.example'
const WS_URL = 'wss://gateway.example/api/ws?ticket=fresh'

// A gateway behind a forward-auth proxy: the proxy's session cookie alongside
// Hermes' own, as they sit in the OAuth partition's jar.
const proxyJar: GatewayCookie[] = [
  { name: 'proxy_session', value: 'proxy-value' },
  { name: 'hermes_session_at', value: 'at-value' }
]

const EXPECTED = 'proxy_session=proxy-value; hermes_session_at=at-value'

function createStore(
  jars: Record<string, GatewayCookie[] | null> = { [GATEWAY]: proxyJar },
  options: { partitions?: Record<string, string>; ttlMs?: number } = {}
) {
  let clock = 1_000
  const onError = vi.fn()
  const readCookies = vi.fn(async (baseUrl: string) => jars[baseUrl] ?? null)

  const store = createGatewayWsCookieStore({
    readCookies,
    resolvePartition: baseUrl => options.partitions?.[baseUrl] ?? LEGACY,
    now: () => clock,
    ttlMs: options.ttlMs,
    onError
  })

  return { advance: (ms: number) => (clock += ms), onError, readCookies, store }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void

  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })

  // A rejection is always awaited through register(); this only keeps Node from
  // seeing an unhandled rejection in the tick between reject() and that await.
  promise.catch(() => undefined)

  return { promise, reject, resolve }
}

function cookieOn(store: ReturnType<typeof createGatewayWsCookieStore>, url: string, resourceType = 'webSocket') {
  const response = store.apply({ url, resourceType, requestHeaders: { Origin: 'app://hermes' } }, {})

  return response?.requestHeaders?.Cookie
}

describe('gateway WebSocket cookie forwarding', () => {
  it('authorizes the exact freshly minted upgrade url', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('leaves the request untouched when nothing is authorized', async () => {
    const { store } = createStore()

    const response = store.apply({ url: WS_URL, resourceType: 'webSocket', requestHeaders: { Origin: 'x' } }, {})

    expect(response.requestHeaders).toBeUndefined()
  })

  // The blocker this store exists to answer: ordinary traffic to the gateway
  // must not pick up the credential just because it matches the origin.
  it('forwards nothing on ordinary HTTP(S) requests under the gateway base', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    for (const url of [`${GATEWAY}/api/status`, `${GATEWAY}/api/ws`, `${GATEWAY}/`, `${GATEWAY}/api/agents?x=1`]) {
      expect(cookieOn(store, url, 'xhr')).toBeUndefined()
      expect(cookieOn(store, url, 'webSocket')).toBeUndefined()
    }
  })

  it('forwards nothing on sibling paths, other origins, or unrelated sockets', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    for (const url of [
      'wss://gateway.example/api/ws/sibling?ticket=fresh',
      'wss://gateway.example/other/api/ws?ticket=fresh',
      'wss://other.example/api/ws?ticket=fresh',
      'wss://gateway.example.evil.test/api/ws?ticket=fresh',
      'wss://gateway.example/api/ws',
      'wss://gateway.example/api/ws?ticket=fresh&extra=1'
    ]) {
      expect(cookieOn(store, url)).toBeUndefined()
    }
  })

  it('refuses a non-WebSocket resource type on the authorized url itself', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL, 'xhr')).toBeUndefined()
    expect(cookieOn(store, WS_URL, 'subFrame')).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  // Ticket rotation: the renderer re-mints before every connect, so only the
  // newest url may carry authority.
  it('drops the previous ticket url when the next one is registered', async () => {
    const { store } = createStore()
    const stale = 'wss://gateway.example/api/ws?ticket=stale'

    await store.register(stale, GATEWAY)
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, stale)).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('expires an upgrade that never happens', async () => {
    const { advance, store } = createStore(undefined, { ttlMs: 60_000 })

    await store.register(WS_URL, GATEWAY)
    advance(59_999)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)

    // A fresh registration, so this measures the TTL rather than the
    // consumption the assertion above already performed.
    await store.register(WS_URL, GATEWAY)
    advance(60_000)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // The url carries a single-use ticket, so the authority it needed is spent
  // once the upgrade has taken it.
  it('consumes the authorization with the upgrade that uses it', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('does not consume the authorization on a refused request', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL, 'xhr')).toBeUndefined()
    expect(cookieOn(store, `${GATEWAY}/api/status`)).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('drops authority on sign-out of that gateway', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)
    store.forget(GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // hermes:cloud:logout clears the PORTAL baseUrl, but the portal and a Cloud
  // agent share the legacy jar — so the agent's url must lose authority too.
  it('drops authority for every url sharing the signed-out jar', async () => {
    const agent = 'https://agent.hermes.example'
    const agentWs = 'wss://agent.hermes.example/api/ws?ticket=fresh'
    const { store } = createStore({ [agent]: proxyJar })

    await store.register(agentWs, agent)
    store.forget('https://portal.nousresearch.com')

    expect(cookieOn(store, agentWs)).toBeUndefined()
  })

  // resolvePartition reads the live connections registry, so the partition a
  // url resolves to can change while an entry is live. Sign-out must still
  // reach the entry it authorized.
  it('drops authority even when the url has since moved partition', async () => {
    const partitions: Record<string, string> = { [GATEWAY]: 'persist:hermes-oauth-one' }

    const store = createGatewayWsCookieStore({
      readCookies: async () => proxyJar,
      resolvePartition: baseUrl => partitions[baseUrl] ?? LEGACY
    })

    await store.register(WS_URL, GATEWAY, 'ws-url:default')

    // A registry edit re-resolves this gateway to a different jar.
    partitions[GATEWAY] = 'persist:hermes-oauth-two'
    store.forget(GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('fences a pending read against a sign-out that resolved another partition', async () => {
    const partitions: Record<string, string> = { [GATEWAY]: 'persist:hermes-oauth-one' }
    const jar = deferred<GatewayCookie[]>()

    const store = createGatewayWsCookieStore({
      readCookies: () => jar.promise,
      resolvePartition: baseUrl => partitions[baseUrl] ?? LEGACY
    })

    const pending = store.register(WS_URL, GATEWAY, 'ws-url:default')

    partitions[GATEWAY] = 'persist:hermes-oauth-two'

    const signOutDone = store.forget(GATEWAY)

    signOutDone()
    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // Overlapping logouts release their scopes independently. When the registry
  // moves a gateway between partitions mid-logout, the partition scope can go
  // idle while the base-url scope is still busy; the partition must still be
  // fenced, or a read taken in its window republishes after both finish.
  it('fences each logout scope independently across a partition move', async () => {
    const a = 'https://a.example'
    const b = 'https://b.example'
    const url = 'wss://b.example/api/ws?ticket=old'
    const read = deferred<GatewayCookie[]>()
    let partitionA = 'persist:shared'

    const store = createGatewayWsCookieStore({
      readCookies: () => read.promise,
      resolvePartition: baseUrl => (baseUrl === a ? partitionA : 'persist:shared')
    })

    const firstDone = store.forget(a)

    partitionA = 'persist:dedicated-a'

    const secondDone = store.forget(a)
    const pending = store.register(url, b)

    firstDone()
    secondDone()
    read.resolve(proxyJar)
    await pending

    expect(cookieOn(store, url)).toBeUndefined()
  })

  it('still releases a scope only once its own overlapping logouts finish', async () => {
    const { store } = createStore()

    const first = store.forget(GATEWAY)
    const second = store.forget(GATEWAY)

    first()
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    second()
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('keeps two gateways on separate partitions independent', async () => {
    const one = 'https://one.example'
    const two = 'https://two.example'
    const oneWs = 'wss://one.example/api/ws?ticket=a'
    const twoWs = 'wss://two.example/api/ws?ticket=b'

    const { store } = createStore(
      { [one]: [{ name: 'session', value: 'one' }], [two]: [{ name: 'session', value: 'two' }] },
      { partitions: { [one]: 'persist:hermes-oauth-one', [two]: 'persist:hermes-oauth-two' } }
    )

    await store.register(oneWs, one)
    await store.register(twoWs, two)

    expect(cookieOn(store, oneWs)).toBe('session=one')
    expect(cookieOn(store, twoWs)).toBe('session=two')

    // Each upgrade consumed its authorization, so re-arm before asking what
    // sign-out takes away -- otherwise the assertions below hold vacuously.
    await store.register(oneWs, one)
    await store.register(twoWs, two)
    store.forget(one)

    expect(cookieOn(store, oneWs)).toBeUndefined()
    expect(cookieOn(store, twoWs)).toBe('session=two')
  })

  // A shared remote serves one socket per profile at a single baseUrl, and a
  // descriptor build mints alongside them. The gateway alone is therefore too
  // coarse an owner: each consumer may only retire its own previous url.
  it('keeps several consumers of one gateway independent', async () => {
    const { store } = createStore()
    const primaryWs = 'wss://gateway.example/api/ws?ticket=primary'
    const pooledWs = 'wss://gateway.example/api/ws?ticket=pooled&profile=work'
    const descriptorWs = 'wss://gateway.example/api/ws?ticket=descriptor'

    await store.register(primaryWs, GATEWAY, 'ws-url:')
    await store.register(pooledWs, GATEWAY, 'registry:cloud:work')
    await store.register(descriptorWs, GATEWAY, 'descriptor:settings')

    expect(cookieOn(store, primaryWs)).toBe(EXPECTED)
    expect(cookieOn(store, pooledWs)).toBe(EXPECTED)
    expect(cookieOn(store, descriptorWs)).toBe(EXPECTED)
  })

  it("retires only the re-minting consumer's own previous url", async () => {
    const { store } = createStore()
    const pooledWs = 'wss://gateway.example/api/ws?ticket=pooled&profile=work'
    const stale = 'wss://gateway.example/api/ws?ticket=stale'
    const rotated = 'wss://gateway.example/api/ws?ticket=rotated'

    await store.register(pooledWs, GATEWAY, 'registry:cloud:work')
    await store.register(stale, GATEWAY, 'ws-url:')
    await store.register(rotated, GATEWAY, 'ws-url:')

    expect(cookieOn(store, stale)).toBeUndefined()
    expect(cookieOn(store, rotated)).toBe(EXPECTED)
    expect(cookieOn(store, pooledWs)).toBe(EXPECTED)

    // Sign-out is still partition-wide, so it takes every consumer with it.
    await store.register(rotated, GATEWAY, 'ws-url:')
    await store.register(pooledWs, GATEWAY, 'registry:cloud:work')
    store.forget(GATEWAY)

    expect(cookieOn(store, rotated)).toBeUndefined()
    expect(cookieOn(store, pooledWs)).toBeUndefined()
  })

  it('never lets one consumer label span two gateways', async () => {
    const other = 'https://other.example'
    const otherWs = 'wss://other.example/api/ws?ticket=other'
    const { store } = createStore({ [GATEWAY]: proxyJar, [other]: [{ name: 'session', value: 'other' }] })

    await store.register(WS_URL, GATEWAY, 'ws-url:')
    await store.register(otherWs, other, 'ws-url:')

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
    expect(cookieOn(store, otherWs)).toBe('session=other')
  })

  // Two Cloud agents deliberately share the legacy jar (oauth-partition.ts), so
  // the partition is the right scope for a sign-out but NOT for replacement:
  // registering B must not cancel A's in-flight handshake.
  it('keeps two gateways sharing one partition independent', async () => {
    const a = 'https://agent-a.example'
    const b = 'https://agent-b.example'
    const aWs = 'wss://agent-a.example/api/ws?ticket=a'
    const bWs = 'wss://agent-b.example/api/ws?ticket=b'

    const { store } = createStore({
      [a]: [{ name: 'proxy_session', value: 'a-value' }],
      [b]: [{ name: 'proxy_session', value: 'b-value' }]
    })

    await store.register(aWs, a)
    await store.register(bWs, b)

    expect(cookieOn(store, aWs)).toBe('proxy_session=a-value')
    expect(cookieOn(store, bWs)).toBe('proxy_session=b-value')

    await store.register(aWs, a)
    await store.register(bWs, b)

    // A read that finds no jar for C must not revoke A either.
    await store.register('wss://agent-c.example/api/ws?ticket=c', 'https://agent-c.example')

    expect(cookieOn(store, aWs)).toBe('proxy_session=a-value')

    // Sign-out still empties the shared jar for both.
    await store.register(aWs, a)
    store.forget(a)

    expect(cookieOn(store, aWs)).toBeUndefined()
    expect(cookieOn(store, bWs)).toBeUndefined()
  })

  // The jar read is asynchronous, so a registration can finish after the work
  // that superseded or revoked it. Neither may publish or delete.
  it('does not republish a cookie read that resolves after sign-out', async () => {
    const jar = deferred<GatewayCookie[]>()

    const store = createGatewayWsCookieStore({
      readCookies: () => jar.promise,
      resolvePartition: () => LEGACY
    })

    const pending = store.register(WS_URL, GATEWAY)

    store.forget(GATEWAY)
    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // clearOauthSession() empties the jar asynchronously, so a read taken while
  // that is in flight sees cookies that are already on their way out.
  it('refuses a read taken while sign-out is still clearing the jar', async () => {
    const { store } = createStore()

    const signOutDone = store.forget(GATEWAY)

    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    signOutDone()

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('refuses a mid-sign-out read that only resolves after the jar is cleared', async () => {
    const jar = deferred<GatewayCookie[]>()

    const store = createGatewayWsCookieStore({
      readCookies: () => jar.promise,
      resolvePartition: () => LEGACY
    })

    const signOutDone = store.forget(GATEWAY)
    const pending = store.register(WS_URL, GATEWAY)

    signOutDone()
    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('lets a registration started after sign-out completes authorize normally', async () => {
    const { store } = createStore()

    store.forget(GATEWAY)()
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('reopens only once concurrent sign-outs have all finished', async () => {
    const { store } = createStore()

    const first = store.forget(GATEWAY)
    const second = store.forget(GATEWAY)

    first()
    first()
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    second()
    await store.register(WS_URL, GATEWAY)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('does not let a slow read replace the newer ticket url it lost to', async () => {
    const stale = 'wss://gateway.example/api/ws?ticket=stale'
    const slow = deferred<GatewayCookie[]>()
    let reads = 0

    const store = createGatewayWsCookieStore({
      readCookies: () => (++reads === 1 ? slow.promise : Promise.resolve(proxyJar)),
      resolvePartition: () => LEGACY
    })

    const pending = store.register(stale, GATEWAY)

    await store.register(WS_URL, GATEWAY)
    slow.resolve(proxyJar)
    await pending

    expect(cookieOn(store, stale)).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('does not let a slow failed read revoke the newer ticket url', async () => {
    const onError = vi.fn()
    const slow = deferred<GatewayCookie[]>()
    let reads = 0

    const store = createGatewayWsCookieStore({
      readCookies: () => (++reads === 1 ? slow.promise : Promise.resolve(proxyJar)),
      resolvePartition: () => LEGACY,
      onError
    })

    const pending = store.register('wss://gateway.example/api/ws?ticket=stale', GATEWAY)

    await store.register(WS_URL, GATEWAY)
    slow.reject(new Error('partition unavailable'))
    await pending

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
    expect(onError).toHaveBeenCalledWith('partition unavailable')
  })

  // The generation ledger is bounded, so an owner can age out of it. A pending
  // read that finds its generation gone must stand down rather than publish
  // against a number some later registration might reuse.
  it('stands down when its owner has aged out of the generation ledger', async () => {
    const jar = deferred<GatewayCookie[]>()
    let reads = 0

    const store = createGatewayWsCookieStore({
      readCookies: () => (++reads === 1 ? jar.promise : Promise.resolve(proxyJar)),
      resolvePartition: () => LEGACY
    })

    const pending = store.register(WS_URL, GATEWAY, 'ws-url:default')

    for (let index = 0; index < 300; index += 1) {
      await store.register(`wss://gateway.example/api/ws?ticket=${index}`, GATEWAY, `churn:${index}`)
    }

    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('preserves headers already merged for the request and appends to any Cookie', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY)

    const merged = store.apply(
      { url: WS_URL, resourceType: 'webSocket' },
      { requestHeaders: { 'CF-Access-Client-Id': 'client-id' } }
    )

    expect(merged.requestHeaders).toEqual({ 'CF-Access-Client-Id': 'client-id', Cookie: EXPECTED })

    await store.register(WS_URL, GATEWAY)

    const appended = store.apply(
      { url: WS_URL, resourceType: 'webSocket' },
      { requestHeaders: { cookie: 'preexisting=1' } }
    )

    expect(appended.requestHeaders).toEqual({ cookie: `preexisting=1; ${EXPECTED}` })
  })

  it('authorizes nothing when the jar is empty or the partition is gone', async () => {
    const jars: Record<string, GatewayCookie[] | null> = { [GATEWAY]: [] }
    const { store } = createStore(jars)

    await store.register(WS_URL, GATEWAY)
    expect(cookieOn(store, WS_URL)).toBeUndefined()

    jars[GATEWAY] = null
    await store.register(WS_URL, GATEWAY)
    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('revokes the previous url when a later read finds the jar emptied', async () => {
    const jars: Record<string, GatewayCookie[] | null> = { [GATEWAY]: proxyJar }
    const { store } = createStore(jars)

    await store.register(WS_URL, GATEWAY)
    jars[GATEWAY] = []
    await store.register('wss://gateway.example/api/ws?ticket=next', GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
    expect(cookieOn(store, 'wss://gateway.example/api/ws?ticket=next')).toBeUndefined()
  })

  it('reports a failed jar read, forwards nothing, and revokes prior authority', async () => {
    const { onError, store } = createStore()

    await store.register(WS_URL, GATEWAY)

    const failing = createGatewayWsCookieStore({
      readCookies: async () => {
        throw new Error('partition unavailable')
      },
      resolvePartition: () => LEGACY,
      onError
    })

    await failing.register(WS_URL, GATEWAY)

    expect(cookieOn(failing, WS_URL)).toBeUndefined()
    expect(onError).toHaveBeenCalledWith('partition unavailable')
  })

  it('ignores a missing url or baseUrl instead of touching live authority', async () => {
    const { readCookies, store } = createStore()

    await store.register(WS_URL, GATEWAY)
    readCookies.mockClear()

    await store.register('', GATEWAY)
    await store.register(WS_URL, '')
    store.forget('')

    expect(readCookies).not.toHaveBeenCalled()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })
})
