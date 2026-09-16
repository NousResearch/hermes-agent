import { describe, expect, it, vi } from 'vitest'

import { cookieAppliesToHost, createGatewayWsCookieStore, type GatewayCookie } from './gateway-ws-cookie'

const LEGACY = 'persist:hermes-oauth'
const GATEWAY = 'https://gateway.example'
const WS_URL = 'wss://gateway.example/api/ws?ticket=fresh'
// One consumer stands in for "the caller that re-mints for this socket" wherever
// a test is not about consumer identity itself. Sharing it keeps these calls
// modelling ONE socket, so rotation still retires its own url.
const CONSUMER = 'ws-url:default:w1:default'

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
  // Keyed by the url the store asks for, falling back to the gateway's own
  // entry, so a test can describe a path-scoped jar without re-implementing
  // cookie matching.
  const readCookies = vi.fn(async (cookieUrl: string, baseUrl: string) => jars[cookieUrl] ?? jars[baseUrl] ?? null)

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

// Sign-out deletes what this feature is willing to forward, so the two have to
// agree on which cookies belong to a gateway.
describe('cookie ownership for sign-out', () => {
  it('matches the exact host and a cookie set on a parent domain', () => {
    expect(cookieAppliesToHost({ domain: 'gateway.example' }, 'gateway.example')).toBe(true)
    // A forward-auth proxy commonly sets its session on the parent domain, and
    // sign-out has to reach it or the session outlives the logout.
    expect(cookieAppliesToHost({ domain: '.example' }, 'gateway.example')).toBe(true)
    expect(cookieAppliesToHost({ domain: 'example' }, 'gateway.example')).toBe(true)
  })

  it('does not match siblings, subdomains, or lookalike suffixes', () => {
    // Deleting a sibling would sign another gateway out of the shared jar.
    expect(cookieAppliesToHost({ domain: 'other.example' }, 'gateway.example')).toBe(false)
    expect(cookieAppliesToHost({ domain: 'sub.gateway.example' }, 'gateway.example')).toBe(false)
    expect(cookieAppliesToHost({ domain: 'gateway.example' }, 'evilgateway.example')).toBe(false)
    expect(cookieAppliesToHost({ domain: 'ateway.example' }, 'gateway.example')).toBe(false)
  })

  it('is case-insensitive and refuses empty input', () => {
    expect(cookieAppliesToHost({ domain: 'GATEWAY.Example' }, 'gateway.example')).toBe(true)
    expect(cookieAppliesToHost({ domain: '' }, 'gateway.example')).toBe(false)
    expect(cookieAppliesToHost(null, 'gateway.example')).toBe(false)
    expect(cookieAppliesToHost({ domain: 'gateway.example' }, '')).toBe(false)
  })
})

describe('gateway WebSocket cookie forwarding', () => {
  it('authorizes the exact freshly minted upgrade url', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  // Chromium selects cookies for the url it is handed. A proxy cookie scoped to
  // `Path=/api/` does not apply to the gateway root, so reading the base url
  // returned a snapshot missing exactly the credential being forwarded.
  it('selects cookies for the upgrade url, not the gateway base', async () => {
    const { readCookies, store } = createStore({
      [GATEWAY]: [{ name: 'hermes_session_at', value: 'at-value' }],
      'https://gateway.example/api/ws': proxyJar
    })

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(readCookies).toHaveBeenCalledWith('https://gateway.example/api/ws', GATEWAY)
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('selects cookies for a reverse-proxy prefix path', async () => {
    const prefix = 'https://prefix.example/hermes'
    const prefixWs = 'wss://prefix.example/hermes/api/ws?ticket=fresh'
    const { readCookies, store } = createStore({
      'https://prefix.example/hermes/api/ws': [{ name: 'prefix_proxy', value: 'prefix-value' }]
    })

    await store.register(prefixWs, prefix, CONSUMER)

    expect(readCookies).toHaveBeenCalledWith('https://prefix.example/hermes/api/ws', prefix)
    expect(cookieOn(store, prefixWs)).toBe('prefix_proxy=prefix-value')
  })

  it('asks for no path the upgrade does not target', async () => {
    const { readCookies, store } = createStore({ 'https://gateway.example/api/ws': proxyJar })

    await store.register(WS_URL, GATEWAY, CONSUMER)

    // An unrelated path's cookies are never requested, so they can never be
    // forwarded: selection is the cookie store's job, on this exact url.
    expect(readCookies).toHaveBeenCalledTimes(1)
    expect(readCookies).not.toHaveBeenCalledWith(expect.stringContaining('/other/'), expect.anything())
  })

  it('maps ws:// to http:// and falls back to the base url when unparseable', async () => {
    const insecure = createStore({ 'http://gateway.example/api/ws': [{ name: 'proxy_session', value: 'plain' }] })

    await insecure.store.register('ws://gateway.example/api/ws?ticket=fresh', 'http://gateway.example', CONSUMER)

    expect(insecure.readCookies).toHaveBeenCalledWith('http://gateway.example/api/ws', 'http://gateway.example')

    const broken = createStore()

    await broken.store.register('not a url', GATEWAY, CONSUMER)

    expect(broken.readCookies).toHaveBeenCalledWith(GATEWAY, GATEWAY)
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

    await store.register(WS_URL, GATEWAY, CONSUMER)

    for (const url of [`${GATEWAY}/api/status`, `${GATEWAY}/api/ws`, `${GATEWAY}/`, `${GATEWAY}/api/agents?x=1`]) {
      expect(cookieOn(store, url, 'xhr')).toBeUndefined()
      expect(cookieOn(store, url, 'webSocket')).toBeUndefined()
    }
  })

  it('forwards nothing on sibling paths, other origins, or unrelated sockets', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

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

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL, 'xhr')).toBeUndefined()
    expect(cookieOn(store, WS_URL, 'subFrame')).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  // Ticket rotation: the renderer re-mints before every connect, so only the
  // newest url may carry authority.
  it('drops the previous ticket url when the next one is registered', async () => {
    const { store } = createStore()
    const stale = 'wss://gateway.example/api/ws?ticket=stale'

    await store.register(stale, GATEWAY, CONSUMER)
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, stale)).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('expires an upgrade that never happens', async () => {
    const { advance, store } = createStore(undefined, { ttlMs: 60_000 })

    await store.register(WS_URL, GATEWAY, CONSUMER)
    advance(59_999)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)

    // A fresh registration, so this measures the TTL rather than the
    // consumption the assertion above already performed.
    await store.register(WS_URL, GATEWAY, CONSUMER)
    advance(60_000)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // The url carries a single-use ticket, so the authority it needed is spent
  // once the upgrade has taken it.
  it('consumes the authorization with the upgrade that uses it', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('does not consume the authorization on a refused request', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL, 'xhr')).toBeUndefined()
    expect(cookieOn(store, `${GATEWAY}/api/status`)).toBeUndefined()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('drops authority on sign-out of that gateway', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)
    store.forget(GATEWAY)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // hermes:cloud:logout clears the PORTAL baseUrl, but the portal and a Cloud
  // agent share the legacy jar — so the agent's url must lose authority too.
  it('drops authority for every url sharing the signed-out jar', async () => {
    const agent = 'https://agent.hermes.example'
    const agentWs = 'wss://agent.hermes.example/api/ws?ticket=fresh'
    const { store } = createStore({ [agent]: proxyJar })

    await store.register(agentWs, agent, CONSUMER)
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
    const pending = store.register(url, b, CONSUMER)

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
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    second()
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  // clearOauthSession passes its filter straight through, so an empty base url
  // clears the WHOLE jar. Nothing may outlive that -- it was the widest clear
  // and the only one the store used to ignore.
  it('revokes every gateway when the whole jar is cleared', async () => {
    const other = 'https://other.example'
    const otherWs = 'wss://other.example/api/ws?ticket=other'
    const { store } = createStore(
      { [GATEWAY]: proxyJar, [other]: [{ name: 'session', value: 'other' }] },
      { partitions: { [other]: 'persist:hermes-oauth-two' } }
    )

    await store.register(WS_URL, GATEWAY, CONSUMER)
    await store.register(otherWs, other, CONSUMER)

    store.forget('')

    expect(cookieOn(store, WS_URL)).toBeUndefined()
    expect(cookieOn(store, otherWs)).toBeUndefined()
  })

  it('fences a read in flight across a whole-jar clear', async () => {
    const jar = deferred<GatewayCookie[]>()
    const store = createGatewayWsCookieStore({
      readCookies: () => jar.promise,
      resolvePartition: () => LEGACY
    })

    const pending = store.register(WS_URL, GATEWAY, CONSUMER)
    const done = store.forget('')

    done()
    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('refuses a read taken while the whole jar is being cleared', async () => {
    const { store } = createStore()

    const done = store.forget('')

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    done()
    await store.register(WS_URL, GATEWAY, CONSUMER)

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

    await store.register(oneWs, one, CONSUMER)
    await store.register(twoWs, two, CONSUMER)

    expect(cookieOn(store, oneWs)).toBe('session=one')
    expect(cookieOn(store, twoWs)).toBe('session=two')

    // Each upgrade consumed its authorization, so re-arm before asking what
    // sign-out takes away -- otherwise the assertions below hold vacuously.
    await store.register(oneWs, one, CONSUMER)
    await store.register(twoWs, two, CONSUMER)
    store.forget(one)

    expect(cookieOn(store, oneWs)).toBeUndefined()
    expect(cookieOn(store, twoWs)).toBe('session=two')
  })

  // A shared remote serves one socket per profile at a single baseUrl, and each
  // window mints its own. The gateway alone is therefore too coarse an owner:
  // each consumer may only retire its own previous url.
  it('keeps several consumers of one gateway independent', async () => {
    const { store } = createStore()
    const primaryWs = 'wss://gateway.example/api/ws?ticket=primary'
    const pooledWs = 'wss://gateway.example/api/ws?ticket=pooled&profile=work'
    const secondWindowWs = 'wss://gateway.example/api/ws?ticket=second-window'

    await store.register(primaryWs, GATEWAY, 'ws-url:')
    await store.register(pooledWs, GATEWAY, 'registry:cloud:work')
    await store.register(secondWindowWs, GATEWAY, 'ws-url:default:w2:default')

    expect(cookieOn(store, primaryWs)).toBe(EXPECTED)
    expect(cookieOn(store, pooledWs)).toBe(EXPECTED)
    expect(cookieOn(store, secondWindowWs)).toBe(EXPECTED)
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

    await store.register(aWs, a, CONSUMER)
    await store.register(bWs, b, CONSUMER)

    expect(cookieOn(store, aWs)).toBe('proxy_session=a-value')
    expect(cookieOn(store, bWs)).toBe('proxy_session=b-value')

    await store.register(aWs, a, CONSUMER)
    await store.register(bWs, b, CONSUMER)

    // A read that finds no jar for C must not revoke A either.
    await store.register('wss://agent-c.example/api/ws?ticket=c', 'https://agent-c.example', CONSUMER)

    expect(cookieOn(store, aWs)).toBe('proxy_session=a-value')

    // Sign-out still empties the shared jar for both.
    await store.register(aWs, a, CONSUMER)
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

    const pending = store.register(WS_URL, GATEWAY, CONSUMER)

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

    await store.register(WS_URL, GATEWAY, CONSUMER)

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
    const pending = store.register(WS_URL, GATEWAY, CONSUMER)

    signOutDone()
    jar.resolve(proxyJar)
    await pending

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  // The contract production uses: the store owns the window, so a caller cannot
  // leave one open and silently disable the feature for the process lifetime.
  it('closes the sign-out window itself when the jar clearing is handed to it', async () => {
    const { store } = createStore()
    let clearing = false

    await store.register(WS_URL, GATEWAY, CONSUMER)

    await store.forgetWhile(GATEWAY, async () => {
      clearing = true
      // Mid-clear: the scope is closed, so a racing read publishes nothing.
      await store.register(WS_URL, GATEWAY, CONSUMER)

      expect(cookieOn(store, WS_URL)).toBeUndefined()
    })

    expect(clearing).toBe(true)

    // Released, so the next sign-in works again.
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('releases the sign-out window even when the jar clearing throws', async () => {
    const { store } = createStore()

    await expect(
      store.forgetWhile(GATEWAY, async () => {
        throw new Error('jar unavailable')
      })
    ).rejects.toThrow('jar unavailable')

    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('lets a registration started after sign-out completes authorize normally', async () => {
    const { store } = createStore()

    store.forget(GATEWAY)()
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })

  it('reopens only once concurrent sign-outs have all finished', async () => {
    const { store } = createStore()

    const first = store.forget(GATEWAY)
    const second = store.forget(GATEWAY)

    first()
    first()
    await store.register(WS_URL, GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBeUndefined()

    second()
    await store.register(WS_URL, GATEWAY, CONSUMER)

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

    const pending = store.register(stale, GATEWAY, CONSUMER)

    await store.register(WS_URL, GATEWAY, CONSUMER)
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

    const pending = store.register('wss://gateway.example/api/ws?ticket=stale', GATEWAY, CONSUMER)

    await store.register(WS_URL, GATEWAY, CONSUMER)
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

    await store.register(WS_URL, GATEWAY, CONSUMER)

    const merged = store.apply(
      { url: WS_URL, resourceType: 'webSocket' },
      { requestHeaders: { 'CF-Access-Client-Id': 'client-id' } }
    )

    expect(merged.requestHeaders).toEqual({ 'CF-Access-Client-Id': 'client-id', Cookie: EXPECTED })

    await store.register(WS_URL, GATEWAY, CONSUMER)

    const appended = store.apply(
      { url: WS_URL, resourceType: 'webSocket' },
      { requestHeaders: { cookie: 'preexisting=1' } }
    )

    expect(appended.requestHeaders).toEqual({ cookie: `preexisting=1; ${EXPECTED}` })
  })

  // The merge itself: where the outgoing headers come from, and what survives.
  it('merges onto the headers the request already carries when the response has none', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    const merged = store.apply(
      { url: WS_URL, resourceType: 'webSocket', requestHeaders: { Origin: 'app://hermes' } },
      {}
    )

    expect(merged.requestHeaders).toEqual({ Cookie: EXPECTED, Origin: 'app://hermes' })
  })

  it('keeps other fields of the response it was handed', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    const merged = store.apply({ url: WS_URL, resourceType: 'webSocket' }, {
      cancel: false,
      requestHeaders: { Origin: 'app://hermes' }
    } as never)

    expect(merged).toEqual({ cancel: false, requestHeaders: { Cookie: EXPECTED, Origin: 'app://hermes' } })
  })

  // Documented fail-open: some call shapes report no resourceType, and the
  // exact-url match is the real gate, so a missing type must still forward.
  it('forwards when Chromium reports no resource type at all', async () => {
    const { store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)

    const merged = store.apply({ url: WS_URL }, {})

    expect(merged.requestHeaders?.Cookie).toBe(EXPECTED)
  })

  it('bounds how many live authorizations it will hold', async () => {
    const { store } = createStore()
    const first = 'wss://gateway.example/api/ws?ticket=first'

    await store.register(first, GATEWAY, 'consumer:first')

    // Each distinct consumer keeps its own entry, so the cap is the only thing
    // standing between many consumers and unbounded growth.
    for (let index = 0; index < 64; index += 1) {
      await store.register(`wss://gateway.example/api/ws?ticket=${index}`, GATEWAY, `consumer:${index}`)
    }

    expect(cookieOn(store, first)).toBeUndefined()
    expect(cookieOn(store, 'wss://gateway.example/api/ws?ticket=63')).toBe(EXPECTED)
  })

  it('applies a default lifetime when the caller sets none', async () => {
    // No ttlMs: the shipped default has to bound the authorization by itself.
    const { advance, store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)
    advance(10 * 60_000)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('authorizes nothing when the jar is empty or the partition is gone', async () => {
    const jars: Record<string, GatewayCookie[] | null> = { [GATEWAY]: [] }
    const { store } = createStore(jars)

    await store.register(WS_URL, GATEWAY, CONSUMER)
    expect(cookieOn(store, WS_URL)).toBeUndefined()

    jars[GATEWAY] = null
    await store.register(WS_URL, GATEWAY, CONSUMER)
    expect(cookieOn(store, WS_URL)).toBeUndefined()
  })

  it('revokes the previous url when a later read finds the jar emptied', async () => {
    const jars: Record<string, GatewayCookie[] | null> = { [GATEWAY]: proxyJar }
    const { store } = createStore(jars)

    await store.register(WS_URL, GATEWAY, CONSUMER)
    jars[GATEWAY] = []
    await store.register('wss://gateway.example/api/ws?ticket=next', GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
    expect(cookieOn(store, 'wss://gateway.example/api/ws?ticket=next')).toBeUndefined()
  })

  it('reports a failed jar read, forwards nothing, and revokes prior authority', async () => {
    const onError = vi.fn()
    let reads = 0
    // One store, whose SECOND read fails: the revocation has to be observed on
    // the authority the first read established, not on a fresh empty store.
    const store = createGatewayWsCookieStore({
      readCookies: async () => {
        if (++reads > 1) {
          throw new Error('partition unavailable')
        }

        return proxyJar
      },
      resolvePartition: () => LEGACY,
      onError
    })

    await store.register(WS_URL, GATEWAY, CONSUMER)
    await store.register('wss://gateway.example/api/ws?ticket=next', GATEWAY, CONSUMER)

    expect(cookieOn(store, WS_URL)).toBeUndefined()
    expect(cookieOn(store, 'wss://gateway.example/api/ws?ticket=next')).toBeUndefined()
    expect(onError).toHaveBeenCalledWith('partition unavailable')
  })

  // A registration missing either half cannot name a socket, so it reads
  // nothing and disturbs nothing. (A sign-out missing its base url is the
  // opposite case -- it clears the whole jar -- and is covered above.)
  it('ignores a missing url or baseUrl instead of touching live authority', async () => {
    const { readCookies, store } = createStore()

    await store.register(WS_URL, GATEWAY, CONSUMER)
    readCookies.mockClear()

    await store.register('', GATEWAY, CONSUMER)
    await store.register(WS_URL, '', CONSUMER)

    expect(readCookies).not.toHaveBeenCalled()
    expect(cookieOn(store, WS_URL)).toBe(EXPECTED)
  })
})
