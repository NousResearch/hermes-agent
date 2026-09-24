/**
 * Behaviour tests for the desktop's portal session (client `hermes-desktop`):
 * system-browser login, single-flight refresh with rotation persisted before
 * use, invalid_grant = signed out, and the §5 agent token exchange. All I/O is
 * injected — no Electron, no sockets, no live portal.
 */

import { EventEmitter } from 'node:events'

import { expect, test } from 'vitest'

import { httpStatusError } from './api-transport'
import { NativeAuthChangedError } from './native-access-token'
import type { NativeTokenSet } from './native-oauth'
import { createPortalSession } from './portal-session'

const PORTAL = 'https://portal.example.test'
const TOKEN_URL = `${PORTAL}/api/oauth/token`

function makeFakeServerFactory(port = 51999) {
  const state: any = { handler: null, handlers: [] as any[], closed: false, pages: [] as string[] }

  const createServer: any = (handler: any) => {
    state.handler = handler
    state.handlers.push(handler)
    const server: any = new EventEmitter()
    server.listen = (_port: number, _host: string, cb: () => void) => cb()
    server.address = () => ({ address: '127.0.0.1', family: 'IPv4', port })

    server.close = () => {
      state.closed = true
    }

    return server
  }

  state.hit = (query: string, handler = state.handler) => {
    const res: any = { writeHead: () => undefined, end: (html: string) => state.pages.push(html) }
    handler({ url: `/callback?${query}` }, res)
  }

  return { createServer, state }
}

function makeStore(initial: Record<string, NativeTokenSet> = {}) {
  const map = new Map<string, NativeTokenSet>(Object.entries(initial))
  const events: string[] = []

  return {
    map,
    events,
    loadTokens: (key: string) => map.get(key) ?? null,
    storeTokens: (key: string, tokens: NativeTokenSet) => {
      events.push(`store:${tokens.refreshToken}`)
      map.set(key, tokens)
    },
    clearTokens: (key: string) => {
      events.push('clear')
      map.delete(key)
    }
  }
}

const fresh = (overrides: Partial<NativeTokenSet> = {}): NativeTokenSet => ({
  accessToken: 'AT-1',
  refreshToken: 'RT-1',
  expiresAt: 10_000,
  provider: 'hermes-desktop',
  userId: '',
  ...overrides
})

function makeSession(opts: {
  store?: ReturnType<typeof makeStore>
  postJson?: any
  now?: number
  createServer?: any
  onSessionOrgChanged?: () => void
}) {
  const store = opts.store ?? makeStore()
  const opened: string[] = []

  const session = createPortalSession({
    resolvePortalBaseUrl: () => PORTAL,
    loadTokens: store.loadTokens,
    storeTokens: store.storeTokens,
    clearTokens: store.clearTokens,
    postJson: opts.postJson ?? (async () => ({})),
    openExternal: async url => {
      opened.push(url)
    },
    createServer: opts.createServer,
    loginTimeoutMs: 5_000,
    nowSeconds: () => opts.now ?? 1_000,
    onSessionOrgChanged: opts.onSessionOrgChanged
  })

  return { session, store, opened }
}

test('login runs the §1 system-browser authorize and §2 code exchange, then stores the token set', async () => {
  const { createServer, state } = makeFakeServerFactory(53123)
  const posts: any[] = []

  const { session, store, opened } = makeSession({
    createServer,
    postJson: async (url: string, body: any) => {
      posts.push({ url, body })

      return {
        access_token: 'AT-new',
        token_type: 'Bearer',
        expires_in: 900,
        refresh_token: 'RT-new',
        scope: 'agents:read agents:connect'
      }
    }
  })

  expect(session.hasLivePortalSession()).toBe(false)
  const pending = session.login()
  await new Promise(r => setTimeout(r, 5))

  const authorize = new URL(opened[0])
  expect(`${authorize.origin}${authorize.pathname}`).toBe(`${PORTAL}/oauth/authorize`)
  expect(authorize.searchParams.get('client_id')).toBe('hermes-desktop')
  expect(authorize.searchParams.get('redirect_uri')).toBe('http://127.0.0.1:53123/callback')

  state.hit(`code=CODE-1&state=${authorize.searchParams.get('state')}`)
  await expect(pending).resolves.toEqual({ signedIn: true })

  expect(posts).toHaveLength(1)
  expect(posts[0].url).toBe(TOKEN_URL)
  expect(posts[0].body).toMatchObject({
    grant_type: 'authorization_code',
    client_id: 'hermes-desktop',
    code: 'CODE-1',
    redirect_uri: 'http://127.0.0.1:53123/callback'
  })
  expect(typeof posts[0].body.code_verifier).toBe('string')
  expect(store.map.get(PORTAL)).toMatchObject({ accessToken: 'AT-new', refreshToken: 'RT-new', expiresAt: 1_900 })
  expect(session.hasLivePortalSession()).toBe(true)
  expect(state.closed).toBe(true)
})

test('Deny in the browser (?error=access_denied) resolves as a clean cancel, never a crash or a token POST', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let posted = false

  const { session, store, opened } = makeSession({
    createServer,
    postJson: async () => {
      posted = true

      return {}
    }
  })

  const pending = session.login()
  await new Promise(r => setTimeout(r, 5))
  state.hit(`error=access_denied&state=${new URL(opened[0]).searchParams.get('state')}`)

  await expect(pending).resolves.toEqual({ signedIn: false, cancelled: true })
  expect(posted).toBe(false)
  expect(store.map.size).toBe(0)
  expect(state.pages.at(-1)).toMatch(/cancel/i)
})

test('a fresh access token is returned without touching the network', async () => {
  let posts = 0
  const { session } = makeSession({ store: makeStore({ [PORTAL]: fresh() }), postJson: async () => posts++ })

  await expect(session.getPortalAccessToken()).resolves.toBe('AT-1')
  expect(posts).toBe(0)
})

test('§3 refresh is single-flight and persists the rotated refresh token before the new access token is used', async () => {
  const store = makeStore({ [PORTAL]: fresh({ expiresAt: 1_010 }) })
  const posts: any[] = []
  let release!: () => void
  const gate = new Promise<void>(r => (release = r))

  const { session } = makeSession({
    store,
    now: 1_000,
    postJson: async (url: string, body: any) => {
      posts.push({ url, body })
      await gate

      return { access_token: 'AT-2', token_type: 'Bearer', expires_in: 900, refresh_token: 'RT-2' }
    }
  })

  const a = session.getPortalAccessToken().then(at => {
    store.events.push(`use:${at}`)

    return at
  })

  const b = session.getPortalAccessToken()
  release()

  await expect(Promise.all([a, b])).resolves.toEqual(['AT-2', 'AT-2'])
  expect(posts).toEqual([
    { url: TOKEN_URL, body: { grant_type: 'refresh_token', client_id: 'hermes-desktop', refresh_token: 'RT-1' } }
  ])
  expect(store.events).toEqual(['store:RT-2', 'use:AT-2'])
})

test('a 400 invalid_grant on refresh clears the store: the user is signed out', async () => {
  const store = makeStore({ [PORTAL]: fresh({ expiresAt: 1 }) })

  const { session } = makeSession({
    store,
    postJson: async () => {
      throw httpStatusError(400, JSON.stringify({ error: 'invalid_grant' }))
    }
  })

  await expect(session.getPortalAccessToken()).resolves.toBeNull()
  expect(store.map.size).toBe(0)
  expect(session.hasLivePortalSession()).toBe(false)
})

test('a transient refresh failure keeps the refresh token for the next attempt', async () => {
  const store = makeStore({ [PORTAL]: fresh({ expiresAt: 1 }) })

  const { session } = makeSession({
    store,
    postJson: async () => {
      throw httpStatusError(503, 'upstream down')
    }
  })

  await expect(session.getPortalAccessToken()).rejects.toMatchObject({ statusCode: 503 })
  expect(store.map.get(PORTAL)?.refreshToken).toBe('RT-1')
  expect(session.hasLivePortalSession()).toBe(true)
})

test('hasLivePortalSession: a refresh token, or an unexpired access token, is a live session', () => {
  const live = (tokens: NativeTokenSet) =>
    makeSession({ store: makeStore({ [PORTAL]: tokens }), now: 1_000 }).session.hasLivePortalSession()

  expect(live(fresh({ expiresAt: 1 }))).toBe(true)
  // No refresh token, but the access token is still good for a while.
  expect(live(fresh({ expiresAt: 5_000, refreshToken: '' }))).toBe(true)
  expect(live(fresh({ expiresAt: 1, refreshToken: '' }))).toBe(false)
  expect(makeSession({}).session.hasLivePortalSession()).toBe(false)
})

test('§5 exchange posts exactly the contract literals with the current portal access token', async () => {
  const posts: any[] = []

  const { session } = makeSession({
    store: makeStore({ [PORTAL]: fresh() }),
    postJson: async (url: string, body: any) => {
      posts.push({ url, body })

      return {
        access_token: 'AGENT-AT',
        issued_token_type: 'urn:ietf:params:oauth:token-type:access_token',
        token_type: 'Bearer',
        expires_in: 900,
        scope: 'agent_dashboard:access'
      }
    }
  })

  await expect(session.exchangeForAgent('agt_1')).resolves.toMatchObject({
    accessToken: 'AGENT-AT',
    refreshToken: '',
    expiresAt: 1_900,
    provider: 'hermes-cloud-agent',
    userId: 'agt_1'
  })
  expect(posts).toEqual([
    {
      url: TOKEN_URL,
      body: {
        grant_type: 'urn:ietf:params:oauth:grant-type:token-exchange',
        client_id: 'hermes-desktop',
        subject_token: 'AT-1',
        subject_token_type: 'urn:ietf:params:oauth:token-type:access_token',
        audience: 'agent:agt_1'
      }
    }
  ])
})

test('§5 exchange: a rejected subject token earns ONE forced portal refresh, then retries once', async () => {
  const store = makeStore({ [PORTAL]: fresh() })
  const bodies: any[] = []

  const { session } = makeSession({
    store,
    postJson: async (_url: string, body: any) => {
      bodies.push(body)

      if (body.grant_type === 'refresh_token') {
        return { access_token: 'AT-2', expires_in: 900, refresh_token: 'RT-2' }
      }

      if (body.subject_token === 'AT-1') {
        throw httpStatusError(400, JSON.stringify({ error: 'invalid_grant' }))
      }

      return { access_token: 'AGENT-AT', expires_in: 900 }
    }
  })

  await expect(session.exchangeForAgent('agt_1')).resolves.toMatchObject({ accessToken: 'AGENT-AT' })
  expect(bodies.map(b => (b.grant_type === 'refresh_token' ? 'refresh' : `exchange:${b.subject_token}`))).toEqual([
    'exchange:AT-1',
    'refresh',
    'exchange:AT-2'
  ])
  expect(store.map.get(PORTAL)?.refreshToken).toBe('RT-2')
})

test('§5 exchange: invalid_grant that survives a fresh subject token means access was lost — no loop', async () => {
  const subjects: string[] = []
  let refreshes = 0

  const { session } = makeSession({
    store: makeStore({ [PORTAL]: fresh() }),
    postJson: async (_url: string, body: any) => {
      if (body.grant_type === 'refresh_token') {
        // Every refresh rotates to a NEW token, so only the attempt bound
        // (not "the refresh returned the same token") can stop a loop.
        refreshes++

        return { access_token: `AT-${refreshes + 1}`, expires_in: 900, refresh_token: `RT-${refreshes + 1}` }
      }

      subjects.push(body.subject_token)
      throw httpStatusError(400, JSON.stringify({ error: 'invalid_grant' }))
    }
  })

  await expect(session.exchangeForAgent('agt_1')).rejects.toMatchObject({ cloudAgentAccessLost: true })
  expect(subjects).toEqual(['AT-1', 'AT-2'])
  expect(refreshes).toBe(1)
})

test('§5 exchange: invalid_target is access lost immediately, with no refresh or retry', async () => {
  const bodies: any[] = []

  const { session } = makeSession({
    store: makeStore({ [PORTAL]: fresh() }),
    postJson: async (_url: string, body: any) => {
      bodies.push(body)
      throw httpStatusError(400, JSON.stringify({ error: 'invalid_target' }))
    }
  })

  const error = await session.exchangeForAgent('agt_1').catch(e => e)
  expect(error).toMatchObject({ cloudAgentAccessLost: true })
  expect(String(error.message)).toMatch(/no longer have access/i)
  expect(bodies).toHaveLength(1)
})

test('§5 exchange without a portal session is a needsCloudLogin error', async () => {
  const { session } = makeSession({})

  await expect(session.exchangeForAgent('agt_1')).rejects.toMatchObject({ needsCloudLogin: true })
})

test('logout clears the stored portal token set', async () => {
  const store = makeStore({ [PORTAL]: fresh() })
  const { session } = makeSession({ store })

  session.logout()

  expect(store.map.size).toBe(0)
  expect(session.hasLivePortalSession()).toBe(false)
  await expect(session.getPortalAccessToken()).resolves.toBeNull()
})

const authorizeState = (url: string) => new URL(url).searchParams.get('state')!
const tick = () => new Promise(r => setTimeout(r, 5))
const codeResponse = (n: number) => ({ access_token: `AT-L${n}`, expires_in: 900, refresh_token: `RT-L${n}` })

test('Deny while already signed in keeps the existing session: { signedIn: true, cancelled: true }', async () => {
  const { createServer, state } = makeFakeServerFactory()
  const store = makeStore({ [PORTAL]: fresh() })
  const { session, opened } = makeSession({ createServer, store })

  const pending = session.login()
  await tick()
  state.hit(`error=access_denied&state=${authorizeState(opened[0])}`)

  await expect(pending).resolves.toEqual({ signedIn: true, cancelled: true })
  expect(store.map.get(PORTAL)).toEqual(fresh())
  expect(store.events).toEqual([])
})

test('logout while the browser flow is pending: the stale redirect rejects and stores nothing', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let posts = 0

  const { session, store, opened } = makeSession({
    createServer,
    postJson: async () => codeResponse(++posts)
  })

  const pending = session.login()
  await tick()
  session.logout()
  state.hit(`code=C&state=${authorizeState(opened[0])}`)

  await expect(pending).rejects.toBeInstanceOf(NativeAuthChangedError)
  expect(store.map.size).toBe(0)
  expect(store.events).toEqual(['clear'])
})

test('a second login while the first is pending: the first redirect rejects; only the newer one stores', async () => {
  const { createServer, state } = makeFakeServerFactory()
  let posts = 0

  const { session, store, opened } = makeSession({
    createServer,
    postJson: async () => codeResponse(++posts)
  })

  const first = session.login()
  await tick()
  const second = session.login()
  await tick()

  state.hit(`code=C1&state=${authorizeState(opened[0])}`, state.handlers[0])
  await expect(first).rejects.toBeInstanceOf(NativeAuthChangedError)
  expect(store.map.size).toBe(0)

  state.hit(`code=C2&state=${authorizeState(opened[1])}`, state.handlers[1])
  await expect(second).resolves.toEqual({ signedIn: true })
  expect(store.map.get(PORTAL)?.accessToken).toBe('AT-L2')
})

test('cancelLogin aborts the pending browser flow as a clean cancel and exposes no stale authorize URL', async () => {
  const { createServer, state } = makeFakeServerFactory()
  const { session, store, opened } = makeSession({ createServer, store: makeStore({ [PORTAL]: fresh() }) })

  expect(session.pendingAuthorizeUrl()).toBeNull()
  const pending = session.login()
  await tick()
  expect(session.pendingAuthorizeUrl()).toBe(opened[0])

  expect(session.cancelLogin()).toBe(true)
  await expect(pending).resolves.toEqual({ signedIn: true, cancelled: true })
  expect(state.closed).toBe(true)
  expect(session.pendingAuthorizeUrl()).toBeNull()
  expect(store.map.get(PORTAL)).toEqual(fresh())
  // Nothing pending: cancelling again is a no-op.
  expect(session.cancelLogin()).toBe(false)
})

test('m1: signing in to a DIFFERENT org reports the org change; the same org (or a first sign-in) does not', async () => {
  const jwt = (org: string) =>
    `${Buffer.from('{}').toString('base64url')}.${Buffer.from(JSON.stringify({ org_id: org })).toString('base64url')}.s`

  const run = async (previous: NativeTokenSet | null, nextOrg: string) => {
    const { createServer, state } = makeFakeServerFactory()
    let changes = 0

    const { session, opened } = makeSession({
      createServer,
      store: makeStore(previous ? { [PORTAL]: previous } : {}),
      postJson: async () => ({ access_token: jwt(nextOrg), expires_in: 900, refresh_token: 'RT-x' }),
      onSessionOrgChanged: () => changes++
    })

    const pending = session.login()
    await tick()
    state.hit(`code=C&state=${authorizeState(opened[0])}`)
    await pending

    return changes
  }

  expect(await run(fresh({ accessToken: jwt('org_a') }), 'org_b')).toBe(1)
  expect(await run(fresh({ accessToken: jwt('org_a') }), 'org_a')).toBe(0)
  expect(await run(null, 'org_b')).toBe(0)
})
