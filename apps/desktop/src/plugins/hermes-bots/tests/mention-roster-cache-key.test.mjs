import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'

// useRoster() keys its query on [...ROSTER_KEY, connectionId] — one cache
// entry per connection the window has been on. The imperative roster readers
// (the @mention completions and the mention middleware) used to call
// getQueryData(ROSTER_KEY) with the BARE key, which is an exact-key match in
// TanStack Query and therefore matched NOTHING: completions offered no
// roster handles and remote @name-device mentions passed through the
// middleware unhandled (the local profiles.list fallback only knows bare
// local names). Reproduced on a two-source install where the same profile
// name exists locally and on an SSH connection ("Vera"): typing
// `@default-vera` did nothing. These tests model the REAL cache semantics —
// entries stored under the suffixed key, exact-key reads missing them,
// prefix reads finding them — so the key-shape regression is pinned.

const pluginSource = readFileSync(new URL('../plugin.js', import.meta.url), 'utf8')

/** Faithful-enough TanStack Query v5 cache: getQueryData matches ONLY the
 *  exact key, getQueriesData prefix-matches a key family. */
function makeQueryClient() {
  const entries = new Map()
  const keyOf = key => JSON.stringify(key)

  return {
    setQueryData: (key, value) => {
      entries.set(keyOf(key), { key, value })
    },
    getQueryData: key => entries.get(keyOf(key))?.value,
    getQueriesData: ({ queryKey }) =>
      [...entries.values()]
        .filter(({ key }) => queryKey.every((part, index) => key[index] === part))
        .map(({ key, value }) => [key, value]),
    invalidateQueries: () => undefined
  }
}

// One profile name on two sources: locally and on the SSH connection "vera".
// The union roster disambiguates the remote row as @default-vera.
const ROSTER = {
  profiles: [
    { name: 'default', connectionId: 'local', connectionKind: 'local', connectionLabel: 'This device' },
    {
      name: 'default',
      handle: 'default-vera',
      connectionId: 'vera',
      connectionKind: 'ssh',
      connectionLabel: 'Vera',
      remoteSource: true,
      sourceScoped: true
    }
  ]
}

function load({ cacheKeyConnection = 'local' } = {}) {
  const registrations = []
  const delivered = []
  let submitted = false
  const atom = value => {
    let current = value

    return { get: () => current, set: next => (current = next), listen: () => () => undefined }
  }
  const jsx = (type, props = {}) => ({ type, props })
  const queryClient = makeQueryClient()

  // Exactly where useRoster writes it: suffixed with the connection id.
  queryClient.setQueryData(['hermes-bots', 'roster', cacheKeyConnection], ROSTER)

  const context = {
    atom,
    jsx,
    jsxs: jsx,
    useQuery: () => ({}),
    useValue: v => (v?.get ? v.get() : v),
    useState: v => [v, () => undefined],
    setTimeout,
    clearTimeout,
    performance,
    window: {
      setTimeout,
      clearTimeout,
      performance,
      requestAnimationFrame: () => 0,
      cancelAnimationFrame: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined
    },
    document: { getElementById: () => null, createElement: () => ({}), head: { appendChild: () => undefined } },
    queryClient,
    host: {
      state: {
        profile: { get: () => 'default', listen: () => () => undefined },
        connectionId: { get: () => 'local', listen: () => () => undefined },
        gateway: { get: () => null, listen: () => () => undefined }
      },
      request: async () => ({}),
      requestProfile: async (route, method, params) => {
        delivered.push([route?.connectionId, route?.profile, method])

        if (method === 'profiles.list') {
          return {
            profiles: [{ name: 'default', ui_meta: { 'hermes-bots': { chat: 'remote-pin' } } }]
          }
        }

        if (method === 'session.resume') {
          if (params?.omit_messages) {
            return { session_id: 'remote-1', session_key: 'remote-pin' }
          }

          // Baseline read before the submit, then the reply poll after it:
          // once the submit happened, hand back an assistant message so the
          // bounded reply poll terminates on its first pass (~2s) instead of
          // idling until the 180s timeout.
          if (!submitted) {
            return { messages: [] }
          }

          return {
            messages: [
              { role: 'user', content: 'ping' },
              { role: 'assistant', content: 'first' },
              { role: 'assistant', content: 'ok' }
            ]
          }
        }

        if (method === 'prompt.submit') {
          submitted = true
        }

        return {}
      },
      onEvent: () => () => undefined
    },
    COMPOSER_AREAS: { middleware: 'composer.middleware', atCompletions: 'composer.atCompletions' },
    PALETTE_AREA: 'palette',
    sdk: new Proxy({}, { get: () => undefined })
  }
  const code = pluginSource
    .replace(/^import \* as sdk from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import\s+\{[\s\S]*?\}\s+from '@hermes\/plugin-sdk'\r?\n/m, '')
    .replace(/^import .* from 'react'\r?\n/m, '')
    .replace(/^import .* from 'react\/jsx-runtime'\r?\n/m, '')
    .replace('export default {', 'globalThis.plugin = {')
    .concat('\nglobalThis.__botMeta = $botMeta;')
  vm.runInNewContext(code, context)
  context.__botMeta.set({})

  const ctx = {
    register: c => registrations.push(c),
    storage: { get: async () => undefined, set: async () => undefined }
  }

  try {
    context.plugin.register(ctx)
  } catch {
    /* registration walks UI surfaces the stub doesn't fully model — the
       contributions registered before any throw are what we assert on */
  }

  const completions = registrations.find(c => c.id === 'mention-completions')
  const middleware = registrations.find(c => c.id === 'mention-middleware')
  assert.ok(completions, 'mention-completions contribution registered')
  assert.ok(middleware, 'mention middleware registered')

  return { provide: completions.data.provide, handler: middleware.data.handler, delivered, queryClient }
}

test('the roster cache entry lives under the connection-suffixed key, invisible to the bare-key read', () => {
  // Documents the regression's shape: the entry useRoster writes is NOT
  // reachable through an exact getQueryData(ROSTER_KEY).
  const { queryClient } = load()

  assert.equal(queryClient.getQueryData(['hermes-bots', 'roster']), undefined)
  assert.equal(queryClient.getQueriesData({ queryKey: ['hermes-bots', 'roster'] }).length, 1)
})

test('mention completions offer remote @name-device handles from the suffixed cache entry', () => {
  const { provide } = load()
  const inserts = provide('').map(item => item.insert)

  assert.ok(inserts.includes('@default-vera'), `expected @default-vera in ${JSON.stringify(inserts)}`)
})

test('the middleware identifies a remote @name-device mention without delivering (identification-only)', async () => {
  const { handler, delivered } = load()
  const result = await handler({ text: '@default-vera what is the disk space on the server?' })

  assert.match(result.text, /@mentions resolved from the Bot Mode roster/)
  assert.match(result.text, /@default-vera/)
  // No CLI handoff is composed and the renderer performs NO delivery — the
  // agent owns messaging via its message_agent tool.
  assert.doesNotMatch(result.text, /hermes -p '?default/)
  await new Promise(resolve => setTimeout(resolve, 0))
  assert.equal(delivered.length, 0, 'middleware must not deliver over Connections')
})

test('a roster cached under another connection id still resolves (fallback entry)', () => {
  // The window moved connections; the stale entry is the only one cached.
  const { provide } = load({ cacheKeyConnection: 'vera' })
  const inserts = provide('').map(item => item.insert)

  assert.ok(inserts.includes('@default-vera'), `expected @default-vera in ${JSON.stringify(inserts)}`)
})
