import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createGlobalRemoteSessionRoute } from './global-remote-session-route'
import { fetchPrimaryProfileSessions, primaryAggregateConnectionId } from './profile-session-routing'

// Mixed remote mode (#81198): app-global remote + an explicit override on the active profile.
const globalConn = {
  authMode: 'token',
  baseUrl: 'https://global.example',
  connectionId: 'conn-global',
  headers: { 'cf-access-client-id': 'abc' }
}

const overrideConn = { baseUrl: 'https://override.example', connectionId: 'conn-override', mode: 'remote' }

function route(overrides: Record<string, unknown> = {}) {
  const ready: unknown[] = []

  const deps = {
    globalRemoteActive: () => true,
    primaryHasRemoteOverride: () => true,
    request: async (conn: { baseUrl: string }) => ({
      sessions: [{ id: `on:${conn.baseUrl}` }],
      total: 1,
      profile_totals: { default: 1 }
    }),
    requestPrimary: async () => ({ sessions: [], total: 0, profile_totals: {} }),
    resolveGlobalRemote: async () => globalConn,
    resolvePrimary: async () => overrideConn,
    waitReady: async (conn: unknown) => {
      ready.push(conn)
    },
    ...overrides
  }

  return { ready, sessions: createGlobalRemoteSessionRoute(deps as never) }
}

test('readiness sees the saved access-proxy headers of the global remote', async () => {
  const { ready, sessions } = route()

  await sessions.request('/api/profiles/sessions?profile=all', 'GET')
  assert.deepEqual(ready, [globalConn])
})

test('an offline global remote settles as a named error instead of holding the sidebar', async () => {
  const { sessions } = route({ readBudgetMs: 50, waitReady: () => new Promise(() => {}) })
  const started = Date.now()

  const result = await fetchPrimaryProfileSessions(
    new URLSearchParams({ profile: 'all' }),
    async () => {
      throw new Error('primary must not serve the aggregate in mixed mode')
    },
    { prefixForMerge: true, aggregate: sessions.fetchOptions() }
  )

  assert.ok(Date.now() - started < 2_000)
  assert.deepEqual(result.sessions, [])
  assert.match(String(result.errors?.[0]?.error), /did not respond within 50ms/)
})

test('the aggregate owner is the global remote it was read from, not the override', async () => {
  const { sessions } = route()

  assert.equal(await primaryAggregateConnectionId(sessions.aggregateBackend), 'conn-global')
  const plain = route({ primaryHasRemoteOverride: () => false }).sessions
  assert.equal(await primaryAggregateConnectionId(plain.aggregateBackend), 'conn-override')
})
