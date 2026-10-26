import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createSessionMutationClient } from '../../shared/src/session-http-mutations'

import { resolveRegistryLocalRoute } from './connection-registry'
import { dispatchRegistryApiRequest } from './registry-api-dispatch'

function fixture(globalRemote: boolean, profileRemoteOverride = false) {
  const pool = new Map<string, { mode: string; scope: string }>()
  const starts: string[] = []
  const claims: string[] = []
  const sent: string[] = []
  let beforeEnsure = () => {}

  const localRoute = (profile: string | undefined) =>
    resolveRegistryLocalRoute(profile, {
      globalRemote,
      profileRemoteOverride
    })

  const deps = {
    afterProfileRequest: () => {},
    claim: async (key: string, ensure: () => Promise<{ mode: string; scope: string }>) => {
      claims.push(key)

      return ensure()
    },
    ensure: async (id: string, profile: string | undefined, passive: boolean) => {
      beforeEnsure()
      const scope = id === 'local' ? localRoute(profile).poolKey : `conn:${id}::${profile || 'default'}`
      let backend = pool.get(scope)

      if (!backend) {
        if (passive) {
          throw new Error(`Passive read: no warm backend for "${scope}"`)
        }

        starts.push(scope)
        backend = { mode: id === 'local' ? 'local' : 'remote', scope }
        pool.set(scope, backend)
      }

      return backend
    },
    fetch: async (backend: { mode: string; scope: string }, path: string) => {
      sent.push(`${backend.scope}:${path}`)

      return { rows: ['session-1'] }
    },
    localRoute,
    path: (path: string, profile: string | undefined) => `${path}?profile=${profile}`,
    timeout: (ms: number | undefined) => ms || 1000,
    tag: (_path: string, response: { rows: string[] }, connectionId: string) => ({
      ...response,
      connectionId
    })
  }

  const dispatch = (method: string, id = 'local', profile = 'research', passive?: boolean) =>
    dispatchRegistryApiRequest({ method, path: '/api/sessions/s-1', profile, passive }, id, profile, profile, deps)

  return { claims, dispatch, pool, sent, starts, beforeBackendLookup: (fn: () => void) => { beforeEnsure = fn } }
}

test('stale local owner GET does not start a forced-local backend on a remote-primary desktop', async () => {
  const desktop = fixture(true)
  // A remote primary may already have its bare profile descriptor; that is
  // not the forced-local child owning conn:local::research.
  desktop.pool.set('research', { mode: 'remote', scope: 'research' })
  desktop.pool.set('conn:local::default', { mode: 'local', scope: 'conn:local::default' })

  await assert.rejects(desktop.dispatch('GET'), /Passive read: no warm backend for "conn:local::research"/)
  assert.deepEqual(desktop.starts, [])
  assert.deepEqual(desktop.claims, [])
  assert.deepEqual(desktop.sent, [])
})

test('an already-pooled This-device child serves a GET without changing the owner', async () => {
  const desktop = fixture(true)
  desktop.pool.set('conn:local::research', { mode: 'local', scope: 'conn:local::research' })

  assert.deepEqual(await desktop.dispatch('GET'), { rows: ['session-1'], connectionId: 'local' })
  assert.deepEqual(desktop.sent, ['conn:local::research:/api/sessions/s-1?profile=research'])
  assert.deepEqual(desktop.starts, [])
})

test('an evicted forced-local child cannot be restarted by a stale-hint GET', async () => {
  const desktop = fixture(true)
  const key = 'conn:local::research'
  desktop.pool.set(key, { mode: 'local', scope: key })
  desktop.beforeBackendLookup(() => {
    desktop.pool.delete(key)
  })
  await assert.rejects(desktop.dispatch('GET'), /Passive read: no warm backend/)
  assert.deepEqual(desktop.starts, [])
  assert.deepEqual(desktop.sent, [])
})

test('a passive hide PATCH uses a warm forced-local child but refuses cold or evicted children', async () => {
  const desktop = fixture(true)
  const key = 'conn:local::research'
  await assert.rejects(desktop.dispatch('PATCH', 'local', 'research', true), /Passive read: no warm backend/)
  desktop.pool.set(key, { mode: 'local', scope: key })
  await desktop.dispatch('PATCH', 'local', 'research', true)
  desktop.pool.delete(key)
  await assert.rejects(desktop.dispatch('PATCH', 'local', 'research', true), /Passive read: no warm backend/)
  assert.deepEqual(desktop.starts, [])
  assert.deepEqual(desktop.claims, [])

  assert.deepEqual(desktop.sent, [`${key}:/api/sessions/s-1?profile=research`])
})

test('a cold forced-local snapshot and its write use the same backend and CAS identity', async () => {
  const pool = new Map<string, { mode: string }>()
  const wire: Array<{ backend: { mode: string }; path: string; method: string; body?: unknown }> = []
  const route = (profile: string | undefined) => resolveRegistryLocalRoute(profile, {
    globalRemote: true, profileRemoteOverride: false
  })
  const dispatch = (path: string, method: string, body?: unknown, passive?: boolean) =>
    dispatchRegistryApiRequest({ path, method, body, passive }, 'local', 'research', 'research', {
      afterProfileRequest: () => {},
      claim: (_key, ensure) => ensure(),
      ensure: async (_id, profile, noSpawn) => {
        const key = route(profile).poolKey
        let backend = pool.get(key)
        if (!backend) {
          if (noSpawn) { throw new Error('Passive read: no warm backend') }
          backend = { mode: 'local' }
          pool.set(key, backend)
        }
        return backend
      },
      fetch: async (backend, requestPath, options) => {
        wire.push({ backend, path: requestPath, method: options.method || 'GET', body: options.body })
        return options.method === 'PATCH'
          ? { session_id: 'cold', ok: true, exists: true, runtime_revision: 19, runtime_generation: 4 }
          : { session_id: 'cold', ok: false, exists: true, runtime_revision: 19, runtime_generation: 4 }
      },
      localRoute: route,
      path: path => path,
      timeout: () => 1000,
      tag: (_path, response) => response
    })

  const mutate = createSessionMutationClient()
  const path = '/api/sessions/cold'
  const result = await mutate('[["local","research"],"cold","PATCH"]',
    () => dispatch(`${path}/mutation-snapshot?profile=research`, 'GET', undefined, false),
    identity => dispatch(path, 'PATCH', { archived: true, profile: 'research', ...identity }))
  assert.equal(result.ok, true)
  assert.deepEqual(wire.map(({ method, path }) => [method, path]), [
    ['GET', '/api/sessions/cold/mutation-snapshot?profile=research'], ['PATCH', '/api/sessions/cold']
  ])
  assert.equal(wire[0].backend, wire[1].backend)
  const { request_id, ...write } = wire[1].body as Record<string, unknown>
  assert.deepEqual(write, { archived: true, profile: 'research', expected_revision: 19, expected_generation: 4 })
  assert.ok(typeof request_id === 'string' && request_id.length > 0)
  assert.deepEqual([...pool.keys()], ['conn:local::research'])
})

test('local writes and deliberate dials still start a cold child; delegate and remote reads keep their routes', async () => {
  const write = fixture(true)
  await write.dispatch('PATCH')
  assert.deepEqual(write.starts, ['conn:local::research'])
  assert.deepEqual(write.claims, ['research'])

  const delegate = fixture(true, true)
  await delegate.dispatch('GET')
  assert.deepEqual(delegate.starts, ['research'])

  const localPrimary = fixture(false)
  await localPrimary.dispatch('GET')
  assert.deepEqual(localPrimary.starts, ['research'])

  const remote = fixture(true)
  await remote.dispatch('GET', 'remote-a')
  assert.deepEqual(remote.starts, ['conn:remote-a::research'])
})
