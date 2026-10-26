import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveRegistryLocalRoute } from './connection-registry'
import { dispatchRegistryApiRequest } from './registry-api-dispatch'

function fixture(globalRemote: boolean, profileRemoteOverride = false) {
  const pool = new Map<string, { mode: string; scope: string }>()
  const starts: string[] = []
  const claims: string[] = []
  const sent: string[] = []

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
    poolKeys: () => pool.keys(),
    timeout: (ms: number | undefined) => ms || 1000,
    tag: (_path: string, response: { rows: string[] }, connectionId: string) => ({
      ...response,
      connectionId
    })
  }

  const dispatch = (method: string, id = 'local', profile = 'research', passive = false) =>
    dispatchRegistryApiRequest({ method, path: '/api/sessions/s-1', profile, passive }, id, profile, profile, deps)

  return { claims, dispatch, pool, sent, starts }
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
