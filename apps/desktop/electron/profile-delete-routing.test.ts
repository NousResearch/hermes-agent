import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  assertLocalProfileCanStart,
  decideProfileDeleteAction,
  dispatchConnectionScopedProfileDelete,
  localProfilePoolKeys,
  ProfileDeletionGate,
  profileNameFromDeleteRequest,
  resolveRouteProfile
} from './profile-delete-routing'

// ---------------------------------------------------------------------------
// profileNameFromDeleteRequest
// ---------------------------------------------------------------------------

test('profileNameFromDeleteRequest parses a DELETE /api/profiles/<name> path', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/worker' }), 'worker')
})

test('profileNameFromDeleteRequest lowercases the profile name', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/Worker' }), 'worker')
})

test('profileNameFromDeleteRequest returns null for non-DELETE methods', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'GET', path: '/api/profiles/worker' }), null)
})

test('profileNameFromDeleteRequest returns null when the path does not match', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/sessions' }), null)
})

test('profileNameFromDeleteRequest returns null for an empty/whitespace name', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/%20' }), null)
})

test('profileNameFromDeleteRequest returns null for an undecodable path segment', () => {
  assert.equal(profileNameFromDeleteRequest({ method: 'DELETE', path: '/api/profiles/%E0%A4%A' }), null)
})

// ---------------------------------------------------------------------------
// decideProfileDeleteAction
// ---------------------------------------------------------------------------

const deps = {
  isDefaultProfile: p => p === 'default',
  isValidProfileName: p => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(p),
  primaryProfileKey: () => 'primary-profile'
}

test('decideProfileDeleteAction is a noop for the default profile', () => {
  assert.deepEqual(decideProfileDeleteAction('default', deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction is a noop for null (no profile parsed)', () => {
  assert.deepEqual(decideProfileDeleteAction(null, deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction is a noop for an invalid profile name', () => {
  assert.deepEqual(decideProfileDeleteAction('Not Valid!', deps), { action: 'noop', profile: null })
})

test('decideProfileDeleteAction tears down the primary backend for the primary profile', () => {
  assert.deepEqual(decideProfileDeleteAction('primary-profile', deps), {
    action: 'teardown-primary',
    profile: 'primary-profile'
  })
})

test('decideProfileDeleteAction tears down the pool backend for any other valid profile', () => {
  assert.deepEqual(decideProfileDeleteAction('worker', deps), { action: 'teardown-pool', profile: 'worker' })
})

// ---------------------------------------------------------------------------
// resolveRouteProfile
// ---------------------------------------------------------------------------

test('resolveRouteProfile routes to the primary backend (null) when a profile was torn down', () => {
  assert.equal(resolveRouteProfile('worker', 'other-profile'), null)
})

test('resolveRouteProfile passes the requested profile through when nothing was torn down', () => {
  assert.equal(resolveRouteProfile(null, 'other-profile'), 'other-profile')
})

test('resolveRouteProfile passes through undefined when nothing was torn down and no profile was requested', () => {
  assert.equal(resolveRouteProfile(null, undefined), undefined)
})

// ---------------------------------------------------------------------------
// ProfileDeletionGate / localProfilePoolKeys
// ---------------------------------------------------------------------------

test('ProfileDeletionGate blocks concurrent starts until deletion releases', () => {
  const gate = new ProfileDeletionGate()
  const release = gate.acquire('Selena')

  assert.equal(gate.blocks('selena'), true)
  assert.equal(gate.blocks('trina'), false)

  release()
  assert.equal(gate.blocks('selena'), false)
})

test('ProfileDeletionGate keeps overlapping deletion leases blocked', () => {
  const gate = new ProfileDeletionGate()
  const releaseFirst = gate.acquire('selena')
  const releaseSecond = gate.acquire('selena')

  releaseFirst()
  assert.equal(gate.blocks('selena'), true)

  releaseSecond()
  assert.equal(gate.blocks('selena'), false)
})

test('ProfileDeletionGate rejects a deferred start when deletion begins while it waits', async () => {
  const gate = new ProfileDeletionGate()
  let continueStart = () => undefined

  const waiting = new Promise<void>(resolve => {
    continueStart = resolve
  })

  const start = (async () => {
    await waiting
    gate.assertCanStart('selena')
  })()

  const release = gate.acquire('selena')

  continueStart()
  await assert.rejects(start, /Profile "selena" is being deleted/)
  release()
})

test('assertLocalProfileCanStart rejects a delayed retry after the profile directory is removed', () => {
  const gate = new ProfileDeletionGate()

  assert.throws(() => assertLocalProfileCanStart('selena', gate, () => false), /Profile "selena" no longer exists/)
  assert.doesNotThrow(() => assertLocalProfileCanStart('default', gate, () => false))
  assert.doesNotThrow(() => assertLocalProfileCanStart('selena', gate, profile => profile === 'selena'))
})

test('localProfilePoolKeys returns every local process scope for one profile', () => {
  assert.deepEqual(localProfilePoolKeys('Selena'), ['selena', 'conn:local::selena'])
  assert.deepEqual(localProfilePoolKeys(''), [])
})

test('resolveRouteProfile preserves a primary-backend route from another routing policy', () => {
  assert.equal(resolveRouteProfile(null, null), null)
})

test('explicit registered local DELETE holds one gate through teardown and dispatch', async () => {
  const events: string[] = []
  const gate = new ProfileDeletionGate()

  const result = await dispatchConnectionScopedProfileDelete(
    { connectionId: 'local', method: 'DELETE', path: '/api/profiles/worker', profile: 'worker' },
    {
      ...deps,
      acquire: profile => {
        events.push(`gate:${profile}`)
        const release = gate.acquire(profile)

        return () => {
          events.push(`release:${profile}`)
          release()
        }
      },
      connectionKind: () => 'local',
      dispatch: async routeProfile => {
        assert.equal(gate.blocks('worker'), true)
        events.push(`dispatch:${routeProfile ?? 'primary'}`)

        return 'deleted'
      },
      prepareLocal: async request => {
        assert.equal(gate.blocks('worker'), true)
        events.push(`prepare:${request.connectionId}:worker`)
      },
      teardownConnection: async () => assert.fail('local deletion must use prepareProfileDeleteRequest')
    }
  )

  assert.equal(result, 'deleted')
  assert.deepEqual(events, [
    'gate:worker',
    'prepare:local:worker',
    'dispatch:primary',
    'release:worker'
  ])
})

test('explicit registered SSH DELETE tears down its source backend before dispatch', async () => {
  const events: string[] = []

  const result = await dispatchConnectionScopedProfileDelete(
    { connectionId: 'build-host', method: 'DELETE', path: '/api/profiles/worker', profile: 'worker' },
    {
      ...deps,
      acquire: profile => {
        events.push(`gate:${profile}`)

        return () => events.push(`release:${profile}`)
      },
      connectionKind: () => 'ssh',
      dispatch: async routeProfile => {
        events.push(`dispatch:${routeProfile ?? 'primary'}`)

        return 'deleted'
      },
      prepareLocal: async () => assert.fail('SSH deletion must not use local profile teardown'),
      teardownConnection: async (connectionId, profile) => {
        events.push(`teardown:${connectionId}:${profile}`)
      }
    }
  )

  assert.equal(result, 'deleted')
  assert.deepEqual(events, [
    'gate:worker',
    'teardown:build-host:worker',
    'dispatch:primary',
    'release:worker'
  ])
})

test('non-identity connection DELETE tears down the logical pool and deletes the backend target', async () => {
  const events: string[] = []

  const request = {
    connectionId: 'source-a',
    method: 'DELETE',
    path: '/api/profiles/backend-worker',
    profile: 'worker'
  }

  await dispatchConnectionScopedProfileDelete(request, {
    ...deps,
    acquire: profile => {
      events.push(`gate:${profile}`)

      return () => events.push(`release:${profile}`)
    },
    connectionKind: () => 'ssh',
    dispatch: async routeProfile => {
      events.push(`delete:${request.path}:${routeProfile ?? 'primary'}`)

      return 'deleted'
    },
    prepareLocal: async () => assert.fail('remote deletion must not use local profile teardown'),
    teardownConnection: async (connectionId, profile) => {
      events.push(`teardown:${connectionId}:${profile}`)
      assert.notEqual(profile, 'backend-worker', 'must not stop the backend-target pool')
    }
  })

  assert.deepEqual(events, [
    'gate:backend-worker',
    'teardown:source-a:worker',
    'delete:/api/profiles/backend-worker:primary',
    'release:backend-worker'
  ])
})

test('connection-scoped default DELETE rejects before gate, preparation, teardown, or dispatch', async () => {
  const events: string[] = []

  await assert.rejects(
    dispatchConnectionScopedProfileDelete(
      { connectionId: 'build-host', method: 'DELETE', path: '/api/profiles/default', profile: 'default' },
      {
        acquire: profile => {
          events.push(`gate:${profile}`)

          return () => events.push(`release:${profile}`)
        },
        connectionKind: () => 'ssh',
        dispatch: async () => {
          events.push('dispatch')

          return 'deleted'
        },
        isDefaultProfile: profile => profile === 'default',
        isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
        prepareLocal: async () => {
          events.push('prepare')
        },
        teardownConnection: async () => {
          events.push('teardown')
        }
      }
    ),
    /default profile cannot be deleted/i
  )

  assert.deepEqual(events, [])
})

test('connection-scoped invalid-name DELETE rejects before gate, preparation, teardown, or dispatch', async () => {
  const events: string[] = []

  await assert.rejects(
    dispatchConnectionScopedProfileDelete(
      { connectionId: 'local', method: 'DELETE', path: '/api/profiles/bad%20name', profile: 'bad name' },
      {
        acquire: profile => {
          events.push(`gate:${profile}`)

          return () => events.push(`release:${profile}`)
        },
        connectionKind: () => 'local',
        dispatch: async () => {
          events.push('dispatch')

          return 'deleted'
        },
        isDefaultProfile: profile => profile === 'default',
        isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
        prepareLocal: async () => {
          events.push('prepare')
        },
        teardownConnection: async () => {
          events.push('teardown')
        }
      }
    ),
    /invalid profile name/i
  )

  assert.deepEqual(events, [])
})
