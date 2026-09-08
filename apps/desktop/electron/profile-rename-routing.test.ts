import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  type ConnectionScopedProfileRenameDeps,
  dispatchConnectionScopedProfileRename,
  prepareProfileRenameLifecycle,
  profileRenameFromRequest,
  type ProfileRenameLifecycleDeps
} from './profile-rename-routing'

const renameRequest = {
  body: { new_name: 'renamed-profile' },
  method: 'PATCH',
  path: '/api/profiles/primary-profile'
}

function lifecycleDeps(events: string[]): ProfileRenameLifecycleDeps {
  return {
    isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
    primaryProfileKey: () => 'primary-profile',
    reloadPrimaryWindow: () => events.push('reload-primary-window'),
    restartPrimaryBackend: async () => {
      events.push('restart-primary-backend')
    },
    teardownPoolBackendAndWait: async profile => {
      events.push(`teardown-pool:${profile}`)
    },
    teardownPrimaryBackendAndWait: async () => {
      events.push('teardown-primary')
    },
    writeActiveDesktopProfile: profile => {
      events.push(`write-active:${profile}`)
    }
  }
}

function connectionDeps(events: string[]): ConnectionScopedProfileRenameDeps<string> {
  return {
    acquire: profile => {
      events.push(`gate:${profile}`)

      return () => events.push(`release:${profile}`)
    },
    connectionKind: () => 'local',
    dispatch: async routeProfile => {
      events.push(`dispatch:${routeProfile ?? 'primary'}`)

      return 'renamed'
    },
    isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
    prepareLocal: async request => {
      events.push(`prepare:${request.connectionId}`)

      return {
        complete: async () => {
          events.push('complete')
        },
        kind: 'pool',
        rename: { newName: 'renamed-profile', oldName: 'primary-profile' },
        rollback: async () => {
          events.push('rollback')
        },
        routeProfile: null
      }
    },
    teardownConnection: async (connectionId, profile) => {
      events.push(`teardown:${connectionId}:${profile}`)
    }
  }
}

test('profileRenameFromRequest parses string and object JSON bodies', () => {
  assert.deepEqual(profileRenameFromRequest(renameRequest), {
    newName: 'renamed-profile',
    oldName: 'primary-profile'
  })
  assert.deepEqual(profileRenameFromRequest({ ...renameRequest, body: JSON.stringify({ new_name: 'String-Body' }) }), {
    newName: 'string-body',
    oldName: 'primary-profile'
  })
})

test('profileRenameFromRequest rejects malformed and reserved rename requests', () => {
  assert.equal(profileRenameFromRequest({ ...renameRequest, method: 'DELETE' }), null)
  assert.equal(profileRenameFromRequest({ ...renameRequest, body: '{' }), null)
  assert.equal(profileRenameFromRequest({ ...renameRequest, body: { new_name: 'default' } }), null)
  assert.equal(profileRenameFromRequest({ ...renameRequest, path: '/api/profiles/default' }), null)
})

test('prepareProfileRenameLifecycle tears down a pooled backend and routes through the primary', async () => {
  const events: string[] = []

  const lifecycle = await prepareProfileRenameLifecycle(
    { ...renameRequest, path: '/api/profiles/worker-profile' },
    lifecycleDeps(events)
  )

  assert.equal(lifecycle?.kind, 'pool')
  assert.equal(lifecycle?.routeProfile, null)
  assert.deepEqual(events, ['teardown-pool:worker-profile'])

  await lifecycle?.complete()
  await lifecycle?.rollback()
  assert.deepEqual(events, ['teardown-pool:worker-profile'])
})

test('prepareProfileRenameLifecycle re-homes a renamed primary after success', async () => {
  const events: string[] = []
  const lifecycle = await prepareProfileRenameLifecycle(renameRequest, lifecycleDeps(events))

  assert.equal(lifecycle?.kind, 'primary')
  assert.equal(lifecycle?.routeProfile, null)
  assert.deepEqual(events, ['write-active:default', 'teardown-primary'])

  await lifecycle?.complete()
  assert.deepEqual(events, [
    'write-active:default',
    'teardown-primary',
    'write-active:renamed-profile',
    'teardown-primary',
    'reload-primary-window'
  ])
})

test('prepareProfileRenameLifecycle restores the original primary after failure', async () => {
  const events: string[] = []
  const lifecycle = await prepareProfileRenameLifecycle(renameRequest, lifecycleDeps(events))

  await lifecycle?.rollback()
  assert.deepEqual(events, [
    'write-active:default',
    'teardown-primary',
    'write-active:primary-profile',
    'teardown-primary',
    'restart-primary-backend'
  ])
})

test('prepareProfileRenameLifecycle restores the original primary when initial teardown fails', async () => {
  const events: string[] = []
  const deps = lifecycleDeps(events)

  deps.teardownPrimaryBackendAndWait = async () => {
    events.push('teardown-primary')
    throw new Error('teardown failed')
  }

  await assert.rejects(prepareProfileRenameLifecycle(renameRequest, deps), /teardown failed/)
  assert.deepEqual(events, [
    'write-active:default',
    'teardown-primary',
    'write-active:primary-profile',
    'restart-primary-backend'
  ])
})

test('prepareProfileRenameLifecycle ignores invalid profile names without side effects', async () => {
  const events: string[] = []

  const lifecycle = await prepareProfileRenameLifecycle(
    { ...renameRequest, body: { new_name: 'Not Valid!' } },
    lifecycleDeps(events)
  )

  assert.equal(lifecycle, null)
  assert.deepEqual(events, [])
})

test('explicit registered local rename runs the local lifecycle under one gate', async () => {
  const events: string[] = []

  const result = await dispatchConnectionScopedProfileRename(
    { ...renameRequest, connectionId: 'local', profile: 'primary-profile' },
    connectionDeps(events)
  )

  assert.equal(result, 'renamed')
  assert.deepEqual(events, ['gate:primary-profile', 'prepare:local', 'dispatch:primary', 'complete', 'release:primary-profile'])
})

test('connection-scoped rename validates its connection and old/new names before side effects', async () => {
  const events: string[] = []
  const deps = connectionDeps(events)

  await assert.rejects(dispatchConnectionScopedProfileRename(renameRequest, deps), /requires a connection/i)
  await assert.rejects(
    dispatchConnectionScopedProfileRename(
      { ...renameRequest, body: { new_name: 'not valid!' }, connectionId: 'local' },
      deps
    ),
    /invalid profile rename/i
  )
  assert.deepEqual(events, [])
})

test('explicit registered SSH rename tears down its exact source backend before owner dispatch', async () => {
  const events: string[] = []
  const deps = connectionDeps(events)
  deps.connectionKind = () => 'ssh'
  deps.prepareLocal = async () => assert.fail('SSH rename must not use the local lifecycle')

  const result = await dispatchConnectionScopedProfileRename(
    { ...renameRequest, connectionId: 'build-host', profile: 'primary-profile' },
    deps
  )

  assert.equal(result, 'renamed')
  assert.deepEqual(events, [
    'gate:primary-profile',
    'teardown:build-host:primary-profile',
    'dispatch:primary',
    'release:primary-profile'
  ])
})

test('explicit local rename rolls back before releasing its gate when dispatch fails', async () => {
  const events: string[] = []
  const deps = connectionDeps(events)

  deps.dispatch = async () => {
    events.push('dispatch:failed')
    throw new Error('rename failed')
  }

  await assert.rejects(
    dispatchConnectionScopedProfileRename(
      { ...renameRequest, connectionId: 'local', profile: 'primary-profile' },
      deps
    ),
    /rename failed/
  )
  assert.deepEqual(events, [
    'gate:primary-profile',
    'prepare:local',
    'dispatch:failed',
    'rollback',
    'release:primary-profile'
  ])
})
