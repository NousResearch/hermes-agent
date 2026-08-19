import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
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
