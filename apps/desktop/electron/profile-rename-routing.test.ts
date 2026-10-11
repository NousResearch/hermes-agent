import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { assertLocalProfileCanStart, localProfilePoolKeys, ProfileDeletionGate } from './profile-delete-routing'
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
  assert.deepEqual(events, ['write-active:default', 'teardown-primary', 'teardown-pool:primary-profile'])

  await lifecycle?.complete()
  assert.deepEqual(events, [
    'write-active:default',
    'teardown-primary',
    'teardown-pool:primary-profile',
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
    'teardown-pool:primary-profile',
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
    'teardown-pool:primary-profile',
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

// ---------------------------------------------------------------------------
// dispatchConnectionScopedProfileRename (PATCH pinned to a registry connection)
// ---------------------------------------------------------------------------

const registryRename = {
  body: { new_name: 'analyst' },
  connectionId: 'local',
  method: 'PATCH',
  path: '/api/profiles/worker',
  profile: 'worker'
}

/**
 * Real gate, real lifecycle and a real profile directory; only the backend dial
 * and the server's PATCH handler are stood in. `dispatch` applies the same start
 * guard `ensureRegistryBackend` applies to the profile it dials, then renames
 * the directory the way the backend would.
 */
function registryRenameHarness(primaryProfile: string) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-registry-rename-'))
  const home = (profile: string) => path.join(root, 'profiles', profile)
  mkdirSync(home('worker'), { recursive: true })
  writeFileSync(path.join(home('worker'), 'config.yaml'), 'model: example\n')

  const gate = new ProfileDeletionGate()
  const runningPools = new Set(localProfilePoolKeys('worker'))
  const events: string[] = []
  let active = primaryProfile
  const canStart = (profile: string) => assertLocalProfileCanStart(profile, gate, key => existsSync(home(key)))

  const deps: ConnectionScopedProfileRenameDeps<{ name: string }> = {
    acquire: profile => gate.acquire(profile),
    afterResponse: () => events.push('after-response'),
    connectionKind: () => 'local',
    dispatch: async routeProfile => {
      canStart(routeProfile ?? 'default')
      assert.throws(() => canStart('worker'), /being deleted/, 'a concurrent old-name start must stay refused')
      assert.equal(runningPools.size, 0, 'every old-name pool must stop before the PATCH')
      events.push(`patch:${routeProfile ?? 'default'}`)
      renameSync(home('worker'), home('analyst'))

      return { name: 'analyst' }
    },
    isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
    logRollbackError: error => events.push(`rollback-error:${String(error)}`),
    prepareLocal: localRequest =>
      prepareProfileRenameLifecycle(localRequest, {
        isValidProfileName: deps.isValidProfileName,
        primaryProfileKey: () => active,
        reloadPrimaryWindow: () => events.push('reload'),
        restartPrimaryBackend: async () => {
          events.push(`restart:${active}`)
        },
        teardownPoolBackendAndWait: async profile => {
          for (const key of localProfilePoolKeys(profile)) {
            runningPools.delete(key)
            events.push(`stop:${key}`)
          }
        },
        teardownPrimaryBackendAndWait: async () => {
          events.push('stop:primary')
        },
        writeActiveDesktopProfile: profile => {
          active = profile
          events.push(`active:${profile}`)
        }
      }),
    teardownConnection: async (connectionId, profile) => {
      events.push(`stop:${connectionId}::${profile}`)
    }
  }

  return {
    active: () => active,
    canStart,
    cleanup: () => rmSync(root, { force: true, recursive: true }),
    deps,
    events,
    gate,
    home,
    runningPools
  }
}

test('registry-scoped local rename dispatches without dialling the profile it gates', async () => {
  const h = registryRenameHarness('default')

  try {
    assert.deepEqual(await dispatchConnectionScopedProfileRename(registryRename, h.deps), { name: 'analyst' })
    assert.deepEqual(h.events, ['stop:worker', 'stop:conn:local::worker', 'patch:default', 'after-response'])
    assert.equal(h.gate.blocks('worker'), false)
    assert.equal(readFileSync(path.join(h.home('analyst'), 'config.yaml'), 'utf8'), 'model: example\n')
    assert.throws(() => h.canStart('worker'), /no longer exists/)
  } finally {
    h.cleanup()
  }
})

test('registry-scoped rename of the primary stops every old-name backend, then re-homes', async () => {
  const h = registryRenameHarness('worker')

  try {
    assert.deepEqual(await dispatchConnectionScopedProfileRename(registryRename, h.deps), { name: 'analyst' })
    assert.equal(h.active(), 'analyst')
    assert.deepEqual(h.events, [
      'active:default',
      'stop:primary',
      'stop:worker',
      'stop:conn:local::worker',
      'patch:default',
      'after-response',
      'active:analyst',
      'stop:primary',
      'reload'
    ])
    assert.equal(h.gate.blocks('worker'), false)
  } finally {
    h.cleanup()
  }
})

test('a failed registry-scoped PATCH restores the primary and releases the gate', async () => {
  const h = registryRenameHarness('worker')
  const failure = new Error('PATCH failed')

  h.deps.dispatch = async routeProfile => {
    assert.equal(routeProfile, null)
    throw failure
  }

  try {
    await assert.rejects(dispatchConnectionScopedProfileRename(registryRename, h.deps), error => error === failure)
    assert.equal(h.active(), 'worker')
    assert.deepEqual(h.events.slice(-3), ['active:worker', 'stop:primary', 'restart:worker'])
    assert.equal(h.events.includes('after-response'), false)
    assert.equal(h.gate.blocks('worker'), false)
    assert.equal(existsSync(h.home('worker')), true)
  } finally {
    h.cleanup()
  }
})

test('a rollback failure is logged and the PATCH error still surfaces', async () => {
  const h = registryRenameHarness('worker')
  const failure = new Error('PATCH failed')
  const prepareLocal = h.deps.prepareLocal

  h.deps.dispatch = async () => {
    throw failure
  }

  h.deps.prepareLocal = async localRequest => {
    const lifecycle = await prepareLocal(localRequest)

    return lifecycle && { ...lifecycle, rollback: async () => Promise.reject(new Error('restart failed')) }
  }

  try {
    await assert.rejects(dispatchConnectionScopedProfileRename(registryRename, h.deps), error => error === failure)
    assert.equal(h.events.at(-1), 'rollback-error:Error: restart failed')
    assert.equal(h.gate.blocks('worker'), false)
  } finally {
    h.cleanup()
  }
})

test('a post-response failure never rolls back a rename the backend already made', async () => {
  const h = registryRenameHarness('worker')
  const failure = new Error('preference write failed')

  h.deps.afterResponse = () => {
    throw failure
  }

  try {
    await assert.rejects(dispatchConnectionScopedProfileRename(registryRename, h.deps), error => error === failure)
    assert.equal(h.active(), 'analyst')
    assert.equal(h.events.at(-1), 'reload')
    assert.equal(
      h.events.some(event => event.startsWith('restart:')),
      false
    )
    assert.equal(h.gate.blocks('worker'), false)
  } finally {
    h.cleanup()
  }
})

test.each(['ssh', 'remote', 'cloud'])('%s rename stops only its connection scope, not local backends', async kind => {
  const h = registryRenameHarness('worker')
  const request = { ...registryRename, connectionId: `build-${kind}`, profile: 'desktop-alias' }

  h.deps.connectionKind = connectionId => {
    assert.equal(connectionId, request.connectionId)

    return kind
  }

  h.deps.dispatch = async routeProfile => {
    assert.equal(routeProfile, null)
    assert.equal(h.gate.blocks('worker'), true)

    return { name: 'analyst' }
  }

  try {
    assert.deepEqual(await dispatchConnectionScopedProfileRename(request, h.deps), { name: 'analyst' })
    assert.deepEqual(h.events, [`stop:${request.connectionId}::desktop-alias`, 'after-response'])
    assert.equal(h.active(), 'worker')
    assert.equal(h.runningPools.size, 2)
    assert.equal(h.gate.blocks('worker'), false)
  } finally {
    h.cleanup()
  }
})

test.each([
  { ...registryRename, path: '/api/profiles/bad%20name' },
  { ...registryRename, body: { new_name: 'bad name' } }
])('an invalid registry-scoped rename rejects before gate, teardown or dispatch', async request => {
  const h = registryRenameHarness('worker')
  const acquire = h.deps.acquire

  h.deps.acquire = profile => {
    h.events.push(`gate:${profile}`)

    return acquire(profile)
  }

  try {
    await assert.rejects(dispatchConnectionScopedProfileRename(request, h.deps), /invalid profile name/i)
    assert.deepEqual(h.events, [])
    assert.equal(existsSync(h.home('worker')), true)
  } finally {
    h.cleanup()
  }
})
