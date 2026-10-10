import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  dispatchProfileMutationWithStartupPreference,
  prepareProfileRenameLifecycle,
  profileMutationIsLocal,
  profileRenameFromRequest,
  type ProfileRenameLifecycleDeps
} from './profile-rename-routing'

const renameRequest = {
  body: { new_name: 'renamed-profile' },
  method: 'PATCH',
  path: '/api/profiles/primary-profile'
}

function lifecycleDeps(events: string[]): ProfileRenameLifecycleDeps {
  let activeProfile = 'primary-profile'

  return {
    isValidProfileName: profile => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(profile),
    primaryProfileKey: () => 'primary-profile',
    readActiveDesktopProfile: () => activeProfile,
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
      activeProfile = profile
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

test('local pooled rename and delete update only the matching next-launch choice after success', async () => {
  const route = {
    primaryProfile: 'primary-profile',
    requestMethod: 'PATCH',
    requestPath: '/api/profiles/worker-profile'
  }

  assert.equal(profileMutationIsLocal('worker-profile', route), true)
  assert.equal(profileMutationIsLocal('worker-profile', { ...route, profileRemoteOverride: true }), false)
  assert.equal(profileMutationIsLocal('worker-profile', { ...route, primaryRemoteActive: true }), false)
  assert.equal(profileMutationIsLocal('worker-profile', { ...route, primaryRemoteActive: true, ownEntry: true }), true)
  assert.equal(profileMutationIsLocal('worker-profile', { ...route, globalRemote: true }), false)

  for (const local of [true, false]) {
    for (const method of ['PATCH', 'DELETE']) {
      for (const subsequentChoice of [null, 'another-profile']) {
        const events: string[] = []
        const deps = lifecycleDeps(events)
        const request = { ...renameRequest, method, path: '/api/profiles/worker-profile' }
        deps.writeActiveDesktopProfile('worker-profile')
        events.length = 0

        await dispatchProfileMutationWithStartupPreference(request, {
          ...deps,
          local,
          dispatch: async () => {
            const lifecycle = method === 'PATCH' ? await prepareProfileRenameLifecycle(request, deps) : null
            assert.equal(deps.readActiveDesktopProfile(), 'worker-profile')

            if (subsequentChoice) {
              deps.writeActiveDesktopProfile(subsequentChoice)
            }

            await lifecycle?.complete()

            return 'mutated'
          }
        })

        assert.equal(
          deps.readActiveDesktopProfile(),
          subsequentChoice ?? (local ? (method === 'PATCH' ? 'renamed-profile' : 'default') : 'worker-profile')
        )
        assert.equal(events.includes('teardown-primary'), false)
      }
    }
  }
})

test('failed mutations and primary rename preserve the remembered workspace independently of the live primary', async () => {
  for (const remembered of ['primary-profile', 'worker-profile']) {
    for (const fails of [false, true]) {
      const events: string[] = []
      const deps = lifecycleDeps(events)
      deps.writeActiveDesktopProfile(remembered)
      events.length = 0

      const operation = dispatchProfileMutationWithStartupPreference(renameRequest, {
        ...deps,
        local: true,
        dispatch: async () => {
          const lifecycle = await prepareProfileRenameLifecycle(renameRequest, deps)

          if (fails) {
            await lifecycle?.rollback()
            throw new Error('rename rejected')
          }

          await lifecycle?.complete()
        }
      })

      if (fails) {
        await assert.rejects(operation, /rename rejected/)
      } else {
        await operation
      }

      assert.equal(
        deps.readActiveDesktopProfile(),
        !fails && remembered === 'primary-profile' ? 'renamed-profile' : remembered
      )
    }
  }

  for (const temporaryPrimary of [false, true]) {
    const events: string[] = []
    const deps = lifecycleDeps(events)

    await assert.rejects(
      dispatchProfileMutationWithStartupPreference(
        { ...renameRequest, method: 'DELETE' },
        {
          ...deps,
          local: true,
          dispatch: async () => {
            if (temporaryPrimary) {
              deps.writeActiveDesktopProfile('default')
            }

            throw new Error('delete rejected')
          }
        }
      ),
      /delete rejected/
    )
    assert.equal(deps.readActiveDesktopProfile(), 'primary-profile')
  }
})
