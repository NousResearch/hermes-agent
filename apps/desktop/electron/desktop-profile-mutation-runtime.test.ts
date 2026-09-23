import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createDesktopProfileMutationRuntime } from './desktop-profile-mutation-runtime'

test('primary profile deletion persists default before tearing down both owned backends', async () => {
  const effects: string[] = []

  const runtime = createDesktopProfileMutationRuntime({
    profileNameRe: /^[a-z0-9][a-z0-9_-]{0,63}$/,
    primaryProfileKey: () => 'work',
    writeActiveDesktopProfile: (profile: string) => effects.push(`write:${profile}`),
    teardownPrimaryBackendAndWait: async () => effects.push('primary'),
    teardownPoolBackendAndWait: async (profile: string) => effects.push(`pool:${profile}`),
    getMainWindow: () => null,
    startHermes: async () => undefined
  })

  assert.equal(await runtime.prepareProfileDeleteRequest({ method: 'DELETE', path: '/api/profiles/work' }), 'work')
  assert.equal(effects[0], 'write:default')
  assert.deepEqual(new Set(effects.slice(1)), new Set(['primary', 'pool:work']))
})
