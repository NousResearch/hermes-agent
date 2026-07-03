import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { setPrimaryDesktopProfile } from './profile-set-routing'

test('setting the existing primary profile persists without teardown or reload', async () => {
  const teardownPrimaryBackendAndWait = vi.fn(async () => undefined)
  const reloadMainWindow = vi.fn()

  const result = await setPrimaryDesktopProfile('keo', {
    primaryProfileKey: () => 'keo',
    writeActiveDesktopProfile: () => 'keo',
    teardownPrimaryBackendAndWait,
    reloadMainWindow
  })

  assert.deepEqual(result, { profile: 'keo' })
  assert.equal(teardownPrimaryBackendAndWait.mock.calls.length, 0)
  assert.equal(reloadMainWindow.mock.calls.length, 0)
})

test('switching the primary profile tears down before reloading', async () => {
  const effects: string[] = []

  const result = await setPrimaryDesktopProfile('keo', {
    primaryProfileKey: () => 'default',
    writeActiveDesktopProfile: () => 'keo',
    teardownPrimaryBackendAndWait: async () => {
      effects.push('teardown')
    },
    reloadMainWindow: () => {
      effects.push('reload')
    }
  })

  assert.deepEqual(result, { profile: 'keo' })
  assert.deepEqual(effects, ['teardown', 'reload'])
})
