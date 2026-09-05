import assert from 'node:assert/strict'

import { test } from 'vitest'

import { electronOwnedUpdateEnvironment } from './update-owner'

test('Electron-owned CLI updates carry the app-swap ownership marker', () => {
  const input = { HERMES_HOME: '/tmp/hermes', PATH: '/tmp/hermes/venv/bin' }
  const env = electronOwnedUpdateEnvironment(input)

  assert.deepEqual(env, {
    ...input,
    HERMES_DESKTOP_UPDATE_OWNER: 'electron'
  })
  assert.equal('HERMES_DESKTOP_UPDATE_OWNER' in input, false)
})
