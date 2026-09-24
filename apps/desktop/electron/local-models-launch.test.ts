import assert from 'node:assert/strict'

import { test } from 'vitest'

import { localModelsEnabled } from './local-models-launch'

test('local models are available on Linux without a launch flag', () => {
  assert.equal(localModelsEnabled([], 'linux'), true)
})

test('the launch flag continues to enable local models on other platforms', () => {
  assert.equal(localModelsEnabled(['--local'], 'freebsd' as NodeJS.Platform), true)
})
