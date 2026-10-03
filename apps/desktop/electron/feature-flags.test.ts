// feature-flags.ts is the single resolver for which gated surfaces are on in
// this artifact. These tests pin the flag table: local models ship on every
// packaged desktop platform (Windows, macOS, Linux) on every channel; the
// launch argv can still force the flag on for unpackaged/dev launches.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCanaryTag, resolveFeatureFlags } from './feature-flags'

const PACKAGED_PLATFORMS = ['win32', 'darwin', 'linux'] as const

test('local models ship on every packaged desktop platform without --local', () => {
  for (const platform of PACKAGED_PLATFORMS) {
    for (const canary of [false, true]) {
      assert.deepEqual(resolveFeatureFlags({ argv: [], canary, platform }), { localModels: true }, platform)
      assert.deepEqual(
        resolveFeatureFlags({ argv: ['Hermes.exe'], canary, platform }),
        { localModels: true },
        platform
      )
    }
  }
})

test('an unpackaged platform keeps local models opt-in via --local', () => {
  for (const canary of [false, true]) {
    assert.deepEqual(resolveFeatureFlags({ argv: [], canary, platform: 'freebsd' }), { localModels: false })
    assert.deepEqual(resolveFeatureFlags({ argv: ['--local'], canary, platform: 'freebsd' }), { localModels: true })
  }
})

test('isCanaryTag recognizes canary stamps and rejects stable/dev tags', () => {
  assert.equal(isCanaryTag('v0.28.0+canary.20260818T123456Z'), true)
  assert.equal(isCanaryTag('v0.28.0-canary.20260818'), false)
  assert.equal(isCanaryTag('v0.28.0+canary.20260818123456'), false)
  assert.equal(isCanaryTag('v0.28.0'), false)
  assert.equal(isCanaryTag(''), false)
  assert.equal(isCanaryTag(null), false)
  assert.equal(isCanaryTag(undefined), false)
})
