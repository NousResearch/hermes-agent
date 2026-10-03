// feature-flags.ts is the single resolver for which gated surfaces are on in
// this artifact. These tests pin the flag table: local models ship on the
// packaged desktop platforms (Windows, macOS) on every channel; elsewhere the
// launch argv opts in with --local.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCanaryTag, localModelsShipOn, resolveFeatureFlags } from './feature-flags'

const shipsLocalModels: boolean = localModelsShipOn(process.platform)

test('local models ship on every supported desktop platform', () => {
  assert.equal(localModelsShipOn('win32'), true)
  assert.equal(localModelsShipOn('darwin'), true)
  assert.equal(localModelsShipOn('linux'), true)
  assert.equal(localModelsShipOn('freebsd'), false)
})

test('without --local, local models follow the platform on every channel', () => {
  for (const canary of [false, true]) {
    assert.deepEqual(resolveFeatureFlags({ argv: [], canary }), { localModels: shipsLocalModels })
    assert.deepEqual(resolveFeatureFlags({ argv: ['Hermes.exe'], canary }), { localModels: shipsLocalModels })
  }
})

test('--local in argv opts into local models on any channel and platform', () => {
  assert.deepEqual(resolveFeatureFlags({ argv: ['Hermes.exe', '--local'], canary: false }), { localModels: true })
  assert.deepEqual(resolveFeatureFlags({ argv: ['--local'], canary: true }), { localModels: true })
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
