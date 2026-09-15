// feature-flags.ts is the single resolver for which gated surfaces are on in
// this artifact. These tests pin the flag table: launch argv opts in on any
// channel, canary builds ride along by default, stable builds stay strict.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCanaryTag, resolveFeatureFlags } from './feature-flags'

test('stable builds need the --local launch flag for local models', () => {
  assert.deepEqual(resolveFeatureFlags({ argv: [], canary: false }), { localModels: false })
  assert.deepEqual(resolveFeatureFlags({ argv: ['Hermes.exe'], canary: false }), { localModels: false })
})

test('--local in argv opts into local models on any channel', () => {
  assert.deepEqual(resolveFeatureFlags({ argv: ['Hermes.exe', '--local'], canary: false }), { localModels: true })
  assert.deepEqual(resolveFeatureFlags({ argv: ['--local'], canary: true }), { localModels: true })
})

test('canary builds get local models by default, no flag needed', () => {
  assert.deepEqual(resolveFeatureFlags({ argv: [], canary: true }), { localModels: true })
  assert.deepEqual(resolveFeatureFlags({ argv: ['Hermes.exe'], canary: true }), { localModels: true })
})

test('isCanaryTag recognizes canary stamps and rejects stable/dev tags', () => {
  assert.equal(isCanaryTag('v0.28.0-canary.20260818'), true)
  assert.equal(isCanaryTag('v0.28.0-canary.20260818123456'), true)
  assert.equal(isCanaryTag('v0.28.0'), false)
  assert.equal(isCanaryTag(''), false)
  assert.equal(isCanaryTag(null), false)
  assert.equal(isCanaryTag(undefined), false)
})
