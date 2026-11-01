// feature-flags.ts is the single resolver for which gated surfaces are on in
// this artifact. These tests pin the flag table: local models ship on Windows,
// macOS and Linux x64/arm64 on every channel; elsewhere the launch argv opts in with
// --local.
import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCanaryTag, releaseTagVersion, resolveFeatureFlags } from './feature-flags'

const shipsLocalModels: boolean =
  process.platform === 'win32' ||
  process.platform === 'darwin' ||
  (process.platform === 'linux' && (process.arch === 'x64' || process.arch === 'arm64'))

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

test('a commit build claim is no canary but names its version', () => {
  assert.equal(isCanaryTag('commit.20261026T180000Z-v0.28.0'), false)
  assert.equal(releaseTagVersion('commit.20261026T180000Z-v0.28.0'), '0.28.0+commit.20261026T180000Z')
  assert.equal(releaseTagVersion('canary.20261026T180000Z-v0.28.0'), '0.28.0+canary.20261026T180000Z')
  assert.equal(releaseTagVersion('commit.20261026T180000Z-0.28.0'), null)
})

test('isCanaryTag recognizes canary stamps and rejects stable/dev tags', () => {
  assert.equal(isCanaryTag('canary.20260818T123456Z-v0.28.0'), true)
  assert.equal(isCanaryTag('v0.28.0-canary.20260818'), false)
  assert.equal(isCanaryTag('v0.28.0+canary.20260818123456'), false)
  assert.equal(isCanaryTag('v0.28.0'), false)
  assert.equal(isCanaryTag(''), false)
  assert.equal(isCanaryTag(null), false)
  assert.equal(isCanaryTag(undefined), false)
})
