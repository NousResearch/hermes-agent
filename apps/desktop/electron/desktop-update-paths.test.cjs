const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { rebuiltMacAppCandidates } = require('./desktop-update-paths.cjs')

test('rebuilt mac app candidates include the x64 electron-builder output', () => {
  const root = '/tmp/hermes-update'
  const candidates = rebuiltMacAppCandidates(root, 'x64')

  assert.deepEqual(candidates, [
    path.join(root, 'apps', 'desktop', 'release', 'mac-x64', 'Hermes.app'),
    path.join(root, 'apps', 'desktop', 'release', 'mac', 'Hermes.app')
  ])
  assert.equal(
    candidates.includes(path.join(root, 'apps', 'desktop', 'release', 'mac-arm64', 'Hermes.app')),
    false
  )
})

test('rebuilt mac app candidates prefer arm64 on Apple Silicon', () => {
  const root = '/tmp/hermes-update'
  const candidates = rebuiltMacAppCandidates(root, 'arm64')

  assert.deepEqual(candidates, [
    path.join(root, 'apps', 'desktop', 'release', 'mac-arm64', 'Hermes.app'),
    path.join(root, 'apps', 'desktop', 'release', 'mac', 'Hermes.app')
  ])
  assert.equal(
    candidates.includes(path.join(root, 'apps', 'desktop', 'release', 'mac-x64', 'Hermes.app')),
    false
  )
})

test('rebuilt mac app candidates fall back to legacy mac output for unknown arch', () => {
  const root = '/tmp/hermes-update'
  assert.deepEqual(rebuiltMacAppCandidates(root, 'ppc'), [
    path.join(root, 'apps', 'desktop', 'release', 'mac', 'Hermes.app')
  ])
})
