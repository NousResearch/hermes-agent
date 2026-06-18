const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { rebuiltMacAppCandidates } = require('./desktop-update-paths.cjs')

test('rebuilt mac app candidates include the x64 electron-builder output', () => {
  const root = '/tmp/hermes-update'
  const candidates = rebuiltMacAppCandidates(root, 'x64')

  assert.deepEqual(candidates, [
    path.join(root, 'apps', 'desktop', 'release', 'mac-x64', 'Hermes.app'),
    path.join(root, 'apps', 'desktop', 'release', 'mac-arm64', 'Hermes.app'),
    path.join(root, 'apps', 'desktop', 'release', 'mac', 'Hermes.app')
  ])
})

test('rebuilt mac app candidates prefer arm64 on Apple Silicon', () => {
  const root = '/tmp/hermes-update'
  const candidates = rebuiltMacAppCandidates(root, 'arm64')

  assert.equal(candidates[0], path.join(root, 'apps', 'desktop', 'release', 'mac-arm64', 'Hermes.app'))
  assert.ok(candidates.includes(path.join(root, 'apps', 'desktop', 'release', 'mac-x64', 'Hermes.app')))
})
