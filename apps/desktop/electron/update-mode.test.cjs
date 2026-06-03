const assert = require('node:assert/strict')
const test = require('node:test')

const { shouldUseLocalUpdater } = require('./update-mode.cjs')

test('shouldUseLocalUpdater blocks local updater handoff for remote desktop connections', () => {
  assert.equal(
    shouldUseLocalUpdater({ mode: 'remote', baseUrl: 'https://hermes.example.test', source: 'settings' }),
    false
  )
})

test('shouldUseLocalUpdater allows local updater for local desktop connections', () => {
  assert.equal(shouldUseLocalUpdater({ mode: 'local', baseUrl: 'http://127.0.0.1:9120' }), true)
})

test('shouldUseLocalUpdater allows local updater when connection state is unavailable', () => {
  assert.equal(shouldUseLocalUpdater(null), true)
})
