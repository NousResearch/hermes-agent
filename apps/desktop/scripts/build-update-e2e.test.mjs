import assert from 'node:assert/strict'
import { describe, test } from 'vitest'

import { makeE2EBuilderConfig } from './build-update-e2e.mjs'

describe('managed update E2E builder config', () => {
  test('isolates the proof app from the real Hermes installation', () => {
    const config = makeE2EBuilderConfig('0.17.100', 'http://127.0.0.1:47892/')

    assert.equal(config.appId, 'com.nousresearch.hermes.update-e2e')
    assert.equal(config.productName, 'Hermes Update E2E')
    assert.equal(config.executableName, 'HermesUpdateE2E')
    assert.deepEqual(config.protocols, [])
    assert.equal(config.extraMetadata.version, '0.17.100')
    assert.match(config.directories.output, /update-e2e[/\\]0\.17\.100$/)
    assert.deepEqual(config.win.target, ['nsis'])
    assert.equal(config.nsis.shortcutName, 'Hermes Update E2E')
  })

  test('can build an isolated candidate that deliberately fails startup health', () => {
    const config = makeE2EBuilderConfig('0.17.101', 'http://localhost:47892/', { failHealth: true })

    assert.equal(config.extraMetadata.hermesUpdateE2EFailHealth, true)
    assert.equal(config.extraMetadata.hermesUpdateHealthTimeoutMs, 3_000)
  })

  test('creates update metadata for a loopback-only generic feed', () => {
    const config = makeE2EBuilderConfig('0.17.101', 'http://localhost:47892/')

    assert.deepEqual(config.publish, [
      {
        provider: 'generic',
        url: 'http://localhost:47892/'
      }
    ])
  })

  test('rejects a remote or invalid feed URL', () => {
    assert.throws(() => makeE2EBuilderConfig('0.17.101', 'https://updates.example.com/'), /loopback/i)
    assert.throws(() => makeE2EBuilderConfig('0.17.101', 'file:///tmp/feed'), /loopback/i)
  })

  test('rejects malformed versions', () => {
    assert.throws(() => makeE2EBuilderConfig('latest', 'http://127.0.0.1:47892/'), /version/i)
  })
})
