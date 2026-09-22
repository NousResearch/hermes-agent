import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import { buildAboutPanelVersionOptions } from './about-version'

describe('native About version', () => {
  it('uses the Hermes version and immutable bundle commit', () => {
    assert.deepEqual(
      buildAboutPanelVersionOptions({
        applicationVersion: '0.20.3',
        installCommit: 'a'.repeat(40)
      }),
      {
        applicationVersion: '0.20.3',
        version: 'aaaaaaa'
      }
    )
  })

  it('keeps the bundle-skew warning in secondary text', () => {
    assert.deepEqual(
      buildAboutPanelVersionOptions({
        applicationVersion: '0.20.3',
        installCommit: 'b'.repeat(40),
        bundleOutOfSync: true
      }),
      {
        applicationVersion: '0.20.3',
        credits: 'App build out of date. Update the desktop app.',
        version: 'bbbbbbb'
      }
    )
  })

  it('omits missing, malformed, and fallback build stamps', () => {
    for (const installCommit of [null, 'not-a-sha', '0'.repeat(40)]) {
      assert.deepEqual(buildAboutPanelVersionOptions({ applicationVersion: '0.20.3', installCommit }), {
        applicationVersion: '0.20.3'
      })
    }
  })

  it('marks locally modified builds', () => {
    assert.equal(
      buildAboutPanelVersionOptions({
        applicationVersion: '0.20.3',
        installCommit: 'c'.repeat(40),
        installDirty: true
      }).version,
      'ccccccc-dirty'
    )
  })
})
