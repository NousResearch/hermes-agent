import { expect, test } from 'vitest'
import { bundleIdentity, verifyBundleStamp } from '../tests/install/e2e-assets/bundle-smoke-metadata.mjs'

const commit = 'a'.repeat(40)
const tag = 'channel.20261028T120000Z-v1.2.3'

test('a channel claim cannot fall back to the ordinary bundled identity without its request', () => {
  expect(() => bundleIdentity(commit, tag)).toThrow('channel request')
})

test.each(['darwin', 'win32'])('a %s channel claim cannot admit an unbound release stamp', platform => {
  const stamp = { commit, tag, source: 'git', dirty: false, branch: 'main',
    payload: 'bundled', distribution: 'desktop-app',
    updateMechanism: platform === 'darwin' ? 'electron-updater' : 'app-installer' }
  expect(() => verifyBundleStamp(stamp, { commit, tag, platform })).toThrow('channel request')
})
