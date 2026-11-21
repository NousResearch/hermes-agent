import { expect, test } from 'vitest'
import { bundleIdentity, verifyBundleStamp, verifyMacMetadata } from '../tests/install/e2e-assets/bundle-smoke-metadata.mjs'
import { buildStampPayload } from '../apps/desktop/scripts/write-build-stamp.mjs'
import { releaseVersion } from '../scripts/msix-shared.mjs'

const commit = '0123456' + 'a'.repeat(33)
const tag = 'commit.20261026T180000Z-v1.2.3'

test('a commit claim keeps the same native identity as its tagless SHA', () => {
  const identity = bundleIdentity(commit, tag)
  expect(identity).toEqual(bundleIdentity(commit))
  expect(identity.appId).toContain(commit.slice(0, 7))
  for (const invalid of ['commit.not-a-claim', `${tag} `, 'channel.20261026T180000Z-v1.2.3']) {
    expect(() => bundleIdentity(commit, invalid)).toThrow('tag')
  }
})

test.each(['darwin', 'win32'])('the real %s commit claim stamp passes fresh-install smoke but never enables updates', platform => {
  const stamp = buildStampPayload({ commit, source: 'commit-build', dirty: false, branch: null },
    { HERMES_DESKTOP_VARIANT: 'bundled', HERMES_BUILD_COMMIT: commit, HERMES_PAYLOAD_TAG: tag },
    platform, { runtime: { commands: { hermes: { executable: 'python', args: [] } } } })
  const options = { commit, tag, platform }
  expect(verifyBundleStamp(stamp, options)).toBe(releaseVersion(tag))
  for (const [key, value] of [['source', 'git'], ['branch', 'main'],
    ['updateMechanism', 'app-installer'], ['updateMechanism', 'electron-updater'],
    ['commit', 'b'.repeat(40)], ['tag', null], ['dirty', true]]) {
    expect(() => verifyBundleStamp({ ...stamp, [key]: value }, options)).toThrow(key)
  }
  for (const invalid of ['commit.not-a-claim', `${tag} `]) {
    expect(() => verifyBundleStamp({ ...stamp, tag: invalid }, { ...options, tag: invalid })).toThrow('tag')
  }
  if (platform === 'darwin') {
    const plist = { CFBundleIdentifier: bundleIdentity(commit, tag).appId,
      CFBundleShortVersionString: releaseVersion(tag), CFBundleExecutable: 'Hermes' }
    expect(verifyMacMetadata(plist, stamp, options)).toBe(releaseVersion(tag))
  }
})
