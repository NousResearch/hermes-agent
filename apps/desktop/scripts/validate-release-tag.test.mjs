import assert from 'node:assert/strict'
import test from 'node:test'

import { expectedDesktopTag, validateDesktopReleaseTag } from './validate-release-tag.mjs'

test('expectedDesktopTag uses the dedicated Desktop release namespace', () => {
  assert.equal(expectedDesktopTag('0.17.0'), 'desktop-v0.17.0')
})

test('validateDesktopReleaseTag accepts only the package version tag', () => {
  assert.equal(validateDesktopReleaseTag('desktop-v0.17.0', '0.17.0'), 'desktop-v0.17.0')
  assert.throws(() => validateDesktopReleaseTag('v0.17.0', '0.17.0'), /must be desktop-v0\.17\.0/)
  assert.throws(() => validateDesktopReleaseTag('desktop-v0.18.0', '0.17.0'), /must be desktop-v0\.17\.0/)
})
