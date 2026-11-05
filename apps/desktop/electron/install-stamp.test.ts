import { expect, test } from 'vitest'

import { installShape, type InstallStamp } from './install-stamp'

function stamp(payload: InstallStamp['payload']): InstallStamp {
  return {
    schemaVersion: 2,
    commit: null,
    commitDate: null,
    branch: null,
    builtAt: null,
    dirty: false,
    source: 'build',
    distribution: 'desktop-app',
    updateMechanism: 'self',
    baseVersion: null,
    displayVersion: null,
    distance: null,
    payload,
    tag: null
  }
}

test.each([
  ['bundled', 'bundled'],
  ['bootstrap', 'checkout'],
  ['light', 'light']
] as const)('installShape(%s) is %s', (payload, shape) => {
  expect(installShape(stamp(payload))).toBe(shape)
})

test('a dev run (no stamp) is a checkout', () => {
  expect(installShape(null)).toBe('checkout')
})
