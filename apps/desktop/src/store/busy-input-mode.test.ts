import { describe, expect, it } from 'vitest'

import { busyInputOwnerKey, normalizeBusyInputMode } from './busy-input-mode'

describe('busy input configuration', () => {
  it.each([undefined, null, '', 'invalid', false, 42])('keeps upstream interrupt default for %s', value => {
    expect(normalizeBusyInputMode(value)).toBe('interrupt')
  })
  it.each(['steer', 'interrupt', 'queue'] as const)('normalizes %s like the gateway', value => {
    expect(normalizeBusyInputMode(` ${value.toUpperCase()} `)).toBe(value)
  })
  it('does not share values across same-named profiles on different backends', () => {
    expect(busyInputOwnerKey('a', 'builder')).not.toBe(busyInputOwnerKey('b', 'builder'))
    expect(busyInputOwnerKey('a', 'builder')).not.toBe(busyInputOwnerKey('a', 'default'))
  })
})
