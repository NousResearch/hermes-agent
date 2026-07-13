import { describe, expect, it } from 'vitest'

import { normalizeProfileKey as normalizeFromProfile } from './profile'
import { normalizeProfileKey } from './profile-key'
import {
  hasSessionIdentity,
  makeSessionIdentity,
  parseSessionIdentityKey,
  sessionIdentityEqual,
  sessionIdentityKey,
  toggleSessionIdentity
} from './session-identity'

describe('profile-key', () => {
  it('keeps the re-export source-compatible without importing session state', () => {
    expect(normalizeProfileKey(null)).toBe('default')
    expect(normalizeProfileKey('  work  ')).toBe('work')
    expect(normalizeFromProfile(undefined)).toBe('default')
    expect(normalizeFromProfile(' work ')).toBe(normalizeProfileKey('work'))
  })
})

describe('session identity', () => {
  it('normalizes construction and compares structurally', () => {
    const normalized = makeSessionIdentity(' work ', 'same-id')

    expect(normalized).toEqual({ profile: 'work', sessionId: 'same-id' })
    expect(sessionIdentityEqual(normalized, { profile: 'work', sessionId: 'same-id' })).toBe(true)
    expect(sessionIdentityEqual(normalized, { profile: 'default', sessionId: 'same-id' })).toBe(false)
  })

  it('uses collision-safe JSON tuples and parses defensively', () => {
    const left = makeSessionIdentity('pro:file', 'sess:1')
    const right = makeSessionIdentity('pro', 'file:sess:1')

    expect(sessionIdentityKey(left)).toBe(JSON.stringify(['pro:file', 'sess:1']))
    expect(sessionIdentityKey(left)).not.toBe(sessionIdentityKey(right))
    expect(parseSessionIdentityKey(sessionIdentityKey(left))).toEqual(left)
    expect(parseSessionIdentityKey('not-json')).toBeNull()
    expect(parseSessionIdentityKey(JSON.stringify(['work']))).toBeNull()
    expect(parseSessionIdentityKey(JSON.stringify([1, 'session']))).toBeNull()
  })

  it('checks and toggles identity arrays without duplicates', () => {
    const defaultSession = makeSessionIdentity('default', 'same-id')
    const workSession = makeSessionIdentity('work', 'same-id')
    const both = toggleSessionIdentity(toggleSessionIdentity([], defaultSession, true), workSession, true)

    expect(hasSessionIdentity(both, defaultSession)).toBe(true)
    expect(hasSessionIdentity(both, workSession)).toBe(true)
    expect(toggleSessionIdentity(both, defaultSession, true)).toBe(both)
    expect(toggleSessionIdentity(both, defaultSession, false)).toEqual([workSession])
  })
})
