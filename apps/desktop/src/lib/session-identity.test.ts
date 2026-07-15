import { describe, expect, it } from 'vitest'

import { parseSessionIdentityKey, sessionIdentityKey, sessionMatchesIdentity } from './session-identity'

describe('session identity', () => {
  it('normalizes the profile and keeps identical stored ids distinct across profiles', () => {
    expect(sessionIdentityKey(' shared-id ', ' alpha ')).toBe(sessionIdentityKey('shared-id', 'alpha'))
    expect(sessionIdentityKey('shared-id', 'alpha')).not.toBe(sessionIdentityKey('shared-id', 'beta'))
    expect(sessionIdentityKey('shared-id', null)).toBe(sessionIdentityKey('shared-id', 'default'))
  })

  it('round-trips compound keys without assuming ids are globally unique', () => {
    expect(parseSessionIdentityKey(sessionIdentityKey('shared-id', 'alpha'))).toEqual({
      profile: 'alpha',
      storedSessionId: 'shared-id'
    })
  })

  it('matches stored and lineage ids only within the owning profile', () => {
    const session = {
      id: 'runtime-id',
      profile: 'alpha',
      _lineage_root_id: 'stored-id'
    }

    expect(sessionMatchesIdentity(session, 'stored-id', 'alpha')).toBe(true)
    expect(sessionMatchesIdentity(session, 'stored-id', 'beta')).toBe(false)
    expect(sessionMatchesIdentity(session, 'runtime-id', 'alpha')).toBe(true)
  })
})
