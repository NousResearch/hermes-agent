import { describe, expect, it, vi } from 'vitest'

import type * as ProfileIdGuard from '../../electron/profile-id-guard'

import { requireProfileRoute } from './browser-startup-profile'

// The stored startup route takes its profile grammar from PROFILE_ID_ARG_RE (state-24, #155595).
// Swapping the owner shows the gate follows it; a re-declared copy keeps accepting `work` and fails.
vi.mock('../../electron/profile-id-guard', async importOriginal => ({
  ...(await importOriginal<typeof ProfileIdGuard>()),
  PROFILE_ID_ARG_RE: /^zz\.[a-z]+$/
}))

describe('requireProfileRoute follows PROFILE_ID_ARG_RE', () => {
  it('accepts what the owner accepts and nothing else', () => {
    expect(requireProfileRoute({ connectionId: null, profile: 'zz.custom' })).toEqual({
      connectionId: null,
      profile: 'zz.custom'
    })
    expect(() => requireProfileRoute({ connectionId: null, profile: 'work' })).toThrow('Invalid profile name.')
  })
})
