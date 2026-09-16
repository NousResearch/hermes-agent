import { describe, expect, it } from 'vitest'

import { resolveModelPickerOwner, resolveModelPickerProviderSetupScope } from './model-picker-owner'

describe('resolveModelPickerOwner', () => {
  it('keeps an active-A focused-B picker on the focused tile owner', () => {
    expect(
      resolveModelPickerOwner({
        ambientConnectionId: 'connection-a',
        ambientProfile: 'profile-a',
        focusedStoredSessionId: 'stored-b',
        selectedStoredSessionId: 'stored-a',
        sessionTiles: [
          {
            ownerRoute: {
              connectionId: 'connection-b',
              profile: 'desktop-b',
              targetProfile: 'backend-b'
            },
            storedSessionId: 'stored-b'
          }
        ]
      })
    ).toEqual({
      connectionId: 'connection-b',
      profile: 'backend-b',
      route: {
        connectionId: 'connection-b',
        profile: 'desktop-b',
        targetProfile: 'backend-b'
      }
    })
  })

  it('uses the ambient owner when the primary session is focused', () => {
    expect(
      resolveModelPickerOwner({
        ambientConnectionId: 'connection-a',
        ambientProfile: 'profile-a',
        focusedStoredSessionId: 'stored-a',
        selectedStoredSessionId: 'stored-a',
        sessionTiles: []
      })
    ).toEqual({ connectionId: 'connection-a', profile: 'profile-a', route: undefined })
  })

  it('preserves an ambient legacy descriptor through nested provider setup', () => {
    const legacyConnection = {
      mode: 'remote' as const,
      baseUrl: 'https://legacy.example',
      token: 'private-token',
      headers: { Authorization: 'private-header' }
    }

    const ambientScope = { connectionId: null, profile: 'profile-a', legacyConnection }
    const owner = { profile: 'profile-a' }

    expect(resolveModelPickerProviderSetupScope(owner, ambientScope)).toBe(ambientScope)
  })

  it('preserves a matching routed owner descriptor through nested provider setup', () => {
    const connectionOwner = {
      mode: 'remote' as const,
      baseUrl: 'https://registered.example',
      token: 'private-token'
    }

    const ambientScope = { connectionId: 'connection-b', profile: 'backend-b', connectionOwner }

    const owner = {
      connectionId: 'connection-b',
      profile: 'backend-b',
      route: { connectionId: 'connection-b', profile: 'desktop-b', targetProfile: 'backend-b' }
    }

    expect(resolveModelPickerProviderSetupScope(owner, ambientScope)).toBe(ambientScope)
  })
})
