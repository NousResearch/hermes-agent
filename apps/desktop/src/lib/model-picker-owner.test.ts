import { describe, expect, it } from 'vitest'

import { resolveModelPickerOwner } from './model-picker-owner'

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
})