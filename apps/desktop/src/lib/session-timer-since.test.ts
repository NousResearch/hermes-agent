import { describe, expect, it } from 'vitest'

import { resolveSessionTimerSince, tileFocusStampOnFocusChange } from './session-timer-since'

describe('resolveSessionTimerSince', () => {
  it('uses the primary focus stamp while the primary surface is focused', () => {
    expect(
      resolveSessionTimerSince({
        focusedStoredSessionId: 'sess-primary',
        primaryFocused: true,
        primarySessionStartedAt: 1_000,
        tileFocus: { since: 9_999, storedId: 'sess-tile' }
      })
    ).toBe(1_000)
  })

  it('uses the tile focus stamp instead of any other clock (#103123)', () => {
    expect(
      resolveSessionTimerSince({
        focusedStoredSessionId: 'sess-tile',
        primaryFocused: false,
        primarySessionStartedAt: 1_000,
        tileFocus: { since: 5_000, storedId: 'sess-tile' }
      })
    ).toBe(5_000)
  })

  it('hides the timer when the tile stamp belongs to a different session', () => {
    expect(
      resolveSessionTimerSince({
        focusedStoredSessionId: 'sess-b',
        primaryFocused: false,
        primarySessionStartedAt: 1_000,
        tileFocus: { since: 5_000, storedId: 'sess-a' }
      })
    ).toBeNull()
  })

  it('hides the timer when a tile is focused but not yet stamped', () => {
    expect(
      resolveSessionTimerSince({
        focusedStoredSessionId: 'sess-tile',
        primaryFocused: false,
        primarySessionStartedAt: 1_000,
        tileFocus: null
      })
    ).toBeNull()
  })
})

describe('tileFocusStampOnFocusChange', () => {
  it('stamps focus time when a tile gains focus', () => {
    expect(tileFocusStampOnFocusChange('sess-a', 'sess-primary', 42)).toEqual({
      since: 42,
      storedId: 'sess-a'
    })
  })

  it('re-stamps when focus moves to a sibling tile', () => {
    expect(tileFocusStampOnFocusChange('sess-b', 'sess-primary', 99)).toEqual({
      since: 99,
      storedId: 'sess-b'
    })
  })

  it('does not stamp while primary is focused, so a later return can re-stamp', () => {
    expect(tileFocusStampOnFocusChange('sess-primary', 'sess-primary', 99)).toBeNull()
    expect(tileFocusStampOnFocusChange(null, 'sess-primary', 99)).toBeNull()
  })
})
