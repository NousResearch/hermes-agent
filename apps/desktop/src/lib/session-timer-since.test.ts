import { describe, expect, it } from 'vitest'

import { nextTileSessionFocusStamp, resolveSessionTimerSince } from './session-timer-since'

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

  it('uses the tile focus stamp for a focused tile instead of DB row age (#103123)', () => {
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
})

describe('nextTileSessionFocusStamp', () => {
  it('stamps Date.now()-style focus time when a tile first gains focus', () => {
    expect(nextTileSessionFocusStamp(null, 'sess-a', false, 42)).toEqual({
      since: 42,
      storedId: 'sess-a'
    })
  })

  it('keeps the stamp while the same tile stays focused', () => {
    const previous = { since: 42, storedId: 'sess-a' }

    expect(nextTileSessionFocusStamp(previous, 'sess-a', false, 99)).toBe(previous)
  })

  it('re-stamps when focus moves to a sibling tile', () => {
    expect(
      nextTileSessionFocusStamp({ since: 42, storedId: 'sess-a' }, 'sess-b', false, 99)
    ).toEqual({ since: 99, storedId: 'sess-b' })
  })

  it('leaves the prior stamp alone while primary is focused', () => {
    const previous = { since: 42, storedId: 'sess-a' }

    expect(nextTileSessionFocusStamp(previous, 'sess-primary', true, 99)).toBe(previous)
  })
})
