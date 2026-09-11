import { describe, expect, it } from 'vitest'

import { resolveShowEarlierAction, shouldAutoShowEarlier } from './transcript-window'

describe('resolveShowEarlierAction', () => {
  it('spends the already-materialized DOM page first', () => {
    expect(resolveShowEarlierAction(3, true)).toBe('dom')
    expect(resolveShowEarlierAction(3, false)).toBe('dom')
  })

  it('expands the store window once the DOM page is exhausted', () => {
    expect(resolveShowEarlierAction(0, true)).toBe('window')
  })

  it('is a no-op when neither DOM nor store has older content', () => {
    expect(resolveShowEarlierAction(0, false)).toBe(null)
  })
})

describe('shouldAutoShowEarlier', () => {
  it('pages older content after an upward wheel at the clamped top edge', () => {
    expect(
      shouldAutoShowEarlier({
        atBottom: false,
        direction: 'up',
        hasOlderContent: true,
        loadSettled: true,
        restorePending: false,
        scrollTop: 0
      })
    ).toBe(true)
  })
})
