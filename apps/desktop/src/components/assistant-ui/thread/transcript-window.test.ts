import { describe, expect, it } from 'vitest'

import { resolveShowEarlierAction } from './transcript-window'

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
