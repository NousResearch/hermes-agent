import { describe, expect, it } from 'vitest'

import { nextPinnedSession } from './pinned-session-navigation'

const rows = [
  { id: 'one', _lineage_root_id: null },
  { id: 'two-tip', _lineage_root_id: 'two-root' },
  { id: 'three', _lineage_root_id: null }
]

describe('nextPinnedSession', () => {
  it('walks the visible order and wraps in both directions', () => {
    expect(nextPinnedSession(rows, 'one', 1)?.id).toBe('two-tip')
    expect(nextPinnedSession(rows, 'three', 1)?.id).toBe('one')
    expect(nextPinnedSession(rows, 'one', -1)?.id).toBe('three')
  })

  it('matches a selected lineage root to its live pinned row', () => {
    expect(nextPinnedSession(rows, 'two-root', 1)?.id).toBe('three')
  })

  it('enters from the matching edge when the active session is not pinned', () => {
    expect(nextPinnedSession(rows, 'other', 1)?.id).toBe('one')
    expect(nextPinnedSession(rows, 'other', -1)?.id).toBe('three')
  })

  it('is a no-op for an empty visible pinned list', () => {
    expect(nextPinnedSession([], 'one', 1)).toBeNull()
  })
})
