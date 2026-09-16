import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { stepPinnedSession } from './pinned-navigation'

const row = (id: string, lineageRootId?: string): SessionInfo =>
  ({ id, pinned: true, ...(lineageRootId ? { _lineage_root_id: lineageRootId } : {}) }) as SessionInfo

describe('stepPinnedSession', () => {
  const rows = [row('a'), row('b'), row('c')]

  it('steps forward through the visible order', () => {
    expect(stepPinnedSession(rows, 'a', 1)).toBe('b')
    expect(stepPinnedSession(rows, 'b', 1)).toBe('c')
  })

  it('steps backward through the visible order', () => {
    expect(stepPinnedSession(rows, 'c', -1)).toBe('b')
    expect(stepPinnedSession(rows, 'b', -1)).toBe('a')
  })

  it('wraps from the last row to the first and back', () => {
    expect(stepPinnedSession(rows, 'c', 1)).toBe('a')
    expect(stepPinnedSession(rows, 'a', -1)).toBe('c')
  })

  it('matches the current session by its lineage root', () => {
    const lineageRows = [row('tip-1', 'root-1'), row('tip-2', 'root-2')]

    expect(stepPinnedSession(lineageRows, 'root-1', 1)).toBe('tip-2')
    expect(stepPinnedSession(lineageRows, 'root-2', -1)).toBe('tip-1')
  })

  it('falls back to the first row (next) or the last row (previous) when nothing matches', () => {
    expect(stepPinnedSession(rows, null, 1)).toBe('a')
    expect(stepPinnedSession(rows, 'unrelated', 1)).toBe('a')
    expect(stepPinnedSession(rows, null, -1)).toBe('c')
    expect(stepPinnedSession(rows, 'unrelated', -1)).toBe('c')
  })

  it('returns null for an empty list and stays put on a single row', () => {
    expect(stepPinnedSession([], 'a', 1)).toBeNull()
    expect(stepPinnedSession([row('only')], 'only', 1)).toBe('only')
    expect(stepPinnedSession([row('only')], null, -1)).toBe('only')
  })
})
