import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { sameCronSignature } from './session-signatures'
import { matchesSessionTags } from './session-tags'

describe('session tag filtering', () => {
  const rows = [
    { id: 'b', tags: ['work'], pinned: true, profile: 'one' },
    { id: 'a', tags: ['home'], pinned: true, profile: 'one' },
    { id: 'c', tags: ['work'], pinned: true, profile: 'two' },
    { id: 'd', tags: [], pinned: true, profile: 'one' }
  ]

  it('ORs labels, ANDs other predicates, preserving pinned order', () => {
    expect(
      rows.filter(row => matchesSessionTags(row, ['work', 'home']) && row.profile === 'one').map(row => row.id)
    ).toEqual(['b', 'a'])
    expect(rows.filter(row => matchesSessionTags(row, [])).length).toBe(4)
    expect(matchesSessionTags({}, ['work'])).toBe(false)
  })
  it('invalidates row signatures for assignment-only changes', () => {
    expect(
      sameCronSignature(
        [{ id: 'a', tags: [] } as unknown as SessionInfo],
        [{ id: 'a', tags: ['work'] } as unknown as SessionInfo]
      )
    ).toBe(false)
  })
})
