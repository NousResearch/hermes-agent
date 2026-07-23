import { describe, expect, it } from 'vitest'

import { reorderedPinnedSessionIds } from './layout'

describe('reorderedPinnedSessionIds', () => {
  it('persists an exact visible reorder', () => {
    expect(reorderedPinnedSessionIds(['a', 'b', 'c'], ['c', 'a', 'b'])).toEqual(['c', 'a', 'b'])
  })

  it('reorders visible pins while preserving stale or unresolved durable pins', () => {
    expect(reorderedPinnedSessionIds(['a', 'stale', 'b', 'c'], ['c', 'a', 'b'])).toEqual(['c', 'a', 'b', 'stale'])
  })

  it('ignores duplicate and unpinned ids from the rendered order', () => {
    expect(reorderedPinnedSessionIds(['a', 'b'], ['b', 'b', 'not-pinned', 'a'])).toEqual(['b', 'a'])
  })
})
