import { describe, expect, it } from 'vitest'

import { selectPoolEvictions } from './pool-eviction'
import { markPoolScopeReleased, poolTouchKeys } from './pool-touch-scope'

describe('poolTouchKeys', () => {
  it('falls back from an explicit local registry scope to its delegated bare profile', () => {
    expect(poolTouchKeys('conn:local::research')).toEqual(['conn:local::research', 'research'])
  })

  it('does not alias non-local registry scopes', () => {
    expect(poolTouchKeys('conn:homelab::research')).toEqual(['conn:homelab::research'])
  })
})

describe('markPoolScopeReleased (#102187)', () => {
  it('a switched-away backend becomes LRU-evictable so the next spawn finds a slot', () => {
    const now = 1_000_000
    const freshMs = 4 * 60_000
    const live = () => ({ process: { pid: 1 }, lastActiveAt: now - 1_000 })
    const pool = new Map([['a', live()], ['b', live()], ['c', live()]])
    expect(selectPoolEvictions(pool.entries(), 2, now, freshMs)).toEqual([])
    markPoolScopeReleased(pool, 'a', now, freshMs)
    expect(selectPoolEvictions(pool.entries(), 2, now, freshMs)).toEqual(['a'])
  })
})
