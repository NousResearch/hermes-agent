import { describe, expect, it } from 'vitest'

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
  const freshMs = 4 * 60_000
  const now = 1_000_000

  it('rewinds a keepalive-fresh backend past the fresh window so LRU eviction reclaims its slot', () => {
    const pool = new Map([
      ['default', { lastActiveAt: now }],
      ['research', { lastActiveAt: now }]
    ])

    markPoolScopeReleased(pool, 'research', now, freshMs)

    // The switched-away scope is now older than the fresh window (selectable
    // by selectRetirementCandidates / evictTo's freshness guard), while the
    // still-live scope stays fresh.
    expect(pool.get('research')!.lastActiveAt!).toBeLessThan(now - freshMs)
    expect(pool.get('default')!.lastActiveAt).toBe(now)
  })

  it('is a no-op for missing entries and never makes a backend look newer', () => {
    const staleAt = now - freshMs - 10_000

    const pool = new Map([
      ['already-stale', { lastActiveAt: staleAt }],
      ['untouched', { lastActiveAt: now }]
    ])

    markPoolScopeReleased(pool, 'gone', now, freshMs)
    markPoolScopeReleased(pool, 'already-stale', now, freshMs)

    expect(pool.get('already-stale')!.lastActiveAt).toBe(staleAt)
    expect(pool.get('untouched')!.lastActiveAt).toBe(now)
    expect(pool.has('gone')).toBe(false)
  })

  it('releases both the composite local scope and its delegated bare profile', () => {
    const pool = new Map([
      ['conn:local::research', { lastActiveAt: now }],
      ['research', { lastActiveAt: now }]
    ])

    markPoolScopeReleased(pool, 'conn:local::research', now, freshMs)

    expect(pool.get('conn:local::research')!.lastActiveAt!).toBeLessThan(now - freshMs)
    expect(pool.get('research')!.lastActiveAt!).toBeLessThan(now - freshMs)
  })
})

