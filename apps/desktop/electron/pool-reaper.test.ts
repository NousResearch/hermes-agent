import { describe, expect, it } from 'vitest'

import { countLocalBackends, partitionIdleReapable, selectLruEvictionCandidates } from './pool-reaper'

describe('partitionIdleReapable', () => {
  it('reaps local backends idle beyond the limit', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object; lastActiveAt: number }]> = [
      ['alpha', { process: {}, lastActiveAt: now - 700_000 }]
    ]

    const { reap, sparedRemote } = partitionIdleReapable(entries, now, 600_000)

    expect(reap).toEqual([{ profile: 'alpha', idleMs: 700 }])
    expect(sparedRemote).toEqual([])
  })

  it('never idle-reaps remote descriptors (no local process)', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: null; lastActiveAt: number }]> = [
      ['remote-profile', { process: null, lastActiveAt: now - 86_400_000 }]
    ]

    const { reap, sparedRemote } = partitionIdleReapable(entries, now, 600_000)

    expect(reap).toEqual([])
    expect(sparedRemote).toEqual(['remote-profile'])
  })

  it('spares local backends still within the idle window', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object; lastActiveAt: number }]> = [
      ['busy', { process: {}, lastActiveAt: now - 60_000 }]
    ]

    const { reap } = partitionIdleReapable(entries, now, 600_000)

    expect(reap).toEqual([])
  })

  it('partitions a mixed pool correctly', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object | null; lastActiveAt: number }]> = [
      ['local-idle', { process: {}, lastActiveAt: now - 900_000 }],
      ['local-active', { process: {}, lastActiveAt: now - 5_000 }],
      ['remote-idle', { process: null, lastActiveAt: now - 900_000 }]
    ]

    const { reap, sparedRemote } = partitionIdleReapable(entries, now, 600_000)

    expect(reap.map(r => r.profile)).toEqual(['local-idle'])
    expect(sparedRemote).toEqual(['remote-idle'])
  })
})

describe('selectLruEvictionCandidates', () => {
  it('excludes remote descriptors from LRU eviction', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object | null; lastActiveAt: number }]> = [
      ['remote', { process: null, lastActiveAt: 0 }],
      ['local-stale', { process: {}, lastActiveAt: now - 200_000 }],
      ['local-fresh', { process: {}, lastActiveAt: now - 10_000 }]
    ]

    expect(selectLruEvictionCandidates(entries, now, 90_000)).toEqual(['local-stale'])
  })

  it('orders candidates least-recently-used first', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object; lastActiveAt: number }]> = [
      ['newer-stale', { process: {}, lastActiveAt: now - 150_000 }],
      ['older-stale', { process: {}, lastActiveAt: now - 500_000 }]
    ]

    expect(selectLruEvictionCandidates(entries, now, 90_000)).toEqual(['older-stale', 'newer-stale'])
  })

  it('spares entries whose renderer socket is still fresh', () => {
    const now = 1_000_000
    const entries: Array<[string, { process: object; lastActiveAt: number }]> = [
      ['live', { process: {}, lastActiveAt: now - 30_000 }]
    ]

    expect(selectLruEvictionCandidates(entries, now, 90_000)).toEqual([])
  })
})

describe('countLocalBackends', () => {
  it('counts only entries with a local child process', () => {
    const entries: Array<[string, { process: object | null; lastActiveAt: number }]> = [
      ['local-a', { process: {}, lastActiveAt: 0 }],
      ['remote-a', { process: null, lastActiveAt: 0 }],
      ['local-b', { process: {}, lastActiveAt: 0 }],
      ['remote-b', { process: null, lastActiveAt: 0 }]
    ]

    expect(countLocalBackends(entries)).toBe(2)
  })

  it('regression: several remotes + one local below the local cap yields no eviction budget', () => {
    // Pool soft cap keep = 3. Total pool size 5 (over the cap), but only 1
    // entry is a local backend. The eviction budget must be localCount - keep
    // = 1 - 3 = negative → clamped by the caller to 0 → nothing is evicted.
    // Before the fix, the budget was backendPool.size - keep = 5 - 3 = 2,
    // which would wrongly evict the single still-needed local backend.
    const now = 1_000_000
    const keep = 3
    const entries: Array<[string, { process: object | null; lastActiveAt: number }]> = [
      ['local-needed', { process: {}, lastActiveAt: now - 900_000 }],
      ['remote-1', { process: null, lastActiveAt: now - 900_000 }],
      ['remote-2', { process: null, lastActiveAt: now - 800_000 }],
      ['remote-3', { process: null, lastActiveAt: now - 700_000 }],
      ['remote-4', { process: null, lastActiveAt: now - 600_000 }]
    ]

    const localCount = countLocalBackends(entries)
    const removable = localCount - Math.max(0, keep)

    expect(localCount).toBe(1)
    expect(removable).toBeLessThanOrEqual(0)

    // Even though the local backend is stale, the (clamped) budget is 0, so
    // the eviction loop in main.ts breaks immediately and evicts nothing.
    const candidates = selectLruEvictionCandidates(entries, now, 90_000)
    expect(candidates).toEqual(['local-needed']) // candidate exists…
    expect(Math.max(0, removable)).toBe(0) // …but budget is zero → spared
  })
})
