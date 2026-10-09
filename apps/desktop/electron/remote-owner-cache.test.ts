import { describe, expect, it } from 'vitest'

import { createRemoteOwnerCache } from './remote-owner-cache'

describe('createRemoteOwnerCache (#58485)', () => {
  it('returns a fresh entry within the TTL and null past it', () => {
    let clock = 1_000
    const cache = createRemoteOwnerCache({ now: () => clock, ttlMs: 30_000 })

    expect(cache.fresh('s1')).toBeNull()
    cache.remember('s1', 'remote-a')
    expect(cache.fresh('s1')?.profile).toBe('remote-a')

    clock += 29_999
    expect(cache.fresh('s1')?.profile).toBe('remote-a')

    clock += 1
    expect(cache.fresh('s1')).toBeNull()
  })

  it('never grows past the limit: the oldest entry is FIFO-evicted', () => {
    const cache = createRemoteOwnerCache({ limit: 3 })

    for (const id of ['s1', 's2', 's3', 's4']) {
      cache.remember(id, null)
    }

    expect(cache.size()).toBe(3)
    // s1 was evicted to make room for s4; the newest entries survive.
    expect(cache.fresh('s1')).toBeNull()
    expect(cache.fresh('s4')).not.toBeNull()
    expect(cache.fresh('s3')).not.toBeNull()
  })

  it('re-remembering an existing id refreshes it without growing the map', () => {
    const cache = createRemoteOwnerCache({ limit: 2 })

    cache.remember('s1', null)
    cache.remember('s2', null)
    cache.remember('s1', 'remote-a')

    expect(cache.size()).toBe(2)
    expect(cache.fresh('s1')?.profile).toBe('remote-a')
  })
})
