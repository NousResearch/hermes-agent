import { describe, expect, it } from 'vitest'

import { partitionIdleReapable } from './pool-reaper'

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

  it('regression (#75396): the descriptor half of a mixed pool survives an idle sweep', () => {
    // The reported failure: a remote-client desktop chatted in two profiles,
    // left one idle past POOL_IDLE_MS, and its parallel session vanished.
    // The reaper must spare the descriptor regardless of how long it idled —
    // dead remotes are the liveness revalidator's call, not the timer's.
    const now = Number(Date.now())

    const entries: Array<[string, { process: object | null; lastActiveAt: number }]> = [
      ['conn:local::default', { process: null, lastActiveAt: now - 24 * 60 * 60_000 }],
      ['conn:remote-host::inbox', { process: null, lastActiveAt: now - 60 * 60_000 }],
      ['local-pinned', { process: {}, lastActiveAt: now }]
    ]

    const { reap, sparedRemote } = partitionIdleReapable(entries, now, 10 * 60_000)

    expect(reap).toEqual([])
    expect(sparedRemote).toEqual(['conn:local::default', 'conn:remote-host::inbox'])
  })
})
