import { describe, expect, it, vi } from 'vitest'

import { createPinnedSessionWriter, reconcilePinnedSessions } from './pinned-session-state'

describe('reconcilePinnedSessions', () => {
  it('bootstraps an absent backend record from legacy local pins', () => {
    expect(
      reconcilePinnedSessions(
        [' root-a ', 'root-a', 'root-b'],
        [' root-a ', 'root-a', 'root-b'],
        { exists: false, pinned_session_ids: [] }
      )
    ).toEqual({ pinnedSessionIds: ['root-a', 'root-b'], shouldPersist: true })
  })

  it('restores a durable backend record when browser storage was wiped', () => {
    expect(reconcilePinnedSessions([], [], { exists: true, pinned_session_ids: ['root-a'] })).toEqual({
      pinnedSessionIds: ['root-a'],
      shouldPersist: false
    })
  })

  it('treats an existing backend record as canonical, including an intentional empty list', () => {
    expect(reconcilePinnedSessions(['stale-local'], ['stale-local'], { exists: true, pinned_session_ids: [] })).toEqual({
      pinnedSessionIds: [],
      shouldPersist: false
    })
  })

  it('applies a pin made during recovery without discarding recovered pins', () => {
    expect(reconcilePinnedSessions([], ['new-pin'], { exists: true, pinned_session_ids: ['saved-a', 'saved-b'] })).toEqual({
      pinnedSessionIds: ['saved-a', 'saved-b', 'new-pin'],
      shouldPersist: true
    })
  })

  it('applies removals, additions, and reordering made during recovery', () => {
    expect(
      reconcilePinnedSessions(['saved-a', 'saved-b'], ['saved-b', 'new-pin'], {
        exists: true,
        pinned_session_ids: ['saved-a', 'saved-b']
      })
    ).toEqual({ pinnedSessionIds: ['saved-b', 'new-pin'], shouldPersist: true })
  })
})

describe('createPinnedSessionWriter', () => {
  it('serializes writes so a slower old value cannot overwrite the latest value', async () => {
    let releaseFirst!: () => void

    const firstWrite = new Promise<void>(resolve => {
      releaseFirst = resolve
    })

    const saved: string[][] = []

    const write = createPinnedSessionWriter(async ids => {
      saved.push(ids)

      if (saved.length === 1) {
        await firstWrite
      }
    })

    const oldWrite = write(['old'])
    const latestWrite = write(['latest'])

    await new Promise(resolve => setTimeout(resolve, 0))
    expect(saved).toEqual([['old']])
    releaseFirst()
    await Promise.all([oldWrite, latestWrite])
    expect(saved).toEqual([['old'], ['latest']])
  })

  it('retries a transient failure even when it affects the final value', async () => {
    const save = vi.fn().mockRejectedValueOnce(new Error('offline')).mockResolvedValue(undefined)
    const waitForRetry = vi.fn().mockResolvedValue(undefined)
    const write = createPinnedSessionWriter(save, waitForRetry)

    await expect(write(['latest'])).resolves.toBeUndefined()
    expect(save).toHaveBeenCalledTimes(2)
    expect(save).toHaveBeenNthCalledWith(1, ['latest'])
    expect(save).toHaveBeenNthCalledWith(2, ['latest'])
    expect(waitForRetry).toHaveBeenCalledTimes(1)
  })
})
